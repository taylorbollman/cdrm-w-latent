"""Exclusive volatile checkpoint staging with persistent verified-publication records.

The distributed saver and cloud retainer remain unchanged. Only this segment's
registered, committed, fully retained checkpoint directories can be pruned.
This helper never contacts storage, loads tensors, or scans historical roots.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from scripts import olmo_campaign_execution_restore as authority

ROOT=Path(__file__).resolve().parents[1]
SSD_MOUNT=Path('/mnt/localssd')
SSD_BASE=SSD_MOUNT/'cdrm-checkpoints'
SCHEMA='olmo-campaign-ssd-journal-v1'
MARKER='ssd-ownership.json'
JOURNAL='ssd-journal.json'
FILES={'state.pt','manifest.json'}
NAME=re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')


class StoragePruneError(RuntimeError):
    """New cloud checkpoint remains published; local cleanup failed closed."""


def _absolute(path):
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Require an absolute path without traversal')
    for candidate in (path,*path.parents):
        if candidate.is_symlink():raise ValueError('Symlink storage paths are forbidden')
    return path


def validate_storage_paths(checkpoint_root,evidence_dir,*,ssd_base=SSD_BASE,
                           persistent_root=ROOT,mount_root=SSD_MOUNT,mount_check=os.path.ismount):
    """Read-only CLI preflight; test roots are explicitly injected by CPU tests."""
    root,evidence,base,persistent,mount=map(_absolute,
        (checkpoint_root,evidence_dir,ssd_base,persistent_root,mount_root))
    if not mount.is_dir() or not mount_check(mount):
        raise ValueError('SSD mount is absent; never stage on a bare boot-disk fallback folder')
    if not base.is_relative_to(mount) or not root.is_relative_to(base):
        raise ValueError('Checkpoint staging must be below the explicit SSD checkpoint base')
    parts=root.relative_to(base).parts
    if len(parts)!=2 or any(NAME.fullmatch(part) is None for part in parts):
        raise ValueError('Require SSD base/<namespace>/<segment> exactly')
    if not evidence.is_relative_to(persistent) or evidence==persistent or evidence.is_relative_to(mount):
        raise ValueError('Small evidence must stay in the persistent project, outside SSD')
    ancestor=root
    while not ancestor.exists():ancestor=ancestor.parent
    if ancestor.stat().st_dev!=mount.stat().st_dev:
        raise ValueError('Checkpoint staging crosses a different filesystem')
    if root.exists() and not root.is_dir():raise ValueError('Checkpoint root is not a directory')
    if evidence.exists() and not evidence.is_dir():raise ValueError('Evidence root is not a directory')
    return {'checkpoint_root':str(root),'evidence_dir':str(evidence),'ssd_mount':str(mount),
            'namespace':parts[0],'segment':parts[1],'durability':'volatile SSD; persistent verified-GCS receipts'}


def _encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def _fsync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)


def _write_durable(path,value,*,once=False):
    """Sync bytes and directory entry before any destructive follow-up."""
    path=_absolute(path);raw=_encoded(value)
    if path.exists() and once:
        if not path.is_file() or path.read_bytes()!=raw:raise FileExistsError('Immutable record differs')
        return hashlib.sha256(raw).hexdigest()
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name);stream.write(raw);stream.flush();os.fsync(stream.fileno())
    try:
        if once:
            os.link(temporary,path);temporary.unlink()
        else:os.replace(temporary,path)
        _fsync_directory(path.parent)
    finally:temporary.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


def _file_pins(path):
    path=_absolute(path)
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1:
            raise ValueError('Only exclusively owned regular checkpoint files are permitted')
        sha,md5=hashlib.sha256(),hashlib.md5();size=0
        while chunk:=os.read(fd,8*1024**2):sha.update(chunk);md5.update(chunk);size+=len(chunk)
        after=os.fstat(fd)
        fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
        if fields(before)!=fields(after) or fields(after)!=fields(path.stat(follow_symlinks=False)):
            raise ValueError('Checkpoint file changed while verifying')
        return {'size_bytes':size,'sha256':sha.hexdigest(),
                'md5_base64':base64.b64encode(md5.digest()).decode()},fields(after)
    finally:os.close(fd)


class SSDCheckpointStorage:
    @classmethod
    def create(cls,checkpoint_root,evidence_dir,*,execution_identity_sha256,storage_prefix,
               keep_local_completed=2,resume_source=None,ssd_base=SSD_BASE,persistent_root=ROOT,
               mount_root=SSD_MOUNT,mount_check=os.path.ismount):
        authority.pin(execution_identity_sha256)
        if type(keep_local_completed) is not int or not 2<=keep_local_completed<=32:
            raise ValueError('Require 2..32 local completed checkpoints including recovery headroom')
        # Same strict URI parser as recovery; no network operation occurs here.
        if (not isinstance(storage_prefix,str) or not storage_prefix.startswith(authority.REMOTE_ROOT)
                or any(p in ('','.','..') for p in storage_prefix[len(authority.REMOTE_ROOT):].split('/'))
                or any(c in storage_prefix for c in ('?','#','%'))):
            raise ValueError('Require an explicit fresh fast-chunks project prefix')
        checks={'ssd_base':ssd_base,'persistent_root':persistent_root,'mount_root':mount_root,'mount_check':mount_check}
        paths=validate_storage_paths(checkpoint_root,evidence_dir,**checks)
        obj=cls();obj.root=Path(paths['checkpoint_root']);obj.evidence=Path(paths['evidence_dir'])
        if obj.root.exists():raise FileExistsError('Use a fresh exclusively owned SSD segment root')
        if not obj.evidence.is_dir():raise ValueError('Create the persistent stage evidence directory first')
        obj.checks=checks;obj.prefix=storage_prefix;obj.identity=execution_identity_sha256
        obj.keep=keep_local_completed
        obj.resume_source=None if resume_source is None else str(_absolute(resume_source))
        obj.receipts=obj.evidence/'checkpoint-publications';obj.journal_path=obj.evidence/JOURNAL
        if obj.receipts.exists() or obj.journal_path.exists() or (obj.evidence/'latest-checkpoint.json').exists():
            raise FileExistsError('Use fresh persistent publication records')
        obj.root.parent.mkdir(parents=True,exist_ok=True)
        validate_storage_paths(obj.root,obj.evidence,**checks)
        obj.root.mkdir();_fsync_directory(obj.root.parent)
        obj.receipts.mkdir();_fsync_directory(obj.evidence)
        obj.ownership={'schema':'olmo-campaign-ssd-ownership-v1',**paths,
            'execution_identity_sha256':obj.identity,'storage_prefix':storage_prefix,
            'keep_local_completed':obj.keep,'resume_source':obj.resume_source}
        obj.marker_sha=_write_durable(obj.root/MARKER,obj.ownership,once=True)
        obj.root_stat=obj._directory_id(obj.root);obj.evidence_stat=obj._directory_id(obj.evidence)
        obj.receipts_stat=obj._directory_id(obj.receipts)
        obj.journal={'schema':SCHEMA,'ownership':obj.ownership,'destinations':{},'published':[],'prune_operations':[]}
        obj.journal_sha=_write_durable(obj.journal_path,obj.journal,once=True)
        obj.latest_sha=None
        return obj

    @staticmethod
    def _directory_id(path):
        value=path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(value.st_mode):raise ValueError('Owned path stopped being a directory')
        return value.st_dev,value.st_ino

    def validate(self):
        validate_storage_paths(self.root,self.evidence,**self.checks)
        for path in (self.root/MARKER,self.journal_path,self.receipts,self.evidence/'latest-checkpoint.json'):_absolute(path)
        if (self._directory_id(self.root)!=self.root_stat or self._directory_id(self.evidence)!=self.evidence_stat
                or self._directory_id(self.receipts)!=self.receipts_stat
                or authority.sha(self.root/MARKER)!=self.marker_sha
                or authority.sha(self.journal_path)!=self.journal_sha):
            raise ValueError('SSD ownership path/marker or persistent journal changed')
        latest=self.evidence/'latest-checkpoint.json'
        if (latest.exists() if self.latest_sha is None else
                not latest.is_file() or authority.sha(latest)!=self.latest_sha):
            raise ValueError('Persistent latest-checkpoint authority changed')
        return {'ownership_unchanged':True,'journal_sha256':self.journal_sha}

    def _commit(self):
        self.journal_sha=_write_durable(self.journal_path,self.journal)

    def destination(self,update):
        self.validate()
        if type(update) is not int or not 0<=update<=999999:raise ValueError('Invalid checkpoint update')
        path=self.root/f'update-{update:06d}'
        if path.exists() or path.is_symlink() or str(update) in self.journal['destinations']:
            raise FileExistsError('Checkpoint destination is not fresh')
        if str(path)==self.resume_source:raise ValueError('Resume source is protected')
        self.journal['destinations'][str(update)]=str(path);self._commit()
        return path

    def validate_local(self,receipt):
        self.validate()
        manifest={key:receipt[key] for key in authority.MANIFEST_FIELDS}
        identity=authority.committed_metadata(manifest)
        update=manifest['counters']['optimizer_updates'];path=_absolute(receipt['directory'])
        if (identity['sha256']!=self.identity or path!=self.root/f'update-{update:06d}'
                or self.journal['destinations'].get(str(update))!=str(path) or str(path)==self.resume_source):
            raise ValueError('Checkpoint is not registered to this execution/segment')
        if self._directory_id(path)[0]!=self.root_stat[0] or set(os.listdir(path))!=FILES:
            raise ValueError('Checkpoint directory has unexpected content or filesystem')
        mp,mi=_file_pins(path/'manifest.json');sp,si=_file_pins(path/'state.pt')
        if (mp['sha256']!=receipt['manifest_sha256'] or sp['sha256']!=manifest['state']['sha256']
                or sp['size_bytes']!=manifest['state']['size_bytes']
                or json.loads((path/'manifest.json').read_text())!=manifest):
            raise ValueError('Local checkpoint differs from its complete committed receipt')
        return {'update':update,'directory':str(path),'manifest':mp,'state':sp,
                'directory_id':self._directory_id(path),'file_ids':{'manifest.json':mi,'state.pt':si}}

    def _retained(self,receipt):
        manifest,objects,identity=authority.publication_metadata(receipt)
        local=self.validate_local(receipt)
        prefix=f'{self.prefix}/update-{local["update"]:06d}/'
        if any(objects[name]['uri']!=prefix+name for name in FILES):
            raise ValueError('Retained checkpoint belongs to another segment prefix')
        if any(objects[name][key]!=local['manifest' if name=='manifest.json' else 'state'][key]
               for name in FILES for key in ('size_bytes','sha256','md5_base64')):
            raise ValueError('Retained object bytes do not match owned local checkpoint')
        return local

    def _prune_one(self,entry):
        self.validate();path=Path(entry['directory'])
        receipt_path=self.evidence/entry['receipt_path']
        if authority.sha(_absolute(receipt_path))!=entry['receipt_sha256']:
            raise ValueError('Persistent retained receipt changed before pruning')
        receipt=json.loads(receipt_path.read_text());checked=self._retained(receipt)
        # No recursive deletion: only the exact two files authenticated above.
        root_fd=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        fd=None
        try:
            fd=os.open(path.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=root_fd)
            st=os.fstat(fd)
            if (st.st_dev,st.st_ino)!=checked['directory_id'] or set(os.listdir(fd))!=FILES:
                raise ValueError('Checkpoint directory changed before deletion')
            for name in ('manifest.json','state.pt'):
                st=os.stat(name,dir_fd=fd,follow_symlinks=False)
                observed=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_nlink)
                if observed!=checked['file_ids'][name]:raise ValueError('Checkpoint file changed before deletion')
            os.unlink('manifest.json',dir_fd=fd);os.fsync(fd)
            os.unlink('state.pt',dir_fd=fd);os.fsync(fd)
            os.rmdir(path.name,dir_fd=root_fd);os.fsync(root_fd)
        finally:
            if fd is not None:os.close(fd)
            os.close(root_fd)

    def publish_retained(self,receipt):
        local=self._retained(receipt);update=local['update']
        if self.journal['published'] and update<=self.journal['published'][-1]['update']:
            raise ValueError('Publication must advance this segment strictly')
        path=self.receipts/f'update-{update:06d}.json'
        pin=_write_durable(path,receipt,once=True)
        entry={'update':update,'directory':local['directory'],'receipt_path':str(path.relative_to(self.evidence)),
               'receipt_sha256':pin,'local_status':'retained'}
        self.journal['published'].append(entry);self._commit()
        # This durable latest authority must precede any pruning intent/deletion.
        self.latest_sha=_write_durable(self.evidence/'latest-checkpoint.json',receipt)
        active=[row for row in self.journal['published'] if row['local_status']=='retained']
        candidates=active[:-self.keep]
        for old in candidates:
            if old['directory']==self.resume_source:continue
            operation={'update':old['update'],'directory':old['directory'],
                'receipt_sha256':old['receipt_sha256'],'replaced_by_update':update,'status':'planned'}
            self.journal['prune_operations'].append(operation);self._commit()
            try:self._prune_one(old)
            except Exception as error:
                operation.update(status='failed',error=f'{type(error).__name__}: {error}')
                old['local_status']='prune_failed';self._commit()
                raise StoragePruneError('Verified checkpoint remains published; local pruning failed, inspect persistent journal') from error
            operation['status']='completed';old['local_status']='pruned';self._commit()
        return {'status':'published_and_local_retention_applied','update':update,
            'receipt_path':str(path),'receipt_sha256':pin,'journal_path':str(self.journal_path),
            'journal_sha256':self.journal_sha,'kept_updates':[row['update'] for row in self.journal['published'] if row['local_status']=='retained'],
            'pruned_updates':[row['update'] for row in self.journal['published'] if row['local_status']=='pruned']}
