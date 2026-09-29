"""Ordered-identity metadata adapter; all storage lifecycle operations are inherited.

The only override is which explicit execution identity is accepted. Registration,
verified publication, durable journals and owned-directory pruning are unchanged.
"""
import json
import os
from scripts import olmo_campaign_execution_restore as authority
from scripts import olmo_pilot_execution_restore as ordered_authority
from scripts.olmo_campaign_ssd_storage import (SSDCheckpointStorage as AcceptedStorage,
    _absolute, _file_pins, FILES, validate_storage_paths)


class SSDCheckpointStorage(AcceptedStorage):
    def validate_local(self,receipt):
        self.validate()
        manifest={key:receipt[key] for key in authority.MANIFEST_FIELDS}
        identity=ordered_authority.committed_metadata(manifest)
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
        manifest,objects,identity=ordered_authority.publication_metadata(receipt)
        local=self.validate_local(receipt)
        prefix=f'{self.prefix}/update-{local["update"]:06d}/'
        if any(objects[name]['uri']!=prefix+name for name in FILES):
            raise ValueError('Retained checkpoint belongs to another segment prefix')
        if any(objects[name][key]!=local['manifest' if name=='manifest.json' else 'state'][key]
               for name in FILES for key in ('size_bytes','sha256','md5_base64')):
            raise ValueError('Retained object bytes do not match owned local checkpoint')
        return local
