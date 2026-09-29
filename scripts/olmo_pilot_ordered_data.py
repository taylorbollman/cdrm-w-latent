"""Versioned finite weighted chunk indexes over immutable document shards.

No tokenization, downloading, training, implicit cycling, or old-index mutation.
Documents are selected before chunking; full chunks retain their original true
boundaries even when their execution order is shuffled. Unused document tails
remain in the corpus and are reported, not silently reassigned to another split.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

from cdrm.pretrained import packed_campaign_data as packed
from cdrm.pretrained.document_shards import verify_document_shards

SCHEMA = 'olmo-pilot-ordered-data-v1'
SUITE_SCHEMA = 'olmo-pilot-ordered-suite-v1'
POLICY = {**packed.POLICY,
    'order': 'hash_ordered_full_chunks_with_deterministic_weighted_stratum_deficit',
    'tail': 'unselected_partial_stratum_tails_retained_in_document_corpus',
    'omitted_cross_chunk_counts': 'original_stratum_stream_neighbors_not_shuffled_execution_adjacency',
    'selection': 'whole_documents_before_full_chunk_selection',
}


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)+'\n').encode()


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _db(path, *, readonly=False):
    connection = sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1', uri=True) if readonly else sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA cache_size=-4096')
    connection.execute('PRAGMA temp_store=FILE')
    if not readonly: connection.execute('PRAGMA journal_mode=DELETE')
    return connection


def _sync(path):
    with path.open('rb') as handle: os.fsync(handle.fileno())


def _write(path, value):
    with path.open('xb') as handle:
        handle.write(_json(value)); handle.flush(); os.fsync(handle.fileno())


def _stratum_chunk(connection, index):
    row = connection.execute('SELECT * FROM chunks WHERE ordinal=?', (index,)).fetchone()
    if row is None: raise ValueError('Ordered chunk is absent')
    return row


def _segments(connection, row):
    start, end = row['source_stream_start'], row['source_stream_start']+row['length']
    # Locate the containing document once, then range-scan only overlapping
    # starts. A dual start/end inequality can otherwise scan half a million
    # unrelated documents for each late chunk under SQLite's range planner.
    anchor = connection.execute('SELECT stream_start FROM documents WHERE stratum=? AND stream_start<=? '
        'ORDER BY stream_start DESC LIMIT 1', (row['stratum'],start)).fetchone()
    if anchor is None:raise ValueError('Ordered chunk starts outside its document stream')
    records = connection.execute('SELECT * FROM documents WHERE stratum=? AND stream_start>=? '
        'AND stream_start<? ORDER BY stream_start', (row['stratum'], anchor[0], end))
    output = []
    for doc in records:
        left, right = max(doc['stream_start'], start), min(doc['stream_end'], end)
        offset = left-doc['stream_start']
        output.append(packed.DocumentSegment(doc['document_index'], doc['document_key'], doc['document_id'],
            doc['source_index'], doc['source_name'], doc['source_line'], doc['shard'],
            doc['token_offset']+offset, offset, left-start, right-left, right == doc['stream_end']))
    if sum(s.length for s in output) != row['length']:
        raise ValueError('Selected document intervals do not cover ordered chunk')
    return tuple(output)


def _at(connection, stratum, position, total):
    if not 0 <= position < total: return None
    row = connection.execute('SELECT document_index FROM documents WHERE stratum=? AND stream_start<=? '
        'ORDER BY stream_start DESC LIMIT 1', (stratum, position)).fetchone()
    return None if row is None else row[0]


def _counts(connection, row, length, segments=None):
    segments = _segments(connection, row) if segments is None else segments
    latent = sum(max(s.length-1, 0) for s in segments)
    kl = sum(max(s.length-2, 0) for s in segments)
    end = row['source_stream_start']+row['length']
    total = connection.execute('SELECT tokens FROM streams WHERE stratum=?', (row['stratum'],)).fetchone()[0]
    at = lambda position: _at(connection, row['stratum'], position, total)
    omitted_ce = int(end < total)
    omitted_latent = int(bool(omitted_ce) and at(end-1) == at(end))
    omitted_kl = sum(int(start >= 0 and start+2 < total and len({at(p) for p in range(start, start+3)}) == 1)
        for start in (end-2, end-1)) if omitted_ce else 0
    ce = max(row['length']-1, 0)
    return packed.PackedCounts(1, row['length'], ce, latent, kl, ce-latent, ce-latent,
        max(row['length']-2, 0)-kl, omitted_ce, omitted_latent, omitted_kl,
        length-row['length'], len(segments), sum(s.completes_document for s in segments))


class OrderedCampaignData(packed.PackedCampaignData):
    """PackedCampaignData-compatible batching/cursors with a new ordered authority.

    Only index initialization and source-coordinate lookup differ. Cursor commit,
    dummy-row allocation, tensor masks and bounded token reads use the existing
    implementation unchanged. No old manifest is relabeled or patched.
    """
    def __init__(self, corpus_dir, index_dir):
        self.corpus, self.index_dir = Path(corpus_dir), Path(index_dir)
        if self.corpus.is_symlink() or self.index_dir.is_symlink():
            raise ValueError('Corpus and ordered index must be real directories')
        manifest_path = self.index_dir/'manifest.json'
        signature = packed._stat(manifest_path)
        self._manifest_bytes = manifest_path.read_bytes()
        if packed._stat(manifest_path) != signature: raise ValueError('Ordered manifest changed during read')
        manifest = json.loads(self._manifest_bytes)
        self.manifest_sha256 = hashlib.sha256(self._manifest_bytes).hexdigest()
        if manifest.get('schema') != SCHEMA or manifest.get('policy') != POLICY:
            raise ValueError('Require the explicit pilot ordered-data schema/policy')
        contract = {k:v for k,v in manifest.items() if k not in ('identity_sha256','index')}
        if manifest.get('identity_sha256') != _digest(contract):
            raise ValueError('Ordered membership/order identity differs')
        self.length, self.split = manifest['length'], manifest['split']
        self.total_tokens, self.total_chunks = manifest['total_tokens'], manifest['total_chunks']
        if (type(self.length) is not int or not 3 <= self.length <= 2048 or type(self.total_chunks) is not int
                or self.total_chunks <= 0 or self.total_tokens != self.length*self.total_chunks
                or self.split not in ('train','dev','confirmation')):
            raise ValueError('Ordered dimensions/split differ')
        self._corpus_signatures = {}
        for relative,pin in manifest['corpus_files'].items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts or not relative or path.as_posix() != relative:
                raise ValueError('Unsafe corpus inventory path')
            target = self.corpus/path; before = packed._stat(target)
            if before[2] != pin['size_bytes'] or packed._sha(target) != pin['sha256']:
                raise ValueError('Corpus bytes differ from ordered authority')
            self._corpus_signatures[relative] = before
        if manifest['index']['path'] != 'documents.sqlite': raise ValueError('Ordered index path differs')
        db = self.index_dir/'documents.sqlite'; before = packed._stat(db)
        if before[2] != manifest['index']['size_bytes'] or packed._sha(db) != manifest['index']['sha256']:
            raise ValueError('Ordered index bytes differ')
        self._index_signatures = {'manifest.json':signature, 'documents.sqlite':before}
        packed._check_files(self.index_dir, self._index_signatures)
        self._connection = _db(db, readonly=True)
        self._token_fds = OrderedDict(); self._closed = False
        self._cursor = packed.PackedCursor(self.manifest_sha256, self.split)
        self._identity = manifest['identity_sha256']; self.pad_id = manifest['pad_id']
        self._reader_contract = self._contract()
        try:
            self.validate_integrity()
            summary = self._connection.execute('SELECT COUNT(*),MIN(ordinal),MAX(ordinal),SUM(length) FROM chunks').fetchone()
            if tuple(summary) != (self.total_chunks,0,self.total_chunks-1,self.total_tokens):
                raise ValueError('Ordered SQL chunk dimensions differ')
        except BaseException:
            self.close(); raise

    def _descriptor(self, index):
        if type(index) is not int or not 0 <= index < self.total_chunks:
            raise ValueError('Ordered chunk index out of bounds')
        row = _stratum_chunk(self._connection, index)
        return packed.ChunkDescriptor(index, _digest({'identity':self._identity,'source_chunk_key':row['chunk_key'],
            'index':index}), index*self.length, row['length'])

    def _ordered_row(self, row):
        if not isinstance(row, packed.ChunkDescriptor) or row != self._descriptor(row.index):
            raise ValueError('Ordered chunk descriptor provenance differs')
        return _stratum_chunk(self._connection, row.index)

    def _segments(self, row):
        return _segments(self._connection, self._ordered_row(row))

    def _chunk_counts(self, row, segments=None):
        return _counts(self._connection, self._ordered_row(row), self.length, segments)

    def source_chunk(self, index):
        self.validate_integrity()
        row = dict(_stratum_chunk(self._connection, index))
        self._descriptor(index)  # Reject bool/out-of-range with the same contract.
        return row

DOC_COLUMNS = ('document_index','stratum','split','excluded','order_key','shard','token_offset','token_count',
    'source_index','source_name','source_line','document_id','document_key','token_sha256','content_token_sha256')
RESERVED_COLUMNS = DOC_COLUMNS+('ordinal','stream_start','stream_end')
DOC_SQL = '''document_index INTEGER PRIMARY KEY, stratum TEXT NOT NULL, split TEXT NOT NULL,
    excluded INTEGER NOT NULL, order_key TEXT NOT NULL, shard TEXT NOT NULL, token_offset INTEGER NOT NULL,
    token_count INTEGER NOT NULL, source_index INTEGER NOT NULL, source_name TEXT NOT NULL,
    source_line INTEGER NOT NULL, document_id TEXT NOT NULL, document_key TEXT NOT NULL,
    token_sha256 TEXT NOT NULL, content_token_sha256 TEXT NOT NULL UNIQUE'''
RESERVED_SQL = DOC_SQL+', ordinal INTEGER NOT NULL, stream_start INTEGER NOT NULL, stream_end INTEGER NOT NULL'
CHUNK_SQL = '''stratum TEXT NOT NULL, source_chunk_index INTEGER NOT NULL, source_stream_start INTEGER NOT NULL,
    length INTEGER NOT NULL, chunk_key TEXT NOT NULL UNIQUE, order_key TEXT NOT NULL'''


def panel_names():
    from scripts import olmo_pilot_data_plan as plan
    return ('train','dev-main','confirmation-main',
            *(f'{split}-source/{stratum}' for split in ('dev','confirmation') for stratum in plan.STRATA))


def _split(panel):
    return panel if panel == 'train' else panel.split('-',1)[0]


def _pinned_json(path, expected):
    path=Path(path)
    before=packed._stat(path)
    if before[2]>32*1024**2 or packed._sha(path)!=expected:
        raise ValueError('Pinned acquisition metadata differs')
    value=json.loads(path.read_bytes())
    if packed._stat(path)!=before:raise ValueError('Pinned acquisition metadata changed')
    return value,before


def _source_contract(config, selected_sources, recipe, acquisition, acquisition_sha, authorities):
    from scripts import olmo_pilot_data_plan as plan
    names=[row['name'] for row in selected_sources]
    mapping={row['name']:row for row in selected_sources}
    upstream={row['name']:row for row in acquisition['sources']}
    if (len(names)!=len(set(names)) or len(upstream)!=len(acquisition['sources'])
            or set(mapping)!=set(upstream) or set(mapping)!=set(authorities)
            or len(config['sources'])!=len(mapping) or acquisition.get('recipe')!=recipe):
        raise ValueError('Acquisition recipe/source membership differs')
    if (names != [row['name'] for row in acquisition['sources']]
            or names != [row['pin']['name'] for row in config['sources']]):
        raise ValueError('Acquisition/corpus/selection source order differs')
    output={}
    seen=set()
    for index,source in enumerate(config['sources']):
        pin=source['pin'];name=pin['name'];selected=mapping.get(name);authority=authorities.get(name)
        if selected is None or name in seen or authority is None:
            raise ValueError('Corpus source membership is missing or duplicated')
        seen.add(name)
        remote=authority['raw_object'];actual_upstream=upstream[name]
        if (any(actual_upstream.get(key)!=value for key,value in selected.items())
                or authority['source_pin']!=pin or authority['upstream_source']!=actual_upstream
                or authority['acquisition_plan_sha256']!=acquisition_sha
                or pin['revision']!=recipe['hf_revision'] or pin['uri']!=remote['uri']
                or pin['sha256']!=remote['sha256']):
            raise ValueError('Corpus source/fragment/upstream mapping differs')
        plan._pin(authority['extraction_manifest_sha256']);plan._pin(remote['sha256'])
        if (not isinstance(remote['uri'],str) or not remote['uri'].startswith('gs://fast-chunks/cdrm-w-latent/')
                or not remote['uri'].endswith('/raw-'+name+'/raw.jsonl')
                or not isinstance(remote['generation'],str) or not remote['generation'].isdigit()
                or int(remote['generation'])<1 or type(remote['size_bytes']) is not int or remote['size_bytes']<=0
                or remote.get('verification')!={'server_size':True,'server_md5':True,'sha256_metadata':True,'download_sha256':True}):
            raise ValueError('Raw fragment lacks exact retained generation verification')
        output[index]=selected
    return output


def _build_catalog(corpus, pending, summary, config, mapping, recipe, excluded):
    from scripts import olmo_pilot_data_plan as plan
    path = pending/'catalog.sqlite'; connection = _db(path)
    try:
        connection.execute('CREATE TABLE candidates ('+DOC_SQL+')')
        connection.execute('CREATE INDEX candidate_rank ON candidates(split,stratum,excluded,order_key,content_token_sha256)')
        connection.execute('CREATE TABLE documents ('+RESERVED_SQL+')')
        connection.execute('CREATE INDEX doc_rank ON documents(split,stratum,ordinal)')
        connection.execute('CREATE TABLE streams (split TEXT, stratum TEXT, tokens INTEGER, documents INTEGER, '
                           'order_sha256 TEXT, PRIMARY KEY(split,stratum))')
        connection.execute('CREATE TABLE chunks (split TEXT NOT NULL,'+CHUNK_SQL+')')
        connection.execute('CREATE INDEX chunk_rank ON chunks(split,stratum,order_key,chunk_key)')
        index = 0
        insert = 'INSERT INTO candidates VALUES ('+','.join('?' for _ in DOC_COLUMNS)+')'
        for shard in summary['shards']:
            with (corpus/shard['path']/'documents.jsonl').open('rb') as stream:
                for raw in stream:
                    record = json.loads(raw)
                    if record['kind'] != 'document': continue
                    content = record['content_token_sha256']
                    expected_split = plan.split_for_content_hash(recipe, content)
                    if record['split'] != expected_split:
                        raise ValueError('Stored document split differs from declared content-hash split')
                    selected = mapping[record['source_index']]
                    item = (index,selected['stratum'],record['split'],int(content in excluded),
                        plan.document_order_key(recipe,content)[0],shard['path'],record['token_offset'],
                        record['token_count'],record['source_index'],selected['name'],record['source_line'],
                        record['document_id'],record['document_key'],record['token_sha256'],content)
                    connection.execute(insert,item);index += 1
        connection.commit()
        train = plan.panel_quotas(recipe,'train'); heldout = plan.heldout_requirements(recipe)
        streams = []
        for split in ('train','dev','confirmation'):
            for stratum in plan.STRATA:
                required = {'tokens':train[stratum]*recipe['panels']['length'],'documents':0} if split=='train' else heldout[stratum]
                if not required['tokens']: continue
                tokens = documents = 0; digest = hashlib.sha256()
                candidates = connection.execute('SELECT * FROM candidates WHERE split=? AND stratum=? AND excluded=0 '
                    'ORDER BY order_key,content_token_sha256', (split,stratum))
                for row in candidates:
                    end = tokens+row['token_count']; values = tuple(row[k] for k in DOC_COLUMNS)+(documents,tokens,end)
                    connection.execute('INSERT INTO documents VALUES ('+','.join('?' for _ in values)+')',values)
                    digest.update(_json(dict(zip(RESERVED_COLUMNS,values))))
                    tokens=end;documents+=1
                    if tokens>=required['tokens'] and documents>=required['documents']: break
                if tokens<required['tokens'] or documents<required['documents']:
                    raise ValueError(f'Insufficient {split}/{stratum}: {tokens} tokens/{documents} documents, require {required}')
                record={'split':split,'stratum':stratum,'tokens':tokens,'documents':documents,'order_sha256':digest.hexdigest()}
                connection.execute('INSERT INTO streams VALUES (?,?,?,?,?)',tuple(record.values()))
                streams.append({**record,'required':required,'whole_document_overshoot_tokens':tokens-required['tokens'],
                    'available_full_chunks':tokens//recipe['panels']['length'],
                    'unselected_partial_tail_tokens':tokens%recipe['panels']['length']})
                stream_identity=_digest({'corpus_manifest_sha256':packed._sha(corpus/'manifest.json'),
                    'split':split,'stratum':stratum,'document_order_sha256':record['order_sha256'],
                    'length':recipe['panels']['length']})
                for chunk_index in range(tokens//recipe['panels']['length']):
                    start=chunk_index*recipe['panels']['length']
                    key=_digest({'stream_identity':stream_identity,'source_chunk_index':chunk_index,
                                 'source_stream_start':start,'length':recipe['panels']['length']})
                    connection.execute('INSERT INTO chunks VALUES (?,?,?,?,?,?,?)',
                        (split,stratum,chunk_index,start,recipe['panels']['length'],key,plan.chunk_order_key(recipe,key)[0]))
        connection.commit()
        selected_docs=connection.execute('SELECT COUNT(*),COUNT(DISTINCT content_token_sha256),SUM(excluded) FROM documents').fetchone()
        if selected_docs[0] != selected_docs[1] or selected_docs[2] != 0:
            raise ValueError('Selected documents overlap across splits or readiness exclusions')
        membership={f'{r["split"]}/{r["stratum"]}':{'documents':r['n'],'tokens':r['tokens'],'excluded_documents':r['excluded_docs'],
            'excluded_tokens':r['excluded_tokens']} for r in connection.execute(
            'SELECT split,stratum,COUNT(*) AS n,SUM(token_count) AS tokens,SUM(excluded) AS excluded_docs,'
            'SUM(excluded*token_count) AS excluded_tokens FROM candidates GROUP BY split,stratum')}
        return streams,membership
    finally:
        connection.close()


def _build_panel(catalog, output, panel, recipe, common, streams):
    from scripts import olmo_pilot_data_plan as plan
    output.mkdir(parents=True); db=output/'documents.sqlite'; connection=_db(db)
    split=_split(panel); quotas=plan.panel_quotas(recipe,panel); selected_strata=[s for s,n in quotas.items() if n]
    try:
        connection.execute('CREATE TABLE documents ('+RESERVED_SQL+')')
        connection.execute('CREATE INDEX doc_stream_start ON documents(stratum,stream_start)')
        connection.execute('CREATE INDEX doc_stream_end ON documents(stratum,stream_end)')
        connection.execute('CREATE TABLE streams (stratum TEXT PRIMARY KEY,tokens INTEGER,documents INTEGER,order_sha256 TEXT)')
        connection.execute('CREATE TABLE chunks (ordinal INTEGER PRIMARY KEY,'+CHUNK_SQL+',selection_rank INTEGER NOT NULL)')
        connection.execute('CREATE TABLE panel_documents (document_index INTEGER PRIMARY KEY,selected_tokens INTEGER NOT NULL)')
        for stratum in selected_strata:
            for row in catalog.execute('SELECT * FROM documents WHERE split=? AND stratum=? ORDER BY ordinal',(split,stratum)):
                connection.execute('INSERT INTO documents VALUES ('+','.join('?' for _ in RESERVED_COLUMNS)+')',tuple(row))
            row=catalog.execute('SELECT stratum,tokens,documents,order_sha256 FROM streams WHERE split=? AND stratum=?',(split,stratum)).fetchone()
            connection.execute('INSERT INTO streams VALUES (?,?,?,?)',tuple(row))
        queues={s:catalog.execute('SELECT stratum,source_chunk_index,source_stream_start,length,chunk_key,order_key '
            'FROM chunks WHERE split=? AND stratum=? ORDER BY order_key,chunk_key LIMIT ?', (split,s,quotas[s]))
            for s in selected_strata}
        selected_rank=dict.fromkeys(selected_strata,0); order_digest=hashlib.sha256()
        for ordinal,stratum in enumerate(plan.weighted_stratum_order(quotas)):
            row=queues[stratum].fetchone()
            if row is None: raise ValueError('Insufficient full chunks for declared panel')
            values=(ordinal,*tuple(row),selected_rank[stratum]);selected_rank[stratum]+=1
            connection.execute('INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)',values)
            order_digest.update(_json({'ordinal':ordinal,**dict(row),'selection_rank':values[-1]}))
        connection.commit()
        counts=packed.PackedCounts(); by_stratum={s:packed.PackedCounts() for s in selected_strata}
        for row in connection.execute('SELECT * FROM chunks ORDER BY ordinal'):
            segments=_segments(connection,row)
            current=_counts(connection,row,recipe['panels']['length'],segments);counts+=current;by_stratum[row['stratum']]+=current
            for segment in segments:
                connection.execute('INSERT INTO panel_documents VALUES (?,?) ON CONFLICT(document_index) '
                    'DO UPDATE SET selected_tokens=selected_tokens+excluded.selected_tokens',
                    (segment.document_index,segment.length))
        connection.commit()
        selected_document_count=connection.execute('SELECT COUNT(*) FROM panel_documents').fetchone()[0]
        reserved_document_count=connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
        membership_by_source={r['source_name']:{'documents':r['documents'],'selected_tokens':r['tokens']} for r in
            connection.execute('SELECT d.source_name,COUNT(*) AS documents,SUM(p.selected_tokens) AS tokens '
                'FROM panel_documents p JOIN documents d USING(document_index) GROUP BY d.source_name')}

        membership_sha=hashlib.sha256()
        for row in connection.execute('SELECT chunk_key FROM chunks ORDER BY chunk_key'):membership_sha.update((row[0]+'\n').encode())
        contract={**common,'schema':SCHEMA,'panel':panel,'split':split,'length':recipe['panels']['length'],
            'policy':POLICY,'stratum_quotas':quotas,'order_sha256':order_digest.hexdigest(),
            'chunk_membership_sha256':membership_sha.hexdigest(),'reserved_documents':reserved_document_count,
            'selected_documents':selected_document_count,'membership_by_source':membership_by_source,'total_tokens':counts.valid_tokens,'total_chunks':counts.packed_rows,
            'final_chunk_tokens':recipe['panels']['length'],'counts':asdict(counts),
            'counts_by_stratum':{s:asdict(v) for s,v in by_stratum.items()},
            'streams':[r for r in streams if r['split']==split and r['stratum'] in selected_strata],
            'panel_overlap_policy':'main/source panels may share chunk keys; never independent replications',
            'catalog_authority':common['selection_authority']}
    finally:
        connection.close()
    _sync(db)
    manifest={**contract,'identity_sha256':_digest(contract),
        'index':{'path':'documents.sqlite','size_bytes':db.stat().st_size,'sha256':packed._sha(db)}}
    _write(output/'manifest.json',manifest)
    return manifest


def build_ordered_data(corpus_dir, output_dir, *, recipe, inventory_bytes, excluded_content_hashes,
                       acquisition_plan_path, acquisition_plan_sha256,
                       source_authorities_path, source_authorities_sha256):
    """Build all declared train/dev/confirmation panels; publish only if all fit.

    SQLite bounds metadata memory. Whole-document reserves and unselected tails
    are explicit; no tokens are copied. Failure publishes no completed suite.
    This builder currently constructs round zero only; prefix extension requires
    an explicit later authority and is not silently synthesized here.
    """
    from scripts import olmo_pilot_data_plan as plan
    recipe=json.loads(_json(recipe));plan.validate_recipe(recipe)
    excluded=plan.verify_exclusions(recipe,excluded_content_hashes)
    selected_sources=plan.source_selection(recipe,inventory_bytes)
    acquisition,acquisition_signature=_pinned_json(acquisition_plan_path,acquisition_plan_sha256)
    authorities,authority_signature=_pinned_json(source_authorities_path,source_authorities_sha256)
    corpus,output=Path(corpus_dir),Path(output_dir)
    if corpus.is_symlink() or output.exists() or output.is_symlink():
        raise ValueError('Require real immutable corpus and fresh ordered-suite output')
    summary=verify_document_shards(corpus)
    if not summary['completed']:raise ValueError('Ordered selection requires completed immutable corpus')
    config,inventory,signatures=packed._verified_inventory(corpus,summary)
    if config['split_policy']!={'seed':recipe['split']['seed'],'weights':recipe['split']['weights']}:
        raise ValueError('Corpus split policy differs from pilot recipe')
    expected_tokenizer={key:recipe['tokenizer'][key] for key in ('repo','revision','sha256')}
    if (any(config['tokenizer'][k]!=v for k,v in expected_tokenizer.items())
            or any(config[k]!=recipe['tokenizer'][k] for k in ('vocab_size','eos_id','token_dtype'))):
        raise ValueError('Corpus native tokenization differs')
    if summary['tokens']>recipe['bounds']['candidate_tokens']:
        raise ValueError('Candidate corpus exceeds declared finite token bound')
    mapping=_source_contract(config,selected_sources,recipe,acquisition,acquisition_plan_sha256,authorities)
    output.parent.mkdir(parents=True,exist_ok=True)
    pending=Path(tempfile.mkdtemp(prefix='.pending-ordered-pilot-',dir=output.parent))
    try:
        streams,availability=_build_catalog(corpus,pending,summary,config,mapping,recipe,excluded)
        catalog_path=pending/'catalog.sqlite';_sync(catalog_path)
        catalog_pin={'path':'catalog.sqlite','size_bytes':catalog_path.stat().st_size,'sha256':packed._sha(catalog_path)}
        common={'recipe':recipe,'recipe_sha256':plan.recipe_sha256(recipe),'round_id':0,
            'selection_authority':catalog_pin,'source_selection':selected_sources,
            'acquisition_authority':{'plan_sha256':acquisition_plan_sha256,
                'source_authorities_sha256':source_authorities_sha256,'source_authorities':authorities},
            'exclusion_authority':recipe['exclusions'],'corpus_manifest_sha256':inventory['manifest.json']['sha256'],
            'corpus_config_sha256':summary['config_sha256'],'corpus_files':inventory,'tokenizer':config['tokenizer'],
            'pad_id':recipe['tokenizer']['pad_id'],'eos_id':config['eos_id'],'vocab_size':config['vocab_size'],
            'token_dtype':config['token_dtype']}
        panels={}
        with closing(_db(catalog_path,readonly=True)) as catalog:
            for panel in panel_names():
                relative='panels/'+panel
                manifest=_build_panel(catalog,pending/relative,panel,recipe,common,streams)
                panels[panel]={'path':relative,'manifest_sha256':packed._sha(pending/relative/'manifest.json'),
                    'identity_sha256':manifest['identity_sha256'],'tokens':manifest['total_tokens'],'chunks':manifest['total_chunks'],
                    'counts':manifest['counts'],'stratum_quotas':manifest['stratum_quotas']}
        packed._check_files(corpus,signatures)
        if (packed._stat(Path(acquisition_plan_path))!=acquisition_signature
                or packed._stat(Path(source_authorities_path))!=authority_signature):
            raise ValueError('Acquisition authorities changed during ordered build')
        contract={'schema':SUITE_SCHEMA,**common,'panels':panels,'availability':availability,'reserved_streams':streams,
            'split_intersections':{'content_hashes':0,'document_indices':0,'readiness_exclusions':0},
            'scope':'Finite data preparation only; no model exposure, training, outcome selection or executor compatibility claim'}
        manifest={**contract,'identity_sha256':_digest(contract)}
        _write(pending/'manifest.json',manifest)
        for directory,_,_ in os.walk(pending,topdown=False):
            fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        os.rename(pending,output)
        fd=os.open(output.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return manifest
    finally:
        if pending.exists():shutil.rmtree(pending)
