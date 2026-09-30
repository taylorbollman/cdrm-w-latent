"""Explicit terminal128 authority; metric math is reused from tested64 observer."""
from copy import deepcopy

import pytest

from scripts import olmo_nfr_final_curves as final
from test_fbt_component_curves import authority


def fixture():
    parent, manifest, spec, sources = authority('NFR', 128, .1)
    metadata = {'parent_update': 64, 'stop_update': 128, 'objective_change': False}
    config = parent['configuration']
    config['schema'] = 'olmo-nfr-128-execution-v1'
    config['continuation'] = metadata
    config['execution_identity']['payload']['continuation'] = metadata
    config['execution_identity']['sha256'] = final.legacy.digest(config['execution_identity']['payload'])
    resolution = {'configuration': config, 'sources': sources, 'continuation': metadata,
                  'scope': {'exact': 'unchanged128'}, 'scope_sha256': '1'*64}
    manifest['manifest_sha256'] = 'f'*64
    publication = {k: deepcopy(manifest[k]) for k in ('manifest_sha256', 'state', 'metadata', 'counters')}
    publication['retention'] = {'download_sha256_verified': True, 'create_only': True,
        'objects': [{'sha256': pin, 'verification': {k: True for k in
            ('download_sha256', 'server_md5', 'server_size', 'sha256_metadata')}}
            for pin in ('f'*64, manifest['state']['sha256'])]}
    parent.update(status='completed_plan', scale='native', segment_completed=True,
        plan_completed=True, segment_stop_after=128, last_verified_cloud_update=128,
        wandb={'status': 'synced'}, final_counters=deepcopy(manifest['counters']),
        published_checkpoints=[publication], continuation=metadata,
        nfr128_scope={'schema': 'olmo-nfr-128-execution-scope-v1',
                     'declaration': resolution['scope'], 'sha256': resolution['scope_sha256']})
    return tuple(deepcopy(x) for x in (parent, manifest, publication, resolution, spec))


@pytest.mark.parametrize('change', ['none', 'update64', 'unfinished', 'cloud', 'wandb',
    'state', 'duplicate_publication', 'configuration', 'resolution', 'cursor', 'continuation'])
def test_only_exact_completed_verified_continuation128_is_accepted(change):
    report, manifest, pub, resolution, spec = fixture()
    if change == 'update64': manifest['counters']['optimizer_updates'] = 64
    elif change == 'unfinished': report['status'] = 'running'
    elif change == 'cloud': pub['retention']['objects'][0]['verification']['download_sha256'] = False
    elif change == 'wandb': report['wandb']['status'] = 'running'
    elif change == 'state': pub['state']['sha256'] = '0'*64
    elif change == 'duplicate_publication': report['published_checkpoints'] *= 2
    elif change == 'configuration': report['configuration']['model'] = {'lambda_kl': 1.}
    elif change == 'resolution': resolution['sources'] = {}
    elif change == 'cursor': manifest['rank_cursors'][0]['cursor']['next_update'] = 64
    elif change == 'continuation': report['continuation']['objective_change'] = True
    if change == 'none':
        final.authenticate_final(report, manifest, pub, resolution, spec, manifest_sha256='f'*64)
    else:
        with pytest.raises(ValueError):
            final.authenticate_final(report, manifest, pub, resolution, spec, manifest_sha256='f'*64)
