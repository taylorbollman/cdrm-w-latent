from dataclasses import asdict
from pathlib import Path
import pytest
from scripts import olmo_topology_execute as execute
from scripts.olmo_allocation_acceptance import fixture_for_rank, logical_fixture, literal_counts
from cdrm.pretrained.campaign_recipe import CampaignRecipe
from cdrm.pretrained.lm_training import TrainingCounters
from scripts.olmo_topology_contract import expected_counters_since_origin

@pytest.mark.parametrize('arm',['B','NFR'])
def test_data_hashes_and_counts_independent_of_partition(arm):
    recipe=CampaignRecipe(arm,sequence_length=8,rt_layers=(0,1),document_policy='continuous-stream-v1')
    plans=execute.tiny_plans(recipe)
    for update in range(3):
        def rows(world):
            all_rows=[]
            for rank in range(world):
                batches,noises,keys=fixture_for_rank(recipe,update,rank=rank,world_size=world)
                all_rows.extend(execute.input_rows(batches,noises,keys))
            return sorted(all_rows,key=lambda x:x['key'])
        assert rows(1)==rows(2)
        assert len(rows(1))==len(logical_fixture(update).rows)
        assert plans[update].counts.valid_tokens==logical_fixture(update).counts.presented_tokens
    source=expected_counters_since_origin(TrainingCounters(),plans,1,recipe,world_size=2,batch_size=2)
    one=expected_counters_since_origin(source,plans,2,recipe,world_size=1,batch_size=2)
    two=expected_counters_since_origin(source,plans,2,recipe,world_size=2,batch_size=2)
    assert {k:v for k,v in asdict(one).items() if k!='microbatches'}=={k:v for k,v in asdict(two).items() if k!='microbatches'}
    assert source.microbatches==4
    assert one.microbatches==8 and two.microbatches==8

@pytest.mark.parametrize('options',[
    ['--phase','seed','--scale','native','--stop-after','128'],
    ['--phase','resume','--scale','tiny','--stop-after','2'],
    ['--phase','seed','--scale','tiny','--stop-after','4'],
])
def test_cli_rejects_unsupported_or_unpinned_scope(tmp_path,options):
    with pytest.raises(SystemExit):
        execute.parse_args(options+['--output-dir',str(tmp_path/'out'),'--checkpoint-root',str(tmp_path/'ckpt'),
                                  '--batch-size','2','--lineage','test'])
