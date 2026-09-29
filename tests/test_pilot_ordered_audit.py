import pytest
import json
from scripts.olmo_pilot_ordered_audit import Stream, audit, sha, open_db, verify_originals
from test_pilot_ordered_data import fixture, cpu_tokenizer


def stream(lengths):
    start=0;rows=[]
    for i,n in enumerate(lengths):
        rows.append(dict(document_index=i,stream_start=start,stream_end=start+n,token_count=n));start+=n
    return Stream(rows)


@pytest.mark.parametrize('lengths,start,n', [([3,1,6],0,4),([3,1,6],1,6),([1,1,1,4],0,4),([9],3,3),([3,3],0,3),([3,3],3,3)])
def test_independent_interval_count_matches_literal_identity_windows(lengths,start,n):
    s=stream(lengths);ids=[i for i,k in enumerate(lengths) for _ in range(k)];actual=s.counts(start,n);chunk=ids[start:start+n]
    assert actual['ce_targets']==n-1
    assert actual['latent_pairs']==sum(a==b for a,b in zip(chunk,chunk[1:]))
    assert actual['kl_triples']==sum(a==b==c for a,b,c in zip(chunk,chunk[1:],chunk[2:]))
    end=start+n
    assert actual['omitted_cross_chunk_ce_targets']==int(end<len(ids))
    assert actual['omitted_cross_chunk_latent_pairs']==int(end<len(ids) and ids[end-1]==ids[end])
    assert actual['omitted_cross_chunk_kl_triples']==sum(int(i>=0 and i+2<len(ids) and ids[i]==ids[i+1]==ids[i+2]) for i in (end-2,end-1))


def test_interval_gap_and_out_of_range_fail():
    with pytest.raises(ValueError,match='Noncontiguous'):Stream([dict(document_index=0,stream_start=1,stream_end=3,token_count=2)])
    with pytest.raises(ValueError,match='outside'):stream([3,4]).counts(5,3)


def test_full_tiny_suite_independent_audit(tmp_path):
    f=fixture(tmp_path)
    result=audit(f.corpus,f.output,sha(f.output/'manifest.json'))
    assert result['status']=='passed' and len(result['panels'])==21
    assert result['all_selected_chunks_counted']==27+18*2+18*2
    assert result['literal_sampled_chunks']>0
    with pytest.raises(ValueError,match='suite pin'):audit(f.corpus,f.output,'0'*64)
    with open_db(f.output/'catalog.sqlite') as db:
        originals={d['document_index']:dict(d) for d in db.execute('SELECT * FROM documents')}
    originals[next(iter(originals))]['token_offset']+=1
    with pytest.raises(ValueError,match='original corpus document'):verify_originals(f.corpus,f.manifest,originals)
    m=json.loads((f.output/'manifest.json').read_bytes());del m['panels']['confirmation-main']
    (f.output/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='21-panel'):audit(f.corpus,f.output,sha(f.output/'manifest.json'))
