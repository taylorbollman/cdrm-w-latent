"""NFR scope mutation oracles and unchanged accepted KL audit proof."""
import ast
from copy import deepcopy
import inspect
import subprocess
import sys

import pytest

from scripts import olmo_nfr_kl_audit as audit
from scripts import olmo_kl_continuation_audit_v2 as accepted_v2


def fixture():
    scope = {"schema": audit.SCOPE_SCHEMA, "activation": "conditional_after_fbt_only_assessment", "arm": "NFR",
        "parent_update": 32, "review_stop": 64, "planned_updates": 128, "kl_weights": [1., .1],
        "storage_prefix": "gs://fast-chunks/cdrm-w-latent/fbt-rt-nextlat/nfr-kl/example",
        **{key: {"path": "/example/" + key, "sha256": str(i) * 64} for i, key in
           enumerate(("declaration", "resolved", "parent_report", "parent_checkpoint"), 1)}}
    scope_pin = "a" * 64
    sources = {"source-" + str(i) + ".py": "b" * 64 for i in range(214)}
    sources[".runtime/olmo-nfr-kl-continuation/scope-declaration.json"] = scope_pin
    payload = {"sources": deepcopy(sources), "plan": {"updates": [{}] * 128}, "model_contract": {
        "mode": {"rt_mode": {"selected_layers": [0, 15], "alpha": 1.}, "num_passes": 4}, "weights": {"latent": 1.}}}
    report = {"scale": "native", "arm": "NFR", "sources": deepcopy(sources),
        "nfr_scope": {"schema": "olmo-nfr-kl-execution-scope-v1", "declaration": deepcopy(scope), "sha256": scope_pin},
        "configuration": {"execution_identity": {"payload": payload}},
        "declaration_sha256": scope["declaration"]["sha256"], "resolved_sha256": scope["resolved"]["sha256"],
        "branch": {"parent_report_sha256": scope["parent_report"]["sha256"],
                   "parent_manifest_sha256": scope["parent_checkpoint"]["sha256"]}}
    return report, scope, scope_pin, sources


def test_native_scope_has_independent_source_identity_and_exact_parent():
    values = fixture(); before = deepcopy(values)
    audit.scope_check(audit.Audit(), *values, "case")
    assert values == before


@pytest.mark.parametrize("mutation", ["NF", "tiny", "scope_unbound", "source_drift", "hidden_source_drift",
    "alternate_parent", "alternate_declaration", "RT_disabled", "wrong_RT_layers", "alpha", "K2", "latent",
    "192plan", "activation", "other_KL", "parent_update", "scope_contents", "scope_SHA", "extra_scope_field"])
def test_scope_cannot_silently_change_intervention_or_ancestry(mutation):
    report, scope, pin, sources = fixture()
    payload = report["configuration"]["execution_identity"]["payload"]
    if mutation == "NF": report["arm"] = "NF"
    elif mutation == "tiny": report["scale"] = "tiny"
    elif mutation == "scope_unbound": sources.pop(".runtime/olmo-nfr-kl-continuation/scope-declaration.json")
    elif mutation == "source_drift": report["sources"]["source-1.py"] = "c" * 64
    elif mutation == "hidden_source_drift": payload["sources"]["source-1.py"] = "c" * 64
    elif mutation == "alternate_parent": report["branch"]["parent_manifest_sha256"] = "c" * 64
    elif mutation == "alternate_declaration": report["declaration_sha256"] = "c" * 64
    elif mutation == "RT_disabled": payload["model_contract"]["mode"]["rt_mode"]["selected_layers"] = []
    elif mutation == "wrong_RT_layers": payload["model_contract"]["mode"]["rt_mode"]["selected_layers"] = [1, 14]
    elif mutation == "alpha": payload["model_contract"]["mode"]["rt_mode"]["alpha"] = .5
    elif mutation == "K2": payload["model_contract"]["mode"]["num_passes"] = 2
    elif mutation == "latent": payload["model_contract"]["weights"]["latent"] = .1
    elif mutation == "192plan": payload["plan"]["updates"] *= 2
    elif mutation == "activation": scope["activation"] = "launch_automatically"
    elif mutation == "other_KL": scope["kl_weights"] = [1., .01]
    elif mutation == "parent_update": scope["parent_update"] = 0
    elif mutation == "scope_contents": report["nfr_scope"]["declaration"]["kl_weights"] = [1., .01]
    elif mutation == "scope_SHA": report["nfr_scope"]["sha256"] = "c" * 64
    elif mutation == "extra_scope_field": scope["unreviewed"] = True
    with pytest.raises(ValueError): audit.scope_check(audit.Audit(), report, scope, pin, sources, "case")


def test_accepted_validator_is_literal_and_only_branch_native_arm_changes():
    def tree(function):
        return ast.parse(inspect.getsource(function))
    assert ast.dump(tree(audit.validate_report)) == ast.dump(tree(accepted_v2.validate_report))
    before = tree(audit.original._branch_check)
    class ReplaceNativeArm(ast.NodeTransformer):
        def visit_Tuple(self, node):
            if [getattr(item, "value", None) for item in node.elts] == ["NF", 32, 64]:
                node.elts[0].value = "NFR"
            return self.generic_visit(node)
    assert ast.dump(ReplaceNativeArm().visit(before)) == ast.dump(tree(audit._branch_check))
    assert audit._parent_check is audit.original._parent_check
    assert audit.training_check is audit.original.training_check
    assert audit.evaluation_check is audit.original.evaluation_check
    assert audit._same_except_kl is audit.original._same_except_kl


def test_no_report_relabeling_or_module_global_patching():
    source = ast.parse(inspect.getsource(audit))
    assert not any(isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                   and isinstance(node.value, ast.Name) and node.value.id == "original" for node in ast.walk(source))
    subprocess.run([sys.executable, "-c", "import sys; import scripts.olmo_nfr_kl_audit; assert 'torch' not in sys.modules; assert 'google.cloud.storage' not in sys.modules"],
                   cwd=audit.ROOT, check=True, capture_output=True, text=True)
