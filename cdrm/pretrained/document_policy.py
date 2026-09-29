"""Versioned document-boundary semantics; no policy is inferred from EOS IDs."""

ISOLATED_DOCUMENTS = "isolated-v1"
CONTINUOUS_STREAM = "continuous-stream-v1"


def validate_document_policy(policy):
    if policy not in (ISOLATED_DOCUMENTS, CONTINUOUS_STREAM):
        raise ValueError("document_policy must be isolated-v1 or continuous-stream-v1")
    return policy


def feedback_eligibility(valid, documents, policy):
    """Local adjacent positions only; neither policy carries state across rows."""
    validate_document_policy(policy)
    eligible = valid[:, 1:] & valid[:, :-1]
    if policy == ISOLATED_DOCUMENTS:
        eligible = eligible & (documents[:, 1:] == documents[:, :-1])
    return eligible
