import uuid
import pytest


@pytest.fixture
def synthetic_user_ids():
    """Generate 10,000 unique synthetic user IDs for batch testing.

    Uses UUID4 to simulate realistic, high-cardinality user identifiers
    (rather than small sequential integers, which could mask hash
    distribution issues that only appear at scale or with non-trivial
    string structure).

    Returns:
        list[str]: 10,000 unique UUID4 strings.
    """
    return [str(uuid.uuid4()) for _ in range(10_000)]