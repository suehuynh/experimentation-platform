"""Tests for core/assignment.py"""
import pytest
from core.assignment import VariantSpec, ExperimentConfig, HashAssignment, HashFunction, murmur_hash


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeHashFn:
    """Deterministic, fully-controllable hash function for boundary testing.
 
    Wraps a dict mapping input strings to fixed raw hash outputs, so a
    test can force a specific user_id to land at an exact normalized
    point (e.g. exactly 0.10, or just under 1.0).
    """
    def __init__(self, mapping: dict, max_value: int = (2**32) - 1):
        self._mapping = mapping
        self.max_value = max_value
    
    def __call__(self, input_str: str) -> int:
        if input_str not in self._mapping:
            raise KeyError(
                f"_FakeHashFn was not configured with a value for '{input_str}'. "
                "Add it to the mapping passed into the fixture."
            )
        return self._mapping[input_str]


@pytest.fixture
def make_fake_hash_fn():
    """Factory fixture: returns a function that builds a _FakeHashFn.
 
    Using a factory (rather than a fixed instance) lets each test supply
    its own user_id -> raw_hash mapping, since different boundary tests
    need different controlled values.
 
    Usage in a test:
        hash_fn = make_fake_hash_fn({"user_a": 0, "user_b": 2**32 - 1})
    """
    def _factory(mapping: dict, max_value: int = (2**32) - 1) -> _FakeHashFn:
        return _FakeHashFn(mapping, max_value=max_value)
    return _factory

# ---------------------------------------------------------------------------
# TestHashAssignment
# ---------------------------------------------------------------------------

class TestHashAssignment:
    def setup_method(self, method) -> None:
        """Set up standard configurations used across HashAssignment tests."""
        self.variants = (
            VariantSpec(name="variant1", percentage=0.1),
            VariantSpec(name="variant2", percentage=0.1),
            VariantSpec(name="control", percentage=0.8),
        )
        self.config_a = ExperimentConfig(
            experiment_id="exp_button_color_2026",
            salt="salt_alpha_99",
            variants=self.variants,
        )
        self.assigner = HashAssignment(hash_fn=murmur_hash)

    def test_deterministic(self) -> None:
        """Verify that the same user-experiment pairing yields an identical
        variant across repetitive calls.
        """
        user_id = "user_voter_101"
        first_run = self.assigner.assign(user_id, self.config_a)

        for i in range(10):
            subsequent_run = self.assigner.assign(user_id, self.config_a)
            assert first_run == subsequent_run, (
                f"Non-deterministic assignment encountered on iteration {i}."
            )

    def test_boundary_assignment_lower_inclusive(self, fake_hash_fn) -> None:
        """Verify a point exactly on a variant's lower cutoff bound is
        assigned to that variant (lower bound is inclusive per
        cumulative_cutoffs' [lower, upper) convention).
        """
        # TODO:
        # - construct fake_hash_fn so a specific user_id maps to a raw
        #   hash that normalizes to EXACTLY the lower bound of "variant2"
        #   (i.e. point == 0.10, the boundary between variant1 and variant2)
        # - assign that user_id and assert the result is "variant2", not
        #   "variant1" -- proving the boundary is handled as you intend
        max_value = 100

        user_id ="boundary_user"
        hash_fn = make_fake_hash_fn({user_id: 10}, max_value=max_value)
        assigner = HashAssignment(hash_fn=hash_fn)

        result = assigner.assign(user_id, self.config_a)
        assert result == "variant2", (
            f"Expected point exactly at 0.10 to land in 'variant2' "
            f"(lower bound inclusive), got '{result}'."
        )

    def test_boundary_assignment_near_upper_exclusive(self, fake_hash_fn) -> None:
        """Verify a point just below 1.0 (the largest possible normalized
        hash value) is still assigned to the last variant, never raising
        the unmapped-point AssertionError.
        """
        max_value = (2**32) - 1

        user_id ="max_hash_user"
        hash_fn = make_fake_hash_fn({user_id: max_value}, max_value=max_value)
        assigner = HashAssignment(hash_fn=hash_fn)

        point = assigner._to_unit_interval(user_id, self.config_a)
        assert point < 1.0, f"Expected normalized point < 1.0, got {point}"

        result = assigner.assign(user_id, self.config_a)
        assert result == "control", (
            f"Expected max possible hash to land in 'control' (last variant), "
            f"got '{result}'."
        )

    def test_coverage_no_unmapped_points(self, synthetic_user_ids) -> None:
        """Verify that assigning a large batch of real, varied user IDs
        through the real hash function never raises the unmapped-point
        AssertionError.
        """
        result = self.assigner.assign_batch(synthetic_user_ids, self.config_a)
        assert len(result) == len(synthetic_user_ids), (
            "Not every user_id received an assignment."
        )

    def test_distribution_matches_configured_percentages(self, synthetic_user_ids) -> None:
        """Verify that across a large batch, the empirical proportion of
        users landing in each variant is close to its configured
        percentage -- this is the test that actually validates
        MurmurHash3's uniformity/avalanche properties, not the
        bucketing logic.
        """
        result = self.assigner.assign_batch(synthetic_user_ids, self.config_a)

        counts = {variant.name: 0 for variant in self.variants}
        for variant_name in result.values():
            counts[variant_name] += 1
        
        n = len(synthetic_user_ids)
        expected_pcts = {v.name : v.percentage for v in self.variants}

        # Tolerance shrinks as sample size grows (law of large numbers).
        # For n=10,000, +/- 0.015 (1.5 percentage points) is generous but safe.
        tolerance = 0.015

        for variant_name, expected_pct in expected_pcts.items():
            empirical_pct = counts[variant_name] / n
            assert abs(empirical_pct - expected_pct) < tolerance, (
                f"Variant '{variant_name}': expected ~{expected_pct:.2%}, "
                f"got {empirical_pct:.2%} (n={n})."
            )

    def test_ramp_stability(self) -> None:
        """Verify a user assigned to a variant at a smaller ramp percentage
        remains assigned to that SAME variant when the ramp grows, given
        the same fixed variant order, experiment_id, and salt.
        """
        config_small_ramp = ExperimentConfig(
            experiment_id="exp_ramp_test",
            salt="salt_ramp_1",
            variants=(
                VariantSpec(name="treatment", percentage=0.10),
                VariantSpec(name="control", percentage=0.90),
            ),
        )

        config_large_ramp = ExperimentConfig(
            experiment_id="exp_ramp_test",  # same experiment_id
            salt="salt_ramp_1",              # same salt
            variants=(
                VariantSpec(name="treatment", percentage=0.30),  # grown
                VariantSpec(name="control", percentage=0.70),
            ),
        )
        # Search a batch of users for one assigned to "treatment" under the
        # small ramp, since we don't know in advance which user_id that is.
        candidate_ids = [f"user_{i}" for i in range(200)]
        treatment_users_small_ramp = [
            uid for uid in candidate_ids
            if self.assigner.assign(uid, config_small_ramp) == "treatment"
        ]
        assert len(treatment_users_small_ramp) > 0, (
            "No users landed in 'treatment' out of 200 candidates - "
            "increase candidate_ids or check hash distribution."
        )

        for uid in treatment_users_small_ramp:
            result = self.assigner.assign(uid, config_large_ramp)
            assert result == "treatment", (
                f"User '{uid}' was in 'treatment' at 10% ramp but moved to "
                f"'{result}' at 30% ramp - ramp stability violated."
            )


    def test_missing_max_value_raises_value_error(self) -> None:
        """Verify HashAssignment.__init__ fails fast with a clear error
        when given a hash_fn lacking a max_value attribute.
        """
        def broken_hash_fn(input_str: str) -> int:
            return 0  # no max_value attribute attached
 
        with pytest.raises(ValueError):
            HashAssignment(hash_fn=broken_hash_fn)


# ---------------------------------------------------------------------------
# TestExperimentConfig
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    def test_percentages_must_sum_to_one(self) -> None:
        """Verify construction fails when variant percentages don't sum
        to 1.0.
        """
        bad_variants = (
            VariantSpec(name="a", percentage=0.6),
            VariantSpec(name="b", percentage=0.5),
        )

        with pytest.raises(AssertionError):
            ExperimentConfig(experiment_id="exp1", salt="s1", variants=bad_variants)
        

    def test_duplicate_variant_names_rejected(self) -> None:
        """Verify construction fails when two variants share a name."""
        bad_variants = (
            VariantSpec(name="a", percentage=0.5),
            VariantSpec(name="a", percentage=0.5),  # duplicate name
        )
        with pytest.raises(AssertionError):
            ExperimentConfig(experiment_id="exp1", salt="s1", variants=bad_variants)

    def test_negative_percentage_rejected(self) -> None:
        """Verify construction fails when a variant has a negative
        percentage.
        """
        bad_variants = (
            VariantSpec(name="a", percentage=-0.1),
            VariantSpec(name="b", percentage=1.1),  # compensates to sum=1.0
        )
        with pytest.raises(AssertionError):
            ExperimentConfig(experiment_id="exp1", salt="s1", variants=bad_variants)