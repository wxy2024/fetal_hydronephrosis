import hashlib
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


VALID_TEMPORAL_EXPERIMENTS = {
    "baseline",
    "first_only",
    "last_only",
    "shuffled",
    "no_explicit_time",
}


def filter_multivisit_cases(
    samples: Sequence[Dict],
    min_visits: int = 2,
) -> Tuple[List[Dict], Dict]:
    """Keep a whole case only when every available side has enough visits."""
    visit_counts_by_case = defaultdict(list)
    for sample in samples:
        visit_counts_by_case[str(sample["case_id"])].append(
            len(sample.get("visits", []))
        )

    eligible_cases = {
        case_id
        for case_id, counts in visit_counts_by_case.items()
        if counts and min(counts) >= min_visits
    }
    filtered = [
        sample for sample in samples if str(sample["case_id"]) in eligible_cases
    ]
    stats = {
        "min_visits": min_visits,
        "cases_before": len(visit_counts_by_case),
        "cases_after": len(eligible_cases),
        "cases_excluded": len(visit_counts_by_case) - len(eligible_cases),
        "side_samples_before": len(samples),
        "side_samples_after": len(filtered),
    }
    return filtered, stats


def _stable_case_seed(case_id: str, shuffle_seed: int) -> int:
    digest = hashlib.sha256(f"{shuffle_seed}:{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _non_identity_permutation(length: int, rng: random.Random) -> List[int]:
    indices = list(range(length))
    rng.shuffle(indices)
    if length > 1 and indices == list(range(length)):
        indices = indices[1:] + indices[:1]
    return indices


def apply_temporal_experiment(
    samples: Sequence[Dict],
    experiment: str,
    shuffle_seed: int = 42,
) -> List[Dict]:
    """Return shallow sample copies with the requested visit sequence."""
    experiment = experiment.strip().lower()
    if experiment not in VALID_TEMPORAL_EXPERIMENTS:
        raise ValueError(
            f"Unknown temporal experiment {experiment!r}; "
            f"expected one of {sorted(VALID_TEMPORAL_EXPERIMENTS)}."
        )

    transformed = []
    for sample in samples:
        visits = list(sample.get("visits", []))
        chronological_visit_ids = [str(v.get("visit_id", "")) for v in visits]

        if experiment == "first_only":
            visits = visits[:1]
        elif experiment == "last_only":
            visits = visits[-1:]
        elif experiment == "shuffled":
            rng = random.Random(
                _stable_case_seed(str(sample["case_id"]), shuffle_seed)
            )
            order = _non_identity_permutation(len(visits), rng)
            visits = [visits[index] for index in order]

        item = dict(sample)
        item["visits"] = visits
        item["num_visits"] = len(visits)
        item["temporal_experiment"] = experiment
        item["temporal_shuffle_seed"] = (
            shuffle_seed if experiment == "shuffled" else None
        )
        item["chronological_visit_ids"] = chronological_visit_ids
        transformed.append(item)

    return transformed
