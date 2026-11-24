import re
from math import ceil, log2
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

MATCH_FIELDS = ("id", "title_contains", "text_contains", "technique_id", "technique_family")
_ACTION_IDENTIFIER_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")
_TECHNIQUE_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])T(\d{4})(?:[._](\d{3}))?(?![A-Za-z0-9._])",
    re.IGNORECASE,
)


def _canonical_technique_id(value: str) -> str:
    return value.upper().replace("_", ".")


def _title_technique_ids(title: str) -> set[str]:
    return {
        f"T{match.group(1)}" + (f".{match.group(2)}" if match.group(2) else "")
        for match in _TECHNIQUE_ID_PATTERN.finditer(title)
    }


def _same_family(wanted: str, found: set[str]) -> bool:
    base = _canonical_technique_id(wanted).split(".")[0]
    return any(technique.split(".")[0] == base for technique in found)


def matches(result: Mapping[str, Any], judgment: Mapping[str, Any]) -> bool:
    if result.get("type") != judgment["type"]:
        return False
    if "id" in judgment and result.get("id") != judgment["id"]:
        return False
    if "title_contains" in judgment and (
        judgment["title_contains"].casefold() not in str(result.get("title", "")).casefold()
    ):
        return False
    if "text_contains" in judgment and (
        judgment["text_contains"].casefold() not in str(result.get("text", "")).casefold()
    ):
        return False
    if "technique_id" in judgment and (
        _canonical_technique_id(judgment["technique_id"])
        not in _title_technique_ids(str(result.get("title", "")))
    ):
        return False
    if "technique_family" in judgment and not _same_family(
        judgment["technique_family"],
        _title_technique_ids(str(result.get("title", ""))),
    ):
        return False
    return True


def grade(judgment: Mapping[str, Any]) -> int:
    return int(judgment.get("grade", 1))


def is_action_judgment(judgment: Mapping[str, Any]) -> bool:
    return any(
        isinstance(judgment.get(field), str)
        and _ACTION_IDENTIFIER_PATTERN.fullmatch(judgment[field]) is not None
        for field in ("title_contains", "text_contains")
    )


def is_action_result(result: Mapping[str, Any]) -> bool:
    title = str(result.get("title", ""))
    result_id = str(result.get("id", ""))
    return title.startswith("Action ") or "actions_catalog.json#" in result_id


def score_action_retrieval(
    results: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    ks: Sequence[int],
) -> Dict[str, float]:
    action_judgments = [judgment for judgment in judgments if is_action_judgment(judgment)]
    if not action_judgments:
        return {}

    metrics: Dict[str, float] = {}
    for k in ks:
        returned_actions = [result for result in results[:k] if is_action_result(result)]
        gains, _ = rank_results(returned_actions, action_judgments)
        true_positives = sum(gain > 0 for gain in gains)
        precision = true_positives / len(returned_actions) if returned_actions else 0.0
        recall = true_positives / len(action_judgments)
        metrics[f"action_precision@{k}"] = precision
        metrics[f"action_recall@{k}"] = recall
        metrics[f"action_f1@{k}"] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return metrics


def candidate_recall(
    candidates: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    reranked_only: bool = False,
) -> float:
    if not judgments:
        return 0.0
    available = [
        candidate for candidate in candidates
        if not reranked_only or candidate.get("rerank_rank") is not None
    ]
    matched_judgments = sum(
        any(
            matches(
                {
                    "type": judgment["type"],
                    "id": candidate.get("id"),
                    "title": candidate.get("title", ""),
                    "text": candidate.get("text", ""),
                },
                judgment,
            )
            for candidate in available
        )
        for judgment in judgments
    )
    return matched_judgments / len(judgments)


def no_action_conflict(
    results: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    k: int,
) -> Optional[float]:
    expects_no_action = any(
        is_action_judgment(judgment)
        and any(
            judgment.get(field) == "no_action"
            for field in ("title_contains", "text_contains")
        )
        for judgment in judgments
    )
    if not expects_no_action:
        return None
    return float(any(
        is_action_result(result)
        and not (
            str(result.get("title", "")).casefold() == "action no_action"
            or str(result.get("id", "")).casefold().endswith("#no_action")
        )
        for result in results[:k]
    ))


def summarize_latency(values: Sequence[float]) -> Dict[str, float | int]:
    if not values:
        return {"queries": 0}
    ordered = sorted(values)
    return {
        "queries": len(ordered),
        "mean_ms": sum(ordered) / len(ordered),
        "median_ms": median(ordered),
        "p95_ms": ordered[max(0, ceil(0.95 * len(ordered)) - 1)],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def rank_results(
    results: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
) -> Tuple[List[int], List[Mapping[str, Any]]]:
    pending = list(judgments)
    gains: List[int] = []
    for result in results:
        candidates = [judgment for judgment in pending if matches(result, judgment)]
        best = max(candidates, key=grade, default=None)
        if best is None:
            gains.append(0)
        else:
            pending.remove(best)
            gains.append(grade(best))
    return gains, pending


def hit_at_k(gains: Sequence[int], k: int) -> float:
    return float(any(gains[:k]))


def recall_at_k(gains: Sequence[int], total_relevant: int, k: int) -> float:
    return sum(1 for gain in gains[:k] if gain) / total_relevant


def precision_at_k(gains: Sequence[int], k: int) -> float:
    return sum(1 for gain in gains[:k] if gain) / k


def reciprocal_rank(gains: Sequence[int]) -> float:
    return next((1 / rank for rank, gain in enumerate(gains, start=1) if gain), 0.0)


def _dcg(gains: Sequence[int]) -> float:
    return sum(gain / log2(rank + 1) for rank, gain in enumerate(gains, start=1))


def ndcg_at_k(gains: Sequence[int], judgments: Sequence[Mapping[str, Any]], k: int) -> float:
    ideal = _dcg(sorted((grade(judgment) for judgment in judgments), reverse=True)[:k])
    return _dcg(gains[:k]) / ideal if ideal else 0.0


def score_ranking(
    results: Sequence[Mapping[str, Any]],
    judgments: Sequence[Mapping[str, Any]],
    ks: Sequence[int],
) -> Tuple[Dict[str, float], List[Mapping[str, Any]]]:
    gains, missed = rank_results(results, judgments)
    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"hit@{k}"] = hit_at_k(gains, k)
        metrics[f"recall@{k}"] = recall_at_k(gains, len(judgments), k)
        metrics[f"precision@{k}"] = precision_at_k(gains, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(gains, judgments, k)
    metrics["mrr"] = reciprocal_rank(gains)
    return metrics, missed


def mean_metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {name: sum(row[name] for row in rows) / len(rows) for name in rows[0]}
