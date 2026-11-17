import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .metrics import MATCH_FIELDS, mean_metrics, score_ranking

DEFAULT_DATASET = Path(__file__).with_name("dataset.json")
DEFAULT_URL = "http://localhost:8000"
DEFAULT_KS = (1, 3, 5, 10)
KINDS = ("case", "kb")
MAX_K = 100  # upper bound accepted by the API for k_cases / k_kb
MAX_CASES_PER_REQUEST = 100  # upper bound accepted by POST /index/cases
LABELS = {"case": "Historical cases", "kb": "Knowledge base", "all": "All"}

Retrieve = Callable[[Mapping[str, Any], int, int], List[dict]]


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_judgment(query_id: str, judgment: Any) -> None:
    if not isinstance(judgment, dict):
        raise ValueError(f"{query_id}: every judgment must be an object")
    unknown = set(judgment) - {"type", "grade", *MATCH_FIELDS}
    if unknown:
        raise ValueError(f"{query_id}: unknown judgment fields {sorted(unknown)}")
    if judgment.get("type") not in KINDS:
        raise ValueError(f"{query_id}: judgment type must be one of {KINDS}")
    if not _positive_int(judgment.get("grade", 1)):
        raise ValueError(f"{query_id}: judgment grade must be a positive integer")
    if not any(field in judgment for field in MATCH_FIELDS):
        raise ValueError(f"{query_id}: a judgment needs at least one of {MATCH_FIELDS}")
    for field in ("title_contains", "text_contains"):
        if field in judgment and not _non_empty_text(judgment[field]):
            raise ValueError(f"{query_id}: {field} must be a non-empty string")
    if "technique_id" in judgment and (
        not isinstance(judgment["technique_id"], str)
        or not re.fullmatch(r"T\d{4}(?:[._]\d{3})?", judgment["technique_id"], re.IGNORECASE)
    ):
        raise ValueError(f"{query_id}: technique_id must be a MITRE ATT&CK ID such as T1059.001")


def validate_dataset(data: Mapping[str, Any]) -> None:
    queries = data.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("the dataset needs a non-empty 'queries' list")

    seen_ids = set()
    for query in queries:
        query_id = query.get("id")
        if not _non_empty_text(query_id) or query_id in seen_ids:
            raise ValueError(f"query ids must be unique non-empty strings (got {query_id!r})")
        seen_ids.add(query_id)

        if not _positive_int(query.get("customer")):
            raise ValueError(f"{query_id}: customer must be a positive integer")

        text, alert = query.get("query"), query.get("alert")
        if (text is None) == (alert is None):
            raise ValueError(f"{query_id}: provide exactly one of 'query' or 'alert'")
        if text is not None and not _non_empty_text(text):
            raise ValueError(f"{query_id}: query must be a non-empty string")
        if alert is not None:
            if not isinstance(alert, dict) or not alert:
                raise ValueError(f"{query_id}: alert must be a non-empty object")
            alert_customer = alert.get("alert_customer_id")
            if alert_customer is not None and alert_customer != query["customer"]:
                raise ValueError(f"{query_id}: customer must match alert.alert_customer_id")

        relevant = query.get("relevant")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"{query_id}: 'relevant' must be a non-empty list")
        for judgment in relevant:
            _validate_judgment(query_id, judgment)

    seen_titles = set()
    for case in data.get("fixture_cases", []):
        if not _positive_int(case.get("customer")):
            raise ValueError("fixture cases need a positive integer customer")
        if not _non_empty_text(case.get("title")) or not _non_empty_text(case.get("text")):
            raise ValueError("fixture cases need a non-empty title and text")
        # judgments locate fixture cases by title, so titles must identify them
        if case["title"] in seen_titles:
            raise ValueError(f"duplicate fixture case title: {case['title']!r}")
        seen_titles.add(case["title"])


def load_dataset(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    validate_dataset(data)
    return data


def parse_ks(text: str) -> List[int]:
    try:
        ks = sorted({int(part) for part in text.split(",") if part.strip()})
    except ValueError:
        raise ValueError(f"invalid --ks value: {text!r}") from None
    if not ks or ks[0] < 1 or ks[-1] > MAX_K:
        raise ValueError(f"--ks values must be between 1 and {MAX_K}")
    return ks


class ServiceClient:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: Optional[dict] = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"{method} {path} failed with HTTP {error.code}: {detail}") from error
        except OSError as error:
            raise RuntimeError(f"cannot reach {self.base_url}: {error}") from error

    def retrieve(self, query: Mapping[str, Any], k_cases: int, k_kb: int) -> List[dict]:
        body: Dict[str, Any] = {"customer": query["customer"], "k_cases": k_cases, "k_kb": k_kb}
        if "alert" in query:
            body["alert"] = query["alert"]
        else:
            body["query"] = query["query"]
        return self._request("POST", "/retrieve", body)

    def index_cases(self, cases: Sequence[Mapping[str, Any]]) -> List[int]:
        ids = self._request("POST", "/index/cases", {"cases": list(cases)})["ids"]
        if len(ids) != len(cases):
            raise RuntimeError("the service returned an unexpected number of case ids")
        return ids

    def delete_case(self, case_id: int) -> None:
        self._request("DELETE", f"/index/cases/{case_id}")


def seed_cases(client: ServiceClient, cases: Sequence[Mapping[str, Any]], seeded: List[int]) -> None:
    # ids are appended as they are created so a failed batch can still be cleaned up
    for start in range(0, len(cases), MAX_CASES_PER_REQUEST):
        batch = [
            {"customer": case["customer"], "title": case["title"], "text": case["text"]}
            for case in cases[start:start + MAX_CASES_PER_REQUEST]
        ]
        seeded.extend(client.index_cases(batch))


def cleanup_cases(client: ServiceClient, seeded: Sequence[int]) -> None:
    for case_id in seeded:
        try:
            client.delete_case(case_id)
        except RuntimeError as error:
            print(f"warning: could not delete fixture case {case_id}: {error}", file=sys.stderr)


def _summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "evaluations": len(rows),
        "metrics": mean_metrics([row["metrics"] for row in rows]),
    }


def evaluate(queries: Sequence[Mapping[str, Any]], retrieve: Retrieve, ks: Sequence[int]) -> Dict[str, Any]:
    """Run every query once and score each result kind (case / kb) as its own ranking.

    The service is asked for the deepest cutoff only; smaller cutoffs are slices of that
    ranking, which holds while the deepest cutoff stays within RERANK_CANDIDATES.
    """
    deepest = max(ks)
    rows: List[Dict[str, Any]] = []
    for query in queries:
        judgments = {
            kind: [judgment for judgment in query["relevant"] if judgment["type"] == kind]
            for kind in KINDS
        }
        results = retrieve(
            query,
            deepest if judgments["case"] else 0,
            deepest if judgments["kb"] else 0,
        )
        for kind in KINDS:
            if not judgments[kind]:
                continue
            ranked = [result for result in results if result.get("type") == kind][:deepest]
            metrics, missed = score_ranking(ranked, judgments[kind], ks)
            rows.append(
                {
                    "query": query["id"],
                    "kind": kind,
                    "relevant": len(judgments[kind]),
                    "returned": len(ranked),
                    "metrics": metrics,
                    "missed": missed,
                    "top": [
                        {"id": r.get("id"), "title": r.get("title"), "score": r.get("score")}
                        for r in ranked[:3]
                    ],
                }
            )

    summary = {
        kind: _summarize([row for row in rows if row["kind"] == kind])
        for kind in KINDS
        if any(row["kind"] == kind for row in rows)
    }
    summary["all"] = _summarize(rows)
    return {"ks": list(ks), "summary": summary, "queries": rows}


def _describe(judgment: Mapping[str, Any]) -> str:
    matchers = ", ".join(f"{field}={judgment[field]!r}" for field in MATCH_FIELDS if field in judgment)
    return f"{judgment['type']} {matchers} (grade {judgment.get('grade', 1)})"


def format_report(report: Mapping[str, Any]) -> str:
    ks = report["ks"]
    lines: List[str] = []
    for name, section in report["summary"].items():
        lines.append(f"{LABELS[name]} - {section['evaluations']} evaluations")
        lines.append(" " * 13 + "".join(f"{'@' + str(k):>8}" for k in ks))
        for metric in ("hit", "recall", "precision", "ndcg"):
            values = "".join(f"{section['metrics'][f'{metric}@{k}']:>8.3f}" for k in ks)
            lines.append(f"  {metric:<11}{values}")
        lines.append(f"  {'mrr':<11}{section['metrics']['mrr']:>8.3f}")
        lines.append("")

    misses = [row for row in report["queries"] if row["missed"]]
    if misses:
        lines.append(f"Judgments not found in the top {max(ks)}:")
        for row in misses:
            lines.append(f"  [{row['query']}] {row['kind']}")
            lines.extend(f"    missing: {_describe(judgment)}" for judgment in row["missed"])
            top = "; ".join(str(item["title"]) for item in row["top"]) or "no results"
            lines.append(f"    top results: {top}")
    return "\n".join(lines).rstrip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure the retrieval quality of a running RAG service through POST /retrieve."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the RAG service")
    parser.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)), help="Comma-separated cutoffs")
    parser.add_argument(
        "--seed-cases",
        action="store_true",
        help="Index the dataset's fixture cases before the run and delete them afterwards",
    )
    parser.add_argument("--timeout", type=float, default=120, help="Per-request timeout in seconds")
    parser.add_argument("--output", type=Path, help="Write the full report as JSON")
    args = parser.parse_args(argv)

    try:
        ks = parse_ks(args.ks)
        dataset = load_dataset(args.dataset)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    fixtures = dataset.get("fixture_cases", [])
    if fixtures and not args.seed_cases:
        print(
            f"warning: the dataset defines {len(fixtures)} fixture cases but --seed-cases was not "
            "given; case metrics only reflect cases that are already indexed",
            file=sys.stderr,
        )

    client = ServiceClient(args.url, args.timeout)
    seeded: List[int] = []
    try:
        if args.seed_cases:
            seed_cases(client, fixtures, seeded)
        report = evaluate(dataset["queries"], client.retrieve, ks)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        cleanup_cases(client, seeded)

    print(format_report(report))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
