import json
import logging
import re
from collections import Counter
from typing import Dict, List, Literal, Optional, Tuple
import config
from .embeddings import embed_query
from .lexical import LexicalIndex, LexicalIndexCache
from .query_expansion import attack_query_expansions
from .reranker import rerank
from .text_cleaner import clean_retrieval_text, split_sentences
from .vector_store import get_collection, index_generation

logger = logging.getLogger(__name__)

_lexical_cache = LexicalIndexCache()


# ============================================================
# Utilities on candidates
# ============================================================

def _hit_key(hit: dict) -> str:
    metadata = hit["metadata"]
    source = metadata.get("source")
    chunk_index = metadata.get("chunk_index")
    if source is not None and chunk_index is not None:
        return f"{source}::{chunk_index}"
    return hit["text"]


def _candidate(text: str, metadata: Optional[dict], distance: Optional[float], origin: str,
               matched_expansion: Optional[str] = None) -> Optional[dict]:
    text = clean_retrieval_text(text or "")
    if not text:
        return None
    candidate = {
        "text": text,
        "metadata": metadata or {},
        "distance": distance,
        "origin": origin,
        "matched_expansion": matched_expansion,
        "vector_rank": None,
        "lexical_corpus_rank": None,   # BM25 rank on filtered entire corpus
        "lexical_rank": None,          # BM25 rank relative to the reranked pool
        "cheap_score": 0.0,
        "rerank_score": None,
        "rerank_rank": None,
        "fused_score": None,
        "fused_rank": None,
    }
    candidate["key"] = _hit_key(candidate)
    return candidate


def _diversity_key(hit: dict) -> str:
    source = str(hit["metadata"].get("source") or "")
    match = config._TECHNIQUE_ID.search(source) or config._TECHNIQUE_ID.search(hit["text"][:300])
    if not match:
        return hit["key"]
    return f"T{match.group(1)}" + (f".{match.group(2)}" if match.group(2) else "")


def _validate_customer(customer_id) -> None:
    if isinstance(customer_id, bool) or not isinstance(customer_id, int) or customer_id <= 0:
        raise ValueError("customer_id must be a positive integer for tenant-scoped retrieval")


def _where(result_kind: Optional[str], customer_id: int) -> dict:
    if result_kind == "case":
        return {"$and": [{"category": "incident_history"}, {"customer": customer_id}]}
    if result_kind == "kb":
        return {"$and": [{"scope": "global"}, {"category": {"$ne": "incident_history"}}]}
    return {"$or": [{"scope": "global"}, {"customer": customer_id}]}


# ============================================================
# Lexical index (BM25)
# ============================================================

def _load_lexical_index(collection, where: dict) -> LexicalIndex:
    data = collection.get(where=where, include=["documents", "metadatas"])
    texts = [document or "" for document in data["documents"]]
    metadatas = [metadata or {} for metadata in data["metadatas"]]
    keys = [_hit_key({"text": t, "metadata": m}) for t, m in zip(texts, metadatas)]
    return LexicalIndex(keys, texts, metadatas)


def _get_lexical_index(collection, where: dict) -> Optional[LexicalIndex]:
    try:
        return _lexical_cache.get(
            json.dumps(where, sort_keys=True),
            (collection.count(), index_generation()),
            config.LEXICAL_INDEX_TTL_SECONDS,
            lambda: _load_lexical_index(collection, where),
        )
    except Exception:
        logger.warning("Lexical index unavailable; using vector candidates only", exc_info=True)
        return None


def _from_ranking(index: LexicalIndex, ranking: List[Tuple[int, float]], known: set,
                  limit: int, origin: str, expansion: Optional[str] = None) -> List[dict]:
    added: List[dict] = []
    if limit <= 0:
        return added
    for position, _ in ranking:
        if len(added) >= limit:
            break
        key = index.keys[position]
        if key in known:
            continue
        candidate = _candidate(index.texts[position], index.metadatas[position], None, origin, expansion)
        if candidate is None:
            continue
        known.add(key)
        added.append(candidate)
    return added


def _category_vector(collection, embedding: List[float], known: set, category: str, limit: int) -> List[dict]:
    if limit <= 0:
        return []
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where={"$and": [{"scope": "global"}, {"category": category}]},
        )
    except Exception:
        logger.warning("Could not retrieve %s candidates", category, exc_info=True)
        return []
    added: List[dict] = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        candidate = _candidate(doc, meta, dist, "category")
        if candidate is not None and candidate["key"] not in known:
            known.add(candidate["key"])
            added.append(candidate)
    return added


def _named_action(collection, name: str, known: set) -> List[dict]:
    try:
        found = collection.get(
            where={"$and": [{"scope": "global"}, {"category": config.ACTION_CATEGORY}]},
            include=["documents", "metadatas"],
        )
    except Exception:
        logger.warning("Could not load catalog action %s", name, exc_info=True)
        return []
    for doc, meta in zip(found["documents"], found["metadatas"]):
        if str((meta or {}).get("source", "")).endswith(f"#{name}"):
            candidate = _candidate(doc, meta, None, "category_lexical")
            if candidate is not None and candidate["key"] not in known:
                known.add(candidate["key"])
                return [candidate]
    return []


# ============================================================
# Fusion and selection
# ============================================================

def _fuse(reranked: List[dict]) -> List[dict]:
    rerank_w = config.RERANK_FUSION_WEIGHT
    lexical_w = config.LEXICAL_FUSION_WEIGHT
    vector_w = config.VECTOR_FUSION_WEIGHT
    k = config._RRF_K

    with_lexical = sorted(
        (c for c in reranked if c["lexical_corpus_rank"] is not None),
        key=lambda c: c["lexical_corpus_rank"],
    )
    for rank, candidate in enumerate(with_lexical, start=1):
        candidate["lexical_rank"] = rank

    for candidate in reranked:
        score = rerank_w / (k + candidate["rerank_rank"])
        if candidate["lexical_rank"]:
            score += lexical_w / (k + candidate["lexical_rank"])
        if candidate["vector_rank"]:
            score += vector_w / (k + candidate["vector_rank"])
        candidate["fused_score"] = score

    fused = sorted(reranked, key=lambda c: (-c["fused_score"], c["rerank_rank"]))
    for rank, candidate in enumerate(fused, start=1):
        candidate["fused_rank"] = rank
    return fused


def select_hits(
    ranked_hits: List[dict],
    top_k: int,
    min_score: Optional[float] = None,
    margin: Optional[float] = None,
    diversify: bool = True,
) -> List[dict]:

    min_score = config.RERANK_MIN_SCORE if min_score is None else min_score
    margin = config.RERANK_SCORE_MARGIN if margin is None else margin
    best = max((h["rerank_score"] for h in ranked_hits if h.get("rerank_score") is not None), default=None)

    selected: List[dict] = []
    seen = set()
    for hit in ranked_hits:
        key = _diversity_key(hit) if diversify else hit["key"]
        if key in seen:
            continue
        score = hit.get("rerank_score")
        if score is not None and (
            (min_score is not None and score < min_score)
            or (margin is not None and best is not None and score < best - margin)
        ):
            continue
        seen.add(key)
        selected.append(hit)
        if len(selected) >= top_k:
            break
    return selected


def merge_chunks(current: str, following: str, title: Optional[str] = None) -> str:
    if title and following.startswith(f"{title}: "):
        following = following[len(title) + 2:]
    current_sentences = split_sentences(current)
    following_sentences = split_sentences(following)
    for shared in range(min(len(current_sentences), len(following_sentences)), 0, -1):
        if current_sentences[-shared:] == following_sentences[:shared]:
            following_sentences = following_sentences[shared:]
            break
    return " ".join([current, *following_sentences]).strip()


def expand_hits(collection, hits: List[dict], count: int) -> None:

    for hit in hits[:count]:
        metadata = hit["metadata"]
        index = metadata.get("chunk_index")
        source = metadata.get("source")
        if not isinstance(index, int) or isinstance(index, bool) or not source:
            continue
        try:
            found = collection.get(ids=[f"{source}::{index + 1}"], include=["documents"])
            documents = found.get("documents") or []
        except Exception:
            logger.warning("Could not load the chunk following %s::%s", source, index, exc_info=True)
            continue
        if documents:
            hit["text"] = merge_chunks(hit["text"], clean_retrieval_text(documents[0]), metadata.get("title"))


# ============================================================
# Diagnostics
# ============================================================

def _diagnostics(pool: List[dict], selected: List[dict], expansions: List[str], result_kind: Optional[str]) -> dict:
    returned = {c["key"]: rank for rank, c in enumerate(selected, start=1)}
    origins = Counter(c["origin"] for c in pool)
    candidates = []
    for c in pool:
        item = {
            "id": c["key"],
            "source": c["metadata"].get("source", ""),
            "title": c["metadata"].get("title") or c["metadata"].get("source", ""),
            "origin": c["origin"],
            "reranked": c["rerank_rank"] is not None,
            "distance": None if c["distance"] is None else float(c["distance"]),
            "cheap_score": c["cheap_score"],
            "rerank_score": c["rerank_score"],
            "fused_score": c["fused_score"],
            "vector_rank": c["vector_rank"],
            "rerank_rank": c["rerank_rank"],
            "lexical_rank": c["lexical_rank"],
            "lexical_corpus_rank": c["lexical_corpus_rank"],
            "fused_rank": c["fused_rank"],
            "returned_rank": returned.get(c["key"]),
        }
        if c["matched_expansion"]:
            item["matched_expansion"] = c["matched_expansion"]
        if result_kind == "kb":
            item["text"] = c["text"]
        candidates.append(item)
    return {
        "candidate_count": len(candidates),
        "reranked_count": sum(1 for c in pool if c["rerank_rank"] is not None),
        "category_added": origins["category"],
        "category_lexical_added": origins["category_lexical"],
        "attack_expansion_added": origins["attack_expansion"],
        "attack_expansions": expansions,
        "lexical_added": origins["lexical"],
        "candidates": candidates,
    }


# ============================================================
# Main pipeline
# ============================================================

def _retrieve(
    query: str,
    customer_id: int,
    top_k: int,
    result_kind: Optional[Literal["case", "kb"]],
    diagnostics: Optional[dict] = None,
    query_embedding: Optional[List[float]] = None,
) -> List[dict]:
    _validate_customer(customer_id)

    collection = get_collection()
    total = collection.count()
    if total == 0:
        if diagnostics is not None:
            diagnostics.update({"candidate_count": 0, "candidates": []})
        return []

    where = _where(result_kind, customer_id)
    embedding = query_embedding if query_embedding is not None else embed_query(query)
    is_kb = result_kind != "case"

    # --- 1. Dense vector search ---
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(max(top_k, config.VECTOR_SEARCH_CANDIDATES), total),
        where=where,
    )
    known: set = set()
    vector_hits: List[dict] = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        candidate = _candidate(doc, meta, dist, "vector")
        if candidate is None or candidate["key"] in known:
            continue
        candidate["vector_rank"] = len(vector_hits) + 1
        known.add(candidate["key"])
        vector_hits.append(candidate)
    pool: List[dict] = list(vector_hits)

    # --- 2. Catalog actions (KB only) ---
    if is_kb:
        action_intent = bool(config._ACTION_INTENT.search(query)) or any(
            c["metadata"].get("category") == config.ACTION_CATEGORY for c in vector_hits
        )
        action_limit = config.ACTION_CANDIDATES if action_intent else min(config.ACTION_CANDIDATES, config.ACTION_UNGATED_CANDIDATES)
        pool += _category_vector(collection, embedding, known, config.ACTION_CATEGORY, action_limit)

        if action_intent and config.ACTION_LEXICAL_CANDIDATES > 0:
            action_where = {"$and": [{"scope": "global"}, {"category": config.ACTION_CATEGORY}]}
            action_index = _get_lexical_index(collection, action_where)
            if action_index is not None:
                pool += _from_ranking(
                    action_index, action_index.search(query), known,
                    config.ACTION_LEXICAL_CANDIDATES, "category_lexical",
                )

        if config._BENIGN_INTENT.search(query):
            pool += _named_action(collection, config.NO_ACTION_NAME, known)

    # --- 3. BM25 on the corpus, ATT&CK expansions, lexical candidates ---
    expansions = attack_query_expansions(query) if is_kb else []
    use_expansions = bool(expansions) and config.ATTACK_EXPANSION_CANDIDATES > 0
    index = _get_lexical_index(collection, where) if (config.LEXICAL_CANDIDATES > 0 or use_expansions) else None
    ranking = index.search(query) if index is not None else []

    if index is not None:
        if use_expansions:
            for expansion in expansions:
                pool += _from_ranking(
                    index, index.search(expansion), known,
                    config.ATTACK_EXPANSION_CANDIDATES, "attack_expansion", expansion,
                )
        pool += _from_ranking(index, ranking, known, config.LEXICAL_CANDIDATES, "lexical")

    # --- 4. Cheap pre-selection (RRF dense + BM25) before the cross-encoder ---
    k = config._RRF_K
    lexical_rank = {position: rank for rank, (position, _) in enumerate(ranking, start=1)}
    for c in pool:
        if index is not None:
            position = index.position_of(c["key"])
            c["lexical_corpus_rank"] = lexical_rank.get(position) if position is not None else None
        score = 0.0
        if c["vector_rank"]:
            score += 1.0 / (k + c["vector_rank"])
        if c["lexical_corpus_rank"]:
            score += 1.0 / (k + c["lexical_corpus_rank"])
        c["cheap_score"] = score

    protected = [c for c in pool if c["origin"] in config._PROTECTED_ORIGINS]
    others = sorted((c for c in pool if c["origin"] not in config._PROTECTED_ORIGINS), key=lambda c: -c["cheap_score"])
    to_rerank = others[:config.RERANK_POOL_SIZE] + protected

    # --- 5. Cross-encoder only on the reduced pool, then fusion and selection ---
    reranked = rerank(query, to_rerank, top_k=len(to_rerank))
    for rank, c in enumerate(reranked, start=1):
        c["rerank_rank"] = rank

    fused = _fuse(reranked)
    selected = select_hits(fused, top_k, diversify=is_kb)
    expand_hits(collection, selected, config.CONTEXT_EXPAND_TOP)

    if diagnostics is not None:
        diagnostics.update(_diagnostics(pool, selected, expansions, result_kind))
    return selected


# ============================================================
# API
# ============================================================

def retrieve(
    query: str,
    customer_id: int,
    top_k: int = config.TOP_K,
    result_kind: Optional[Literal["case", "kb"]] = None,
) -> List[dict]:
    return _retrieve(query, customer_id, top_k, result_kind)


def retrieve_with_diagnostics(
    query: str,
    customer_id: int,
    top_k: int = config.TOP_K,
    result_kind: Optional[Literal["case", "kb"]] = None,
) -> Tuple[List[dict], dict]:
    diagnostics: dict = {}
    hits = _retrieve(query, customer_id, top_k, result_kind, diagnostics)
    return hits, diagnostics


def retrieve_results(query: str, customer_id: int, k_cases: int, k_kb: int) -> List[dict]:
    results, _ = _retrieve_results(query, customer_id, k_cases, k_kb)
    return results


def retrieve_results_with_diagnostics(
    query: str,
    customer_id: int,
    k_cases: int,
    k_kb: int,
) -> Tuple[List[dict], Dict[str, dict]]:
    diagnostics: Dict[str, dict] = {}
    results, _ = _retrieve_results(query, customer_id, k_cases, k_kb, diagnostics)
    return results, diagnostics


def _retrieve_results(
    query: str,
    customer_id: int,
    k_cases: int,
    k_kb: int,
    diagnostics: Optional[Dict[str, dict]] = None,
) -> Tuple[List[dict], Dict[str, dict]]:
    _validate_customer(customer_id)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    for name, limit in (("k_cases", k_cases), ("k_kb", k_kb)):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 100:
            raise ValueError(f"{name} must be an integer between 0 and 100")

    embedding = embed_query(query) if (k_cases or k_kb) else None

    results = []
    for result_kind, limit in (("case", k_cases), ("kb", k_kb)):
        if limit == 0:
            continue
        kind_diagnostics: Optional[dict] = {} if diagnostics is not None else None
        hits = _retrieve(query, customer_id, limit, result_kind, kind_diagnostics, embedding)
        if diagnostics is not None:
            diagnostics[result_kind] = kind_diagnostics

        for hit in hits:
            metadata = hit["metadata"]
            record_id = metadata.get("record_id")
            if result_kind == "case":
                if isinstance(record_id, bool) or not isinstance(record_id, int):
                    raise ValueError("Retrieved case is missing a numeric record_id")
                result_id = record_id
            else:
                result_id = record_id or metadata.get("source", "")
            results.append(
                {
                    "type": result_kind,
                    "id": result_id,
                    "title": metadata.get("title") or metadata.get("source", ""),
                    "text": hit["text"],
                    "score": float(hit["fused_score"]),
                }
            )

    return sorted(results, key=lambda r: r["score"], reverse=True), diagnostics or {}
