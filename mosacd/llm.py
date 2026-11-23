from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import networkx as nx
from openai import OpenAI
from .propagation import evidence_for_edge


ANSWER_RE = re.compile(r"<\s*answer\s*>\s*([ABCDE])\s*<\s*/\s*answer\s*>", re.I)


@dataclass
class ChatResult:
    text: str
    model: str
    latency_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    total_tokens: int = 0
    provider_cost_usd: float | None = None
    estimated_cost_usd: float | None = None
    cache_hit: bool = False
    request_id: str | None = None
    cache_key: str | None = None


@dataclass
class UsageSummary:
    requests: int = 0
    live_requests: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_latency_seconds: float = 0.0
    provider_cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    provider_cost_available: bool = False
    estimated_cost_available: bool = False


class TrackedChatClient:
    """OpenAI-compatible chat client with raw logging and cost accounting.

    Credentials are read only from environment variables. No key is stored in
    this repository. Models containing ``/`` default to OpenRouter; all others
    default to OpenAI, unless ``base_url`` is supplied explicitly.
    """

    def __init__(
        self,
        *,
        pricing: Mapping[str, Mapping[str, float | None]] | None = None,
        log_path: str | Path | None = None,
        response_cache_path: str | Path | None = None,
        use_response_cache: bool = False,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        self.pricing = dict(pricing or {})
        self.log_path = Path(log_path) if log_path else None
        self.response_cache_path = Path(response_cache_path) if response_cache_path else None
        self.use_response_cache = use_response_cache
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.summary_data = UsageSummary()
        self._cache: dict[str, dict[str, Any]] = {}
        self._clients: dict[tuple[str | None, str], OpenAI] = {}
        if self.response_cache_path and self.response_cache_path.exists():
            for line in self.response_cache_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if row.get("cache_key"):
                        self._cache[row["cache_key"]] = row
                except json.JSONDecodeError:
                    continue

    @staticmethod
    def load_pricing(path: str | Path | None) -> dict[str, dict[str, float | None]]:
        if path is None:
            return {}
        return json.loads(Path(path).read_text())

    def _provider_config(self, model: str) -> tuple[str | None, str]:
        if self.base_url:
            env = self.api_key_env or "OPENAI_API_KEY"
            return self.base_url, env
        if "/" in model:
            return "https://openrouter.ai/api/v1", self.api_key_env or "OPENROUTER_API_KEY"
        return None, self.api_key_env or "OPENAI_API_KEY"

    def _client(self, model: str) -> OpenAI:
        base_url, env_name = self._provider_config(model)
        key = os.getenv(env_name)
        if not key:
            raise RuntimeError(
                f"Missing {env_name}. Set it in the environment before running MosaCD."
            )
        cache_key = (base_url, env_name)
        if cache_key not in self._clients:
            kwargs: dict[str, Any] = {"api_key": key}
            if base_url:
                kwargs["base_url"] = base_url
            self._clients[cache_key] = OpenAI(**kwargs)
        return self._clients[cache_key]

    @staticmethod
    def _hash_request(payload: Mapping[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def _obj_get(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, Mapping):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def _extract_usage(self, response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        prompt = int(self._obj_get(usage, "prompt_tokens", 0) or 0)
        completion = int(self._obj_get(usage, "completion_tokens", 0) or 0)
        total = int(self._obj_get(usage, "total_tokens", prompt + completion) or 0)
        details = self._obj_get(usage, "prompt_tokens_details", None)
        cached = int(self._obj_get(details, "cached_tokens", 0) or 0)
        provider_cost = self._obj_get(usage, "cost", None)
        if provider_cost is None:
            extra = getattr(response, "model_extra", None) or {}
            provider_cost = extra.get("cost") or (extra.get("usage") or {}).get("cost")
        try:
            provider_cost = float(provider_cost) if provider_cost is not None else None
        except (TypeError, ValueError):
            provider_cost = None
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_prompt_tokens": cached,
            "total_tokens": total,
            "provider_cost_usd": provider_cost,
        }

    def _estimate_cost(
        self, model: str, prompt_tokens: int, cached_tokens: int, completion_tokens: int
    ) -> float | None:
        rates = self.pricing.get(model)
        if not rates:
            return None
        in_rate = rates.get("input_per_million")
        out_rate = rates.get("output_per_million")
        cached_rate = rates.get("cached_input_per_million", in_rate)
        if in_rate is None or out_rate is None or cached_rate is None:
            return None
        uncached = max(prompt_tokens - cached_tokens, 0)
        return (
            uncached * float(in_rate)
            + cached_tokens * float(cached_rate)
            + completion_tokens * float(out_rate)
        ) / 1_000_000.0

    def _record(self, row: dict[str, Any]) -> None:
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
        if self.response_cache_path and not row.get("cache_hit"):
            self.response_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.response_cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, default=str) + "\n")
            self._cache[row["cache_key"]] = row

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        model: str,
        temperature: float,
        top_p: float | None = None,
        seed: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ChatResult:
        payload = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "messages": list(messages),
        }
        cache_key = self._hash_request(payload)
        if self.use_response_cache and cache_key in self._cache:
            row = self._cache[cache_key]
            result = ChatResult(
                text=row.get("text", ""),
                model=model,
                latency_seconds=0.0,
                prompt_tokens=int(row.get("prompt_tokens", 0)),
                completion_tokens=int(row.get("completion_tokens", 0)),
                cached_prompt_tokens=int(row.get("cached_prompt_tokens", 0)),
                total_tokens=int(row.get("total_tokens", 0)),
                provider_cost_usd=row.get("provider_cost_usd"),
                estimated_cost_usd=0.0,
                cache_hit=True,
                request_id=row.get("request_id"),
                cache_key=cache_key,
            )
            self._update_summary(result)
            return result

        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": 1.0 if model == "gpt-5" else temperature,
            "messages": list(messages),
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if seed is not None:
            kwargs["seed"] = seed

        start = time.perf_counter()
        response = self._client(model).chat.completions.create(**kwargs)
        latency = time.perf_counter() - start
        text = response.choices[0].message.content or ""
        usage = self._extract_usage(response)
        estimated = self._estimate_cost(
            model,
            usage["prompt_tokens"],
            usage["cached_prompt_tokens"],
            usage["completion_tokens"],
        )
        result = ChatResult(
            text=text,
            model=model,
            latency_seconds=latency,
            estimated_cost_usd=estimated,
            request_id=getattr(response, "id", None),
            cache_key=cache_key,
            **usage,
        )
        row = {
            **asdict(result),
            "cache_hit": False,
            "messages": list(messages),
            "metadata": dict(metadata or {}),
            "created_at_unix": time.time(),
        }
        self._record(row)
        self._update_summary(result)
        return result

    def _update_summary(self, result: ChatResult) -> None:
        s = self.summary_data
        s.requests += 1
        if result.cache_hit:
            s.cache_hits += 1
        else:
            s.live_requests += 1
        s.prompt_tokens += result.prompt_tokens
        s.cached_prompt_tokens += result.cached_prompt_tokens
        s.completion_tokens += result.completion_tokens
        s.total_tokens += result.total_tokens
        s.llm_latency_seconds += result.latency_seconds
        if result.provider_cost_usd is not None:
            s.provider_cost_usd += result.provider_cost_usd
            s.provider_cost_available = True
        if result.estimated_cost_usd is not None:
            s.estimated_cost_usd += result.estimated_cost_usd
            s.estimated_cost_available = True

    def summary(self) -> dict[str, Any]:
        return asdict(self.summary_data)


PROMPT_FULL = """
You are a senior researcher in causal discovery. We are studying the following dataset:

{data_desc}

The two target variables under review are {u} and {v}.

Conditional-independence tests mentioning these variables:

{ci_bullets}

Neighbour chain(s) that must normally remain non-collider:

{chains}

The nodes involved are described as below:

{node_desc}

Choose one explanation that best fits domain knowledge and/or decides a CI test is unreliable (avoid selecting D or E unless other options are strongly against common sense):

A. Undecided. We don't know enough to confidently pick a directionality.
B. Changing the state of {u} causally affects {v}, and {v} causally affects {u_theOther_2v}.
C. Changing the state of {v} causally affects {u}, and {u} causally affects {v_theOther_2u}.
D. Changing the state of {u} causally affects {v}, and {u_theOther_2v} also causally affects {v}, **violating corresponding CI tests**.
E. Changing the state of {v} causally affects {u}, and {v_theOther_2u} also causally affects {u}, **violating corresponding CI tests**.

Think step-by-step before selecting:
1. Mechanisms - What known causal pathways (biological, physical, etc.) support each direction?
2. Counterfactual test - What would happen if we intervened on one node? What would we expect?
3. Empirical check - Point to one key piece of information that favors/weakens a direction.
4. Comparison - Briefly weigh A vs B vs C vs D vs E and choose the most plausible.

Return exactly three lines:
1. Reasoning in support of one direction.
2. Reasoning against the weaker/less plausible direction.
3. Final choice: <Answer>A/B/C/D/E</Answer>
"""


PROMPT_NONE2U = """
You are a senior researcher in causal discovery. We are studying the following dataset:

{data_desc}

The two target variables under review are {u} and {v}.

Conditional-independence tests mentioning these variables:

{ci_bullets}

Neighbour chain(s) that must normally remain non-collider:

{chains}

The nodes involved are described as below:

{node_desc}

Choose one explanation that best fits domain knowledge and/or decides a CI test is unreliable (avoid selecting D unless other options are strongly against common sense):

A. Undecided. We don't know enough to confidently pick a directionality.
B. Changing the state of {u} causally affects {v}, and {v} causally affects {u_theOther_2v}.
C. Changing the state of {v} causally affects {u}.
D. Changing the state of {u} causally affects {v}, and {u_theOther_2v} also causally affects {v}, **violating corresponding CI tests**.

Think step-by-step before selecting:
1. Mechanisms - What known causal pathways (biological, physical, etc.) support each direction?
2. Counterfactual test - What would happen if we intervened on one node? What would we expect?
3. Empirical check - Point to one key piece of information that favors/weakens a direction.
4. Comparison - Briefly weigh A vs B vs C vs D and choose the most plausible.

Return exactly three lines:
1. Reasoning in support of one direction.
2. Reasoning against the weaker/less plausible direction.
3. Final choice: <Answer>A/B/C/D</Answer>
"""


PROMPT_NONE2V = """
You are a senior researcher in causal discovery. We are studying the following dataset:

{data_desc}

The two target variables under review are {u} and {v}.

Conditional-independence tests mentioning these variables:

{ci_bullets}

Neighbour chain(s) that must normally remain non-collider:

{chains}

The nodes involved are described as below:

{node_desc}

Choose one explanation that best fits domain knowledge and/or decides a CI test is unreliable (avoid selecting D unless other options are strongly against common sense):

A. Undecided. We don't know enough to confidently pick a directionality.
B. Changing the state of {u} causally affects {v}.
C. Changing the state of {v} causally affects {u}, and {u} causally affects {v_theOther_2u}.
D. Changing the state of {v} causally affects {u}, and {v_theOther_2u} also causally affects {u}, **violating corresponding CI tests**.

Think step-by-step before selecting:
1. Mechanisms - What known causal pathways (biological, physical, etc.) support each direction?
2. Counterfactual test - What would happen if we intervened on one node? What would we expect?
3. Empirical check - Point to one key piece of information that favors/weakens a direction.
4. Comparison - Briefly weigh A vs B vs C vs D and choose the most plausible.

Return exactly three lines:
1. Reasoning in support of one direction.
2. Reasoning against the weaker/less plausible direction.
3. Final choice: <Answer>A/B/C/D</Answer>
"""


PROMPT_NONE = """
You are a senior researcher in causal discovery. We are studying the following dataset:

{data_desc}

The two target variables under review are {u} and {v}.

The nodes involved are described as below:

{node_desc}

Choose one explanation that best fits domain knowledge:

A. Undecided. We don't know enough to confidently pick a directionality.
B. Changing the state of {u} causally affects {v}.
C. Changing the state of {v} causally affects {u}.

Think step-by-step before selecting:
1. Mechanisms - What known causal pathways (biological, physical, etc.) support each direction?
2. Counterfactual test - What would happen if we intervened on one node? What would we expect?
3. Empirical check - Point to one key piece of information that favors/weakens a direction.
4. Comparison - Briefly weigh A vs B vs C and choose the most plausible.

Return exactly three lines:
1. Reasoning in support of one direction.
2. Reasoning against the weaker/less plausible direction.
3. Final choice: <Answer>A/B/C</Answer>
"""


def parse_answer(text: str) -> str | None:
    match = ANSWER_RE.search(text or "")
    return match.group(1).upper() if match else None


def _ci_bullets(
    u,
    v,
    u_other_to_v,
    v_other_to_u,
    separating_sets: Mapping,
    pval_sets: Mapping,
) -> str:
    lines = []
    for other in u_other_to_v:
        key = frozenset((u, other))
        seps = separating_sets[key]
        p = pval_sets[key]
        s_txt = ", ".join(map(str, seps[0]))
        lines.append(
            f"{u} is not rejected as independent of {other} given {{{s_txt}}} "
            f"(p={p:.3g}); the {u}--{other} edge is removed."
        )
    for other in v_other_to_u:
        key = frozenset((v, other))
        seps = separating_sets[key]
        p = pval_sets[key]
        s_txt = ", ".join(map(str, seps[0]))
        lines.append(
            f"{v} is not rejected as independent of {other} given {{{s_txt}}} "
            f"(p={p:.3g}); the {v}--{other} edge is removed."
        )
    return "\n".join(lines)


def build_mosacd_prompt(
    *,
    u,
    v,
    data_description: str,
    node_description: Mapping,
    undir_graph: nx.Graph,
    separating_sets: Mapping,
    pval_sets: Mapping,
) -> tuple[str, str]:
    u_other_to_v, v_other_to_u = evidence_for_edge(
        u, v, undir_graph, separating_sets
    )
    related = list(dict.fromkeys([u, v, *u_other_to_v, *v_other_to_u]))
    node_desc = "\n".join(
        f"- **{node}**: {node_description.get(node, str(node))}" for node in related
    )
    bullets = _ci_bullets(
        u, v, u_other_to_v, v_other_to_u, separating_sets, pval_sets
    )
    common = {
        "data_desc": data_description,
        "node_desc": node_desc,
        "u": u,
        "v": v,
        "u_theOther_2v": u_other_to_v,
        "v_theOther_2u": v_other_to_u,
        "ci_bullets": bullets,
    }
    if not u_other_to_v and not v_other_to_u:
        return PROMPT_NONE.format(**common), "none"
    if u_other_to_v and not v_other_to_u:
        common["chains"] = f"{u} --- {v} --- {u_other_to_v}"
        return PROMPT_NONE2U.format(**common), "none2u"
    if not u_other_to_v and v_other_to_u:
        common["chains"] = f"{v_other_to_u} --- {u} --- {v}"
        return PROMPT_NONE2V.format(**common), "none2v"
    common["chains"] = f"{v_other_to_u} --- {u} --- {v} --- {u_other_to_v}"
    return PROMPT_FULL.format(**common), "full"


def answer_to_direction(answer: str | None, prompt_kind: str) -> tuple[str | None, bool]:
    """Map an answer to fwd/bwd and whether it explicitly violates CI evidence."""
    if answer is None or answer == "A":
        return None, False
    if answer == "B":
        return "fwd", False
    if answer == "C":
        return "bwd", False
    if answer == "D":
        if prompt_kind in {"full", "none2u"}:
            return "fwd", True
        if prompt_kind == "none2v":
            return "bwd", True
    if answer == "E" and prompt_kind == "full":
        return "bwd", True
    return None, False


def query_mosacd_votes(
    *,
    dataset: str,
    data_description: str,
    node_description: Mapping,
    skeleton: nx.Graph,
    separating_sets: Mapping,
    pval_sets: Mapping,
    client: TrackedChatClient,
    model: str = "gpt-4o-mini",
    temperature: float = 0.6,
    n_votes: int = 5,
    seed: int | None = None,
    raw_vote_path: str | Path | None = None,
) -> tuple[Counter, list[dict[str, Any]]]:
    votes: Counter = Counter()
    raw_records: list[dict[str, Any]] = []
    raw_path = Path(raw_vote_path) if raw_vote_path else None
    if raw_path:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text("")

    for edge_u, edge_v in skeleton.edges():
        for order_index, (u, v) in enumerate(((edge_u, edge_v), (edge_v, edge_u))):
            prompt, prompt_kind = build_mosacd_prompt(
                u=u,
                v=v,
                data_description=data_description,
                node_description=node_description,
                undir_graph=skeleton,
                separating_sets=separating_sets,
                pval_sets=pval_sets,
            )
            for repeat in range(n_votes):
                result = client.complete(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    metadata={
                        "method": "mosacd",
                        "dataset": dataset,
                        "edge_u": edge_u,
                        "edge_v": edge_v,
                        "source": u,
                        "target": v,
                        "order_index": order_index,
                        "repeat": repeat,
                        "prompt_kind": prompt_kind,
                    },
                )
                answer = parse_answer(result.text)
                direction, violates_ci = answer_to_direction(answer, prompt_kind)
                if direction:
                    votes[(u, v, direction)] += 1
                record = {
                    "dataset": dataset,
                    "edge": [edge_u, edge_v],
                    "source": u,
                    "target": v,
                    "order_index": order_index,
                    "repeat": repeat,
                    "prompt_kind": prompt_kind,
                    "answer": answer,
                    "direction": direction,
                    "violates_ci": violates_ci,
                    **asdict(result),
                }
                raw_records.append(record)
                if raw_path:
                    with raw_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, default=str) + "\n")
    return votes, raw_records


def stage_seed_sets(
    skeleton: nx.Graph,
    votes: Mapping,
    raw_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct raw, cross-order, and CI/cycle-filtered seed sets.

    The legacy aggregate vote CSVs do not retain B/C versus D/E answer codes.
    In that case, the CI-answer filter is marked unavailable and only cycle
    filtering is applied at the final seed stage.
    """
    raw_directed = []
    raw_undirected = []
    cross_directed = []
    cross_undirected = []
    candidates: list[dict[str, Any]] = []

    grouped: dict[tuple, list] = defaultdict(list)
    if raw_records:
        for row in raw_records:
            grouped[(row["source"], row["target"])].append(row)

    for u, v in skeleton.edges():
        uv_fwd = int(votes.get((u, v, "fwd"), 0))
        uv_bwd = int(votes.get((u, v, "bwd"), 0))
        vu_fwd = int(votes.get((v, u, "fwd"), 0))
        vu_bwd = int(votes.get((v, u, "bwd"), 0))
        if uv_fwd > uv_bwd:
            raw_directed.append((u, v))
        elif uv_bwd > uv_fwd:
            raw_directed.append((v, u))
        else:
            raw_undirected.append(tuple(sorted((u, v), key=str)))

        direction = None
        if uv_fwd > uv_bwd and vu_fwd < vu_bwd:
            direction = (u, v)
        elif vu_fwd > vu_bwd and uv_fwd < uv_bwd:
            direction = (v, u)
        if direction is None:
            cross_undirected.append(tuple(sorted((u, v), key=str)))
            continue
        cross_directed.append(direction)

        margin = abs(uv_fwd - uv_bwd) + abs(vu_fwd - vu_bwd)
        ci_pass = True
        ci_available = bool(raw_records)
        modal_answers = []
        if raw_records:
            for source, target, expected in (
                (u, v, "fwd" if direction == (u, v) else "bwd"),
                (v, u, "bwd" if direction == (u, v) else "fwd"),
            ):
                rows = grouped.get((source, target), [])
                supporting = [r for r in rows if r.get("direction") == expected]
                answer_counts = Counter(r.get("answer") for r in supporting)
                modal = answer_counts.most_common(1)[0][0] if answer_counts else None
                modal_answers.append(modal)
                if modal in {"D", "E"}:
                    ci_pass = False
        candidates.append(
            {
                "edge": direction,
                "margin": margin,
                "ci_pass": ci_pass,
                "ci_filter_available": ci_available,
                "modal_answers": modal_answers,
            }
        )

    # Add strongest seeds first and reject cycle-creating candidates.
    filtered: list[tuple] = []
    rejected_ci, rejected_cycle = [], []
    graph = nx.DiGraph()
    graph.add_nodes_from(skeleton.nodes())
    for candidate in sorted(candidates, key=lambda x: (-x["margin"], str(x["edge"]))):
        edge = tuple(candidate["edge"])
        if not candidate["ci_pass"]:
            rejected_ci.append(edge)
            continue
        graph.add_edge(*edge)
        if not nx.is_directed_acyclic_graph(graph):
            graph.remove_edge(*edge)
            rejected_cycle.append(edge)
        else:
            filtered.append(edge)
    filtered_pairs = {frozenset(e) for e in filtered}
    filtered_undirected = [
        tuple(sorted((u, v), key=str))
        for u, v in skeleton.edges()
        if frozenset((u, v)) not in filtered_pairs
    ]

    return {
        "raw_directed": raw_directed,
        "raw_undirected": raw_undirected,
        "cross_directed": cross_directed,
        "cross_undirected": cross_undirected,
        "filtered_directed": filtered,
        "filtered_undirected": filtered_undirected,
        "rejected_ci": rejected_ci,
        "rejected_cycle": rejected_cycle,
        "ci_filter_available": bool(raw_records),
        "candidates": candidates,
    }
