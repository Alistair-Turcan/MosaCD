from __future__ import annotations

from typing import Any
import networkx as nx
from pgmpy.estimators import ExpertKnowledge, PC
from pgmpy.estimators.CITests import (
    chi_square, g_sq, gcm, log_likelihood, modified_log_likelihood,
    pearsonr, pillai_trace,
)


SUPPORTED_TESTS = {
    "chi_square": chi_square,
    "g_sq": g_sq,
    "log_likelihood": log_likelihood,
    "modified_log_likelihood": modified_log_likelihood,
    "pearsonr": pearsonr,
    "pillai": pillai_trace,
    "gcm": gcm,
}


def build_skeleton_ci(
    *,
    data,
    variant: str = "orig",
    ci_test: str = "chi_square",
    significance_level: float = 0.01,
    max_cond_vars: int = 3,
    expert_knowledge: ExpertKnowledge | None = None,
    enforce_expert_knowledge: bool = False,
    **kwargs: Any,
) -> tuple[nx.Graph, dict, dict]:
    """Project skeleton search used in the submitted experiments.

    ``variant`` is one of ``orig`` (PC), ``stable`` (PC-stable skeleton), or
    ``conservative`` (stable skeleton while retaining all minimum-cardinality
    separating sets at the first accepted level).
    """
    if data is None:
        raise ValueError("data must be specified")
    if ci_test not in SUPPORTED_TESTS:
        raise ValueError(f"Unsupported CI test: {ci_test}")
    if variant not in {"orig", "stable", "conservative"}:
        raise ValueError(f"Unknown variant: {variant}")

    test_fn = SUPPORTED_TESTS[ci_test]
    separating_sets: dict = {}
    p_value_sets: dict = {}

    expert_knowledge = expert_knowledge or ExpertKnowledge()
    if getattr(expert_knowledge, "search_space", None):
        expert_knowledge.limit_search_space(data.columns)

    variables = list(data.columns.values)
    graph = nx.complete_graph(n=variables, create_using=nx.Graph)
    temporal_ordering = expert_knowledge.temporal_ordering
    if enforce_expert_knowledge:
        graph.remove_edges_from(expert_knowledge.forbidden_edges)

    lim_neighbors = 0
    while not all(len(list(graph.neighbors(var))) < lim_neighbors for var in variables):
        if variant == "orig":
            for u, v in list(graph.edges()):
                if enforce_expert_knowledge and (u, v) in expert_knowledge.required_edges:
                    continue
                for separating_set in PC._get_potential_sepsets(
                    u, v, temporal_ordering, graph, lim_neighbors
                ):
                    p_value = test_fn(
                        u,
                        v,
                        separating_set,
                        data=data,
                        independencies=None,
                        significance_level=significance_level,
                        boolean=False,
                        **kwargs,
                    )[1]
                    if p_value >= significance_level:
                        separating_sets[frozenset((u, v))] = [separating_set]
                        p_value_sets[frozenset((u, v))] = p_value
                        if graph.has_edge(u, v):
                            graph.remove_edge(u, v)
                        break
        else:
            all_sepsets_at_k: dict[tuple, list] = {}
            for u, v in list(graph.edges()):
                if enforce_expert_knowledge and (u, v) in expert_knowledge.required_edges:
                    continue
                seps_found = []
                pvals_found = []
                for separating_set in PC._get_potential_sepsets(
                    u, v, temporal_ordering, graph, lim_neighbors
                ):
                    p_value = test_fn(
                        u,
                        v,
                        separating_set,
                        data=data,
                        independencies=None,
                        significance_level=significance_level,
                        boolean=False,
                        **kwargs,
                    )[1]
                    if p_value >= significance_level:
                        seps_found.append(separating_set)
                        pvals_found.append(p_value)
                if not seps_found:
                    continue
                if variant == "stable":
                    best_idx = max(range(len(pvals_found)), key=pvals_found.__getitem__)
                    all_sepsets_at_k[(u, v)] = [seps_found[best_idx]]
                    p_value_sets[frozenset((u, v))] = pvals_found[best_idx]
                else:
                    min_len = min(len(s) for s in seps_found)
                    candidates = [
                        (s, p)
                        for s, p in zip(seps_found, pvals_found)
                        if len(s) == min_len
                    ]
                    candidates.sort(key=lambda sp: sp[1], reverse=True)
                    all_sepsets_at_k[(u, v)] = [s for s, _ in candidates]
                    p_value_sets[frozenset((u, v))] = candidates[0][1]

            graph.remove_edges_from(all_sepsets_at_k.keys())
            for (u, v), seps in all_sepsets_at_k.items():
                key = frozenset((u, v))
                if key in separating_sets:
                    min_prev = min(len(s) for s in separating_sets[key])
                    seps = [s for s in seps if len(s) <= min_prev]
                separating_sets[key] = seps

        if lim_neighbors >= max_cond_vars:
            break
        lim_neighbors += 1

    return graph, separating_sets, p_value_sets
