from __future__ import annotations

import time
from typing import Mapping

import networkx as nx
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .llm import TrackedChatClient, query_mosacd_votes, stage_seed_sets
from .propagation import order_by_votes, orient_edges_legacy, seed_edge_sets
from .skeleton import SUPPORTED_TESTS, build_skeleton_ci


def _graph_dict(graph):
    """Serialize reciprocal PDAG arcs as one undirected edge, not two arrows."""
    edges = set(graph.edges())
    directed = sorted((u, v) for u, v in edges if (v, u) not in edges)
    undirected = sorted({tuple(sorted((u, v))) for u, v in edges if (v, u) in edges})
    return {
        "nodes": list(graph.nodes()),
        "directed_edges": [list(edge) for edge in directed],
        "undirected_edges": [list(edge) for edge in undirected],
    }


def _validate(data, description, variables, ci_test, alpha, max_cond_vars, n_votes):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    if data.shape[0] < 3 or data.shape[1] < 2:
        raise ValueError("Provide at least three rows and two variables")
    if not data.columns.is_unique or any(
        not isinstance(c, str) or not c.strip() for c in data.columns
    ):
        raise ValueError("Column names must be unique, nonempty strings")
    if data.isna().any().any():
        raise ValueError("Missing values are not supported; preprocess them explicitly")
    for column in data:
        if data[column].nunique() < 2:
            raise ValueError(f"Constant column: {column}")
        if is_numeric_dtype(data[column]) and not np.isfinite(data[column].to_numpy()).all():
            raise ValueError(f"Non-finite values in column: {column}")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a nonempty string")
    if not isinstance(variables, Mapping) or set(variables) != set(data.columns):
        raise ValueError("variables must describe exactly the CSV columns")
    if any(not isinstance(v, str) or not v.strip() for v in variables.values()):
        raise ValueError("Each variable description must be a nonempty string")
    if ci_test not in SUPPORTED_TESTS:
        raise ValueError(f"Unsupported CI test: {ci_test}")
    if ci_test in {"pearsonr", "gcm", "pillai"} and not all(
        is_numeric_dtype(data[c]) for c in data
    ):
        raise ValueError(f"{ci_test} requires numeric data")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if isinstance(max_cond_vars, bool) or not isinstance(max_cond_vars, int) or max_cond_vars < 0:
        raise ValueError("max_cond_vars must be a nonnegative integer")
    if isinstance(n_votes, bool) or not isinstance(n_votes, int) or n_votes < 1:
        raise ValueError("n_votes must be a positive integer")


def discover(
    data: pd.DataFrame,
    *,
    description: str,
    variables: Mapping[str, str],
    ci_test: str = "chi_square",
    alpha: float = 0.01,
    max_cond_vars: int = 3,
    model: str = "gpt-4o-mini",
    temperature: float = 0.6,
    n_votes: int = 5,
    seed: int | None = 0,
    complete: bool = True,
    dry_run: bool = False,
    base_url: str | None = None,
    api_key_env: str | None = None,
) -> dict:
    """Infer a graph without benchmark datasets or ground-truth labels.

    ``n_votes`` requests are made in each prompt order for each skeleton edge.
    ``dry_run`` computes only the CI skeleton and makes no API requests.
    ``complete=False`` returns the partially directed graph before vote completion.
    """
    _validate(data, description, variables, ci_test, alpha, max_cond_vars, n_votes)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a nonempty string")
    if not np.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must lie between 0 and 2")
    client = TrackedChatClient(base_url=base_url, api_key_env=api_key_env)
    if not dry_run:
        # Fail on missing credentials before paying the skeleton-search cost.
        client._client(model)

    start = time.perf_counter()
    skeleton, separating_sets, p_values = build_skeleton_ci(
        data=data, variant="orig", ci_test=ci_test,
        significance_level=alpha, max_cond_vars=max_cond_vars,
    )
    skeleton_seconds = time.perf_counter() - start
    result = {
        "status": "dry_run" if dry_run else "completed",
        "config": {
            "ci_test": ci_test, "alpha": alpha, "variant": "orig",
            "max_cond_vars": max_cond_vars, "model": model,
            "temperature": temperature, "n_votes": n_votes, "seed": seed,
            "complete": complete,
        },
        "skeleton_edges": [list(e) for e in skeleton.edges()],
        "expected_api_requests": 2 * n_votes * skeleton.number_of_edges(),
        "timings_seconds": {"skeleton": skeleton_seconds},
    }
    if dry_run:
        # An nx.Graph has no arrows; use the same JSON schema as the full run.
        result["graph"] = {
            "nodes": list(skeleton.nodes()),
            "directed_edges": [],
            "undirected_edges": [list(e) for e in skeleton.edges()],
        }
        result["usage"] = {"live_requests": 0}
        return result

    start = time.perf_counter()
    votes, records = query_mosacd_votes(
        dataset="user_data",
        data_description=description,
        node_description=variables,
        skeleton=skeleton,
        separating_sets=separating_sets,
        pval_sets=p_values,
        client=client, model=model, temperature=temperature,
        n_votes=n_votes, seed=seed,
    )
    result["timings_seconds"]["llm"] = time.perf_counter() - start
    start = time.perf_counter()
    seeds = stage_seed_sets(skeleton, votes, records)
    directed, undirected = seed_edge_sets(skeleton, seeds["filtered_directed"])
    graph, _ = orient_edges_legacy(
        skeleton, directed, undirected, separating_sets, p_values,
    )
    if complete:
        graph = order_by_votes(graph, votes, cross_order=True)
    result["timings_seconds"]["orientation"] = time.perf_counter() - start
    result["graph"] = _graph_dict(graph)
    result["directed_part_is_acyclic"] = nx.is_directed_acyclic_graph(
        nx.DiGraph(result["graph"]["directed_edges"])
    )
    # Token usage only: do not misrepresent an unavailable price as zero cost.
    usage = client.summary()
    result["usage"] = {
        key: usage[key] for key in (
            "live_requests", "prompt_tokens", "cached_prompt_tokens",
            "completion_tokens", "total_tokens",
        )
    }
    return result
