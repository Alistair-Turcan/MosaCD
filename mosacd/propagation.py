from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping
import networkx as nx
from pgmpy.base import PDAG


def _canon(u, v):
    return tuple(sorted((u, v), key=lambda x: str(x)))


def graph_from_edge_sets(nodes, directed_edges, undirected_edges) -> PDAG:
    pdag = PDAG(
        directed_ebunch=set(directed_edges),
        undirected_ebunch={_canon(*e) for e in undirected_edges},
    )
    pdag.add_nodes_from(nodes)
    return pdag


def seed_edge_sets(skeleton: nx.Graph, directed_edges: Iterable[tuple]) -> tuple[list, set]:
    directed = list(dict.fromkeys(directed_edges))
    undirected = {_canon(u, v) for u, v in skeleton.edges()}
    for u, v in directed:
        undirected.discard(_canon(u, v))
    return directed, undirected


def evidence_for_edge(u, v, undir_graph: nx.Graph, separating_sets: Mapping):
    u_other_to_v = []
    v_other_to_u = []
    for pair, seps in separating_sets.items():
        if u in pair and all(v in s for s in seps):
            other = next(node for node in pair if node != u)
            if undir_graph.has_edge(v, other):
                u_other_to_v.append(other)
        elif v in pair and all(u in s for s in seps):
            other = next(node for node in pair if node != v)
            if undir_graph.has_edge(u, other):
                v_other_to_u.append(other)
    return u_other_to_v, v_other_to_u


def find_halfdir_triples(
    directed_edges: Iterable[tuple],
    undirected_edges: Iterable[tuple],
    separating_sets: Mapping,
    p_value_sets: Mapping,
) -> set[tuple]:
    undirected = {_canon(*e) for e in undirected_edges}
    dir_out = defaultdict(set)
    dir_any = set()
    for u, v in directed_edges:
        dir_out[u].add(v)
        dir_any.add(_canon(u, v))
    adj_any = dir_any | undirected
    new_dir: set[tuple] = set()

    # Non-collider evidence first.
    for pair in sorted(p_value_sets, key=lambda p: p_value_sets[p], reverse=True):
        x, y = sorted(pair, key=str)
        if _canon(x, y) in adj_any:
            continue
        for x_cur, y_cur in ((x, y), (y, x)):
            for z in list(dir_out.get(x_cur, ())):
                pair_yz = _canon(y_cur, z)
                if z in (x_cur, y_cur) or pair_yz not in undirected:
                    continue
                s_list = separating_sets[pair]
                if s_list and all(z in s for s in s_list):
                    if pair_yz not in dir_any:
                        new_dir.add((z, y_cur))
                        dir_out[z].add(y_cur)
                        dir_any.add(pair_yz)
                        undirected.remove(pair_yz)

    # Collider evidence second.
    for pair in sorted(p_value_sets, key=lambda p: p_value_sets[p], reverse=True):
        x, y = sorted(pair, key=str)
        if _canon(x, y) in adj_any:
            continue
        for x_cur, y_cur in ((x, y), (y, x)):
            for z in list(dir_out.get(x_cur, ())):
                pair_yz = _canon(y_cur, z)
                if z in (x_cur, y_cur) or pair_yz not in undirected:
                    continue
                s_list = separating_sets[pair]
                if s_list and all(z not in s for s in s_list):
                    if pair_yz not in dir_any:
                        new_dir.add((y_cur, z))
                        dir_out[y_cur].add(z)
                        dir_any.add(pair_yz)
                        undirected.remove(pair_yz)
    return new_dir


def find_undir_v_triples(
    undirected_edges: Iterable[tuple], separating_sets: Mapping, p_value_sets: Mapping
) -> set[tuple]:
    undirected = {_canon(*e) for e in undirected_edges}
    neighbors: dict = {}
    for u, v in undirected:
        neighbors.setdefault(u, set()).add(v)
        neighbors.setdefault(v, set()).add(u)
    new_directed: set[tuple] = set()
    for pair in sorted(p_value_sets, key=lambda p: p_value_sets[p], reverse=True):
        x, y = sorted(pair, key=str)
        for z in neighbors.get(x, set()) & neighbors.get(y, set()):
            if all(z not in s for s in separating_sets[pair]):
                xz, yz = _canon(x, z), _canon(y, z)
                if xz in undirected and yz in undirected:
                    undirected.remove(xz)
                    undirected.remove(yz)
                    new_directed.add((x, z))
                    new_directed.add((y, z))
    return new_directed


def _propagation_round(
    undir_graph: nx.Graph,
    directed_edges: list[tuple],
    undirected_edges: set[tuple],
    separating_sets: Mapping,
    p_value_sets: Mapping,
) -> tuple[int, int]:
    """One legacy SPR propagation round, excluding least-conflict orientation."""
    added = 0
    collider_count = 0

    # Length-2 directed-path acyclicity rule used in the submitted code.
    for z in undir_graph.nodes():
        xs = [u for u, v in directed_edges if v == z]
        ys = [v for u, v in directed_edges if u == z]
        for x in xs:
            for y in ys:
                pair = _canon(x, y)
                if pair in undirected_edges:
                    directed_edges.append((x, y))
                    undirected_edges.remove(pair)
                    added += 1

    for edge in find_halfdir_triples(
        directed_edges, undirected_edges, separating_sets, p_value_sets
    ):
        pair = _canon(*edge)
        if pair in undirected_edges:
            directed_edges.append(edge)
            undirected_edges.remove(pair)
            added += 1

    for edge in find_undir_v_triples(undirected_edges, separating_sets, p_value_sets):
        pair = _canon(*edge)
        if pair in undirected_edges:
            directed_edges.append(edge)
            undirected_edges.remove(pair)
            added += 1
            collider_count += 1

    return added, collider_count


def _least_conflict_pass(
    undir_graph: nx.Graph,
    directed_edges: list[tuple],
    undirected_edges: set[tuple],
    separating_sets: Mapping,
) -> int:
    added = 0
    for u, v in list(undirected_edges):
        pair = _canon(u, v)
        u_other_to_v, v_other_to_u = evidence_for_edge(
            u, v, undir_graph, separating_sets
        )
        uv_violated = sum((node, v) in directed_edges for node in u_other_to_v)
        vu_violated = sum((node, u) in directed_edges for node in v_other_to_u)
        if uv_violated < vu_violated:
            directed_edges.append((u, v))
            undirected_edges.remove(pair)
            added += 1
        elif uv_violated > vu_violated:
            directed_edges.append((v, u))
            undirected_edges.remove(pair)
            added += 1
    return added


def orient_edges_legacy(
    undir_graph: nx.Graph,
    directed_edges: Iterable[tuple],
    undirected_edges: Iterable[tuple],
    separating_sets: Mapping,
    p_value_sets: Mapping,
) -> tuple[PDAG, int]:
    """Exact refactoring of the existing interleaved SPR/least-conflict loop."""
    directed = list(dict.fromkeys(directed_edges))
    undirected = {_canon(*e) for e in undirected_edges}
    previous_count = -1
    collider_count = 0
    while len(undirected) != previous_count:
        previous_count = len(undirected)
        _, c = _propagation_round(
            undir_graph, directed, undirected, separating_sets, p_value_sets
        )
        collider_count += c
        _least_conflict_pass(undir_graph, directed, undirected, separating_sets)
    return graph_from_edge_sets(undir_graph.nodes(), directed, undirected), collider_count


def order_by_votes(pdag, vote: Mapping, *, cross_order: bool = True) -> nx.DiGraph:
    """Complete a PDAG using the vote-derived topological order."""
    weighted = nx.DiGraph()
    weighted.add_nodes_from(pdag.nodes())
    for u, v in pdag.to_undirected().edges():
        uv_fwd = vote.get((u, v, "fwd"), 0)
        uv_bwd = vote.get((u, v, "bwd"), 0)
        if cross_order:
            vu_fwd = vote.get((v, u, "fwd"), 0)
            vu_bwd = vote.get((v, u, "bwd"), 0)
            if uv_fwd >= uv_bwd and vu_fwd <= vu_bwd:
                net = (uv_fwd + vu_bwd) - (uv_bwd + vu_fwd)
                if net > 0:
                    weighted.add_edge(u, v, weight=net)
            elif uv_fwd <= uv_bwd and vu_fwd >= vu_bwd:
                net = (uv_bwd + vu_fwd) - (uv_fwd + vu_bwd)
                if net > 0:
                    weighted.add_edge(v, u, weight=net)
        else:
            if uv_fwd > uv_bwd:
                weighted.add_edge(u, v, weight=uv_fwd - uv_bwd)
            elif uv_bwd > uv_fwd:
                weighted.add_edge(v, u, weight=uv_bwd - uv_fwd)

    while not nx.is_directed_acyclic_graph(weighted):
        cycle = nx.find_cycle(weighted)
        weakest = min(cycle, key=lambda e: weighted.edges[e].get("weight", 0))
        weighted.remove_edge(*weakest)

    topo = list(nx.topological_sort(weighted))
    position = {node: i for i, node in enumerate(topo)}
    edges = set(pdag.edges())
    undirected = {_canon(u, v) for u, v in edges if (v, u) in edges}
    directed = {(u, v) for u, v in edges if (v, u) not in edges}
    for u, v in undirected:
        if position.get(u, 0) < position.get(v, 0):
            directed.add((u, v))
        else:
            directed.add((v, u))
    out = nx.DiGraph()
    out.add_nodes_from(pdag.nodes())
    out.add_edges_from(directed)
    return out
