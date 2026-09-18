"""Facet-based node discovery with a sparse similarity graph and CPM communities."""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from openviking.models.embedder.base import embed_compat

from .facets import _cosine, _require_dense_vector
from .llm import WikiLLMRunner
from .nodes import NodeDiscoveryResult, _complete_with_validation_retry
from .prompts import build_facet_community_prompt
from .schemas import (
    DocumentFacetSet,
    SourceAssignmentItem,
    SourceAssignmentResponse,
    SourceFacetMatch,
    TopicFacet,
    WikiNode,
    WikiNodeDiscoveryItem,
)
from .uri import sanitize_node_id


@dataclass(frozen=True)
class FacetGraphEdge:
    left_facet_id: str
    right_facet_id: str
    score: float


class ScalableNodeDiscoveryRunner:
    """Discover one Wiki layer from facet representations."""

    def __init__(
        self,
        llm: WikiLLMRunner,
        embedder: Any,
        *,
        neighbor_limit: int = 20,
        edge_score_threshold: float = 0.72,
        cpm_resolution: float = 0.7,
        leiden_seed: int = 0,
    ):
        self.llm = llm
        self.embedder = embedder
        self.neighbor_limit = max(1, neighbor_limit)
        self.edge_score_threshold = edge_score_threshold
        self.cpm_resolution = cpm_resolution
        self.leiden_seed = leiden_seed
        self.neighbor_search_backend = "exact"
        self.last_run_artifact: dict[str, Any] = {}
        self.last_run_edges: list[dict[str, Any]] = []

    async def discover_layer(
        self,
        *,
        facet_sets_by_source_id: dict[str, DocumentFacetSet],
        depth: int,
        min_sources_per_node: int,
        reserved_node_ids: set[str] | None = None,
    ) -> NodeDiscoveryResult:
        completed_sets = dict(facet_sets_by_source_id)
        source_ids = set(completed_sets)
        facets = [
            (facet_set.doc_id, facet)
            for source_id in sorted(source_ids)
            if (facet_set := completed_sets.get(source_id)) is not None
            for facet in facet_set.topic_facets
        ]
        if not facets:
            self.last_run_artifact = {
                "depth": depth,
                "neighbor_search_backend": "none",
                "facet_count": 0,
                "source_count": len(source_ids),
                "edge_count": 0,
                "community_count": 0,
                "mature_community_count": 0,
                "neighbor_limit": self.neighbor_limit,
                "edge_score_threshold": self.edge_score_threshold,
                "cpm_resolution": self.cpm_resolution,
                "communities": [],
            }
            self.last_run_edges = []
            return NodeDiscoveryResult(
                nodes=[],
                source_assignments=SourceAssignmentResponse(
                    assignments=[], unassigned_source_ids=sorted(source_ids)
                ),
            )

        vectors = await asyncio.gather(
            *(embed_compat(self.embedder, facet.facet_text) for _, facet in facets)
        )
        dense_vectors = [
            _require_dense_vector(result, facet.facet_id)
            for result, (_, facet) in zip(vectors, facets)
        ]
        edges, self.neighbor_search_backend = _build_similarity_graph(
            facets,
            dense_vectors,
            neighbor_limit=self.neighbor_limit,
            edge_score_threshold=self.edge_score_threshold,
            seed=self.leiden_seed,
        )
        communities = discover_cpm_communities(
            [facet.facet_id for _, facet in facets],
            edges,
            resolution=self.cpm_resolution,
            seed=self.leiden_seed,
        )
        facet_by_id = {facet.facet_id: (doc_id, facet) for doc_id, facet in facets}
        raw_mature = [
            community
            for community in communities
            if len({facet_by_id[facet_id][0] for facet_id in community})
            >= min_sources_per_node
        ]
        mature = (
            _merge_communities_with_same_sources(raw_mature, facet_by_id)
            if depth > 1
            else raw_mature
        )
        mature_ids = {tuple(community) for community in raw_mature}
        parent_layer_stop_reason = (
            "no_strict_node_count_reduction"
            if depth > 1 and len(mature) >= len(source_ids)
            else None
        )
        self.last_run_artifact = {
            "depth": depth,
            "neighbor_search_backend": self.neighbor_search_backend,
            "facet_count": len(facets),
            "source_count": len(source_ids),
            "edge_count": len(edges),
            "community_count": len(communities),
            "mature_community_count": len(mature),
            "raw_mature_community_count": len(raw_mature),
            "parent_layer_stop_reason": parent_layer_stop_reason,
            "neighbor_limit": self.neighbor_limit,
            "edge_score_threshold": self.edge_score_threshold,
            "cpm_resolution": self.cpm_resolution,
            "communities": [
                {
                    "facet_ids": community,
                    "source_ids": sorted(
                        {facet_by_id[facet_id][0] for facet_id in community}
                    ),
                    "mature": tuple(community) in mature_ids,
                }
                for community in communities
            ],
        }
        self.last_run_edges = [
            {
                "left_facet_id": edge.left_facet_id,
                "right_facet_id": edge.right_facet_id,
                "score": edge.score,
            }
            for edge in edges
        ]
        if parent_layer_stop_reason is not None:
            return NodeDiscoveryResult(
                nodes=[],
                source_assignments=SourceAssignmentResponse(
                    assignments=[], unassigned_source_ids=sorted(source_ids)
                ),
            )

        discovered = []
        for community in mature:
            prompt = build_facet_community_prompt(
                list(completed_sets.values()), community
            )
            item = await _complete_with_validation_retry(
                self.llm,
                step="facet_community",
                prompt=prompt,
                schema=WikiNodeDiscoveryItem.model_json_schema(),
                validate=WikiNodeDiscoveryItem.model_validate,
            )
            discovered.append((community, item))

        nodes = _build_nodes(
            [item for _, item in discovered],
            depth=depth,
            reserved_node_ids=reserved_node_ids or set(),
        )
        assignments: list[SourceAssignmentItem] = []
        assigned_source_ids: set[str] = set()
        for node, (community, item) in zip(nodes, discovered):
            matches: dict[str, list[SourceFacetMatch]] = defaultdict(list)
            for facet_id in community:
                source_id, facet = facet_by_id[facet_id]
                matches[source_id].append(
                    SourceFacetMatch(
                        facet_id=facet.facet_id,
                        facet_text=facet.facet_text,
                        source_refs=facet.source_refs,
                    )
                )
            assigned_source_ids.update(matches)
            assignments.append(
                SourceAssignmentItem(
                    node_id=node.node_id,
                    source_ids=sorted(matches),
                    support_scope=item.scope,
                    facet_matches_by_source_id=dict(matches),
                )
            )
        return NodeDiscoveryResult(
            nodes=nodes,
            source_assignments=SourceAssignmentResponse(
                assignments=assignments,
                unassigned_source_ids=sorted(source_ids - assigned_source_ids),
            ),
        )


def _merge_communities_with_same_sources(
    communities: list[list[str]],
    facet_by_id: dict[str, tuple[str, TopicFacet]],
) -> list[list[str]]:
    """Collapse parent candidates backed by exactly the same child nodes.

    A child may support several parents, but two parents with the same complete
    child set are one structural parent even when Leiden split their facets into
    different topical communities. Keep all supporting facets on that parent.
    """
    merged_by_sources: dict[tuple[str, ...], set[str]] = {}
    order: list[tuple[str, ...]] = []
    for community in communities:
        source_signature = tuple(
            sorted({facet_by_id[facet_id][0] for facet_id in community})
        )
        if source_signature not in merged_by_sources:
            merged_by_sources[source_signature] = set()
            order.append(source_signature)
        merged_by_sources[source_signature].update(community)
    return [sorted(merged_by_sources[signature]) for signature in order]


def build_sparse_similarity_graph(
    facets: list[tuple[str, TopicFacet]],
    vectors: list[list[float]],
    *,
    neighbor_limit: int,
    edge_score_threshold: float,
) -> list[FacetGraphEdge]:
    """Build an exact top-k graph while excluding same-document edges."""
    return _build_exact_similarity_graph(
        facets,
        vectors,
        neighbor_limit=neighbor_limit,
        edge_score_threshold=edge_score_threshold,
    )


def _build_similarity_graph(
    facets: list[tuple[str, TopicFacet]],
    vectors: list[list[float]],
    *,
    neighbor_limit: int,
    edge_score_threshold: float,
    seed: int,
) -> tuple[list[FacetGraphEdge], str]:
    if len(facets) >= 256:
        try:
            return (
                _build_hnsw_similarity_graph(
                    facets,
                    vectors,
                    neighbor_limit=neighbor_limit,
                    edge_score_threshold=edge_score_threshold,
                    seed=seed,
                ),
                "hnsw",
            )
        except ImportError:
            pass
    return (
        _build_exact_similarity_graph(
            facets,
            vectors,
            neighbor_limit=neighbor_limit,
            edge_score_threshold=edge_score_threshold,
        ),
        "exact",
    )


def _build_exact_similarity_graph(
    facets: list[tuple[str, TopicFacet]],
    vectors: list[list[float]],
    *,
    neighbor_limit: int,
    edge_score_threshold: float,
) -> list[FacetGraphEdge]:
    if len(facets) != len(vectors):
        raise ValueError("facets and vectors must have the same length")
    pair_scores: dict[tuple[int, int], float] = {}
    for left, (left_doc_id, _) in enumerate(facets):
        candidates: list[tuple[float, int]] = []
        for right, (right_doc_id, _) in enumerate(facets):
            if left == right or left_doc_id == right_doc_id:
                continue
            score = _cosine(vectors[left], vectors[right])
            if score >= edge_score_threshold:
                candidates.append((score, right))
        candidates.sort(key=lambda item: (-item[0], facets[item[1]][1].facet_id))
        for score, right in candidates[:neighbor_limit]:
            pair = (min(left, right), max(left, right))
            pair_scores[pair] = max(score, pair_scores.get(pair, float("-inf")))
    return [
        FacetGraphEdge(
            left_facet_id=facets[left][1].facet_id,
            right_facet_id=facets[right][1].facet_id,
            score=score,
        )
        for (left, right), score in sorted(pair_scores.items())
    ]


def _build_hnsw_similarity_graph(
    facets: list[tuple[str, TopicFacet]],
    vectors: list[list[float]],
    *,
    neighbor_limit: int,
    edge_score_threshold: float,
    seed: int,
) -> list[FacetGraphEdge]:
    try:
        import hnswlib
        import numpy as np
    except ImportError as exc:
        raise ImportError("HNSW neighbor search requires hnswlib and numpy") from exc

    if len(facets) != len(vectors):
        raise ValueError("facets and vectors must have the same length")
    if not facets:
        return []
    dimension = len(vectors[0])
    if not dimension or any(len(vector) != dimension for vector in vectors):
        raise RuntimeError("facet embedding dimensions do not match")

    matrix = np.asarray(vectors, dtype=np.float32)
    index = hnswlib.Index(space="cosine", dim=dimension)
    index.init_index(
        max_elements=len(facets),
        ef_construction=max(100, neighbor_limit * 4),
        M=16,
        random_seed=seed,
    )
    index.add_items(matrix, np.arange(len(facets)), num_threads=1)
    max_same_document = max(Counter(doc_id for doc_id, _ in facets).values())
    query_limit = min(len(facets), neighbor_limit + max_same_document + 1)
    index.set_ef(max(64, query_limit))
    labels, distances = index.knn_query(matrix, k=query_limit, num_threads=1)

    pair_scores: dict[tuple[int, int], float] = {}
    for left, (neighbors, neighbor_distances) in enumerate(zip(labels, distances)):
        retained = 0
        for right_value, distance_value in zip(neighbors, neighbor_distances):
            right = int(right_value)
            if left == right or facets[left][0] == facets[right][0]:
                continue
            score = 1.0 - float(distance_value)
            if score < edge_score_threshold:
                continue
            pair = (min(left, right), max(left, right))
            pair_scores[pair] = max(score, pair_scores.get(pair, float("-inf")))
            retained += 1
            if retained >= neighbor_limit:
                break
    return [
        FacetGraphEdge(
            left_facet_id=facets[left][1].facet_id,
            right_facet_id=facets[right][1].facet_id,
            score=score,
        )
        for (left, right), score in sorted(pair_scores.items())
    ]


def discover_cpm_communities(
    facet_ids: list[str],
    edges: list[FacetGraphEdge],
    *,
    resolution: float,
    seed: int = 0,
) -> list[list[str]]:
    """Run Leiden/CPM for a facet similarity graph."""
    if not facet_ids:
        return []
    if not edges:
        return [[facet_id] for facet_id in sorted(facet_ids)]
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise RuntimeError(
            "facet_graph requires python-igraph and leidenalg; install the wiki extra"
        ) from exc

    index = {facet_id: position for position, facet_id in enumerate(facet_ids)}
    graph = ig.Graph(
        n=len(facet_ids),
        edges=[
            (index[edge.left_facet_id], index[edge.right_facet_id]) for edge in edges
        ],
        directed=False,
    )
    partition = leidenalg.find_partition(
        graph,
        leidenalg.CPMVertexPartition,
        weights=[edge.score for edge in edges],
        resolution_parameter=resolution,
        seed=seed,
    )
    return [sorted(facet_ids[index] for index in community) for community in partition]
def _build_nodes(
    discovered: list[WikiNodeDiscoveryItem],
    *,
    depth: int,
    reserved_node_ids: set[str],
) -> list[WikiNode]:
    used_ids = set(reserved_node_ids)
    nodes: list[WikiNode] = []
    for item in discovered:
        base_id = sanitize_node_id(item.title)
        node_id = base_id
        suffix = 2
        while node_id in used_ids:
            node_id = f"{base_id}_{suffix}"
            suffix += 1
        used_ids.add(node_id)
        nodes.append(
            WikiNode(
                node_id=node_id, title=item.title, depth=depth, scope=item.scope
            )
        )
    return nodes
