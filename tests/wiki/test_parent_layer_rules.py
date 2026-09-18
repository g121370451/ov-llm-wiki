from openviking.wiki.pipeline import _assign_parent_node_links, _reject_nodes_with_insufficient_refs
from openviking.wiki.schemas import (
    SourceAssignmentResult,
    SourceRef,
    WikiNode,
)


def test_parent_layer_keeps_parent_with_enough_child_nodes():
    parent = _node("parent", depth=3)
    result = _assignment_result("parent", ["child_a", "child_b", "child_c"])

    layer_nodes, active_nodes, filtered_result = _reject_nodes_with_insufficient_refs(
        [parent],
        [parent],
        result,
        min_sources=3,
        depth=3,
    )

    assert active_nodes[0].status == "active"
    assert layer_nodes[0].child_node_ids == ["child_a", "child_b", "child_c"]
    assert [ref.doc_id for ref in filtered_result.source_refs_by_node["parent"]] == [
        "child_a",
        "child_b",
        "child_c",
    ]


def test_parent_layer_rejects_parent_with_too_few_child_nodes():
    parent = _node("parent", depth=3)
    result = _assignment_result("parent", ["child_a", "child_b"])

    layer_nodes, active_nodes, _ = _reject_nodes_with_insufficient_refs(
        [parent],
        [parent],
        result,
        min_sources=3,
        depth=3,
    )

    assert active_nodes == []
    assert layer_nodes[0].status == "rejected"


def test_parent_layer_rejects_duplicate_parent_with_same_child_set():
    parents = [
        _node("parent_a", depth=3),
        _node("parent_b", depth=3),
    ]
    result = SourceAssignmentResult(
        source_refs_by_node={
            "parent_a": [
                _source_ref("child_a"),
                _source_ref("child_b"),
                _source_ref("child_c"),
            ],
            "parent_b": [
                _source_ref("child_c"),
                _source_ref("child_b"),
                _source_ref("child_a"),
            ],
        }
    )

    layer_nodes, active_nodes, filtered_result = _reject_nodes_with_insufficient_refs(
        parents,
        parents,
        result,
        min_sources=3,
        depth=3,
    )

    assert [node.node_id for node in active_nodes] == ["parent_a"]
    assert layer_nodes[0].status == "active"
    assert layer_nodes[1].status == "rejected"
    assert set(filtered_result.source_refs_by_node) == {"parent_a"}


def test_parent_layer_keeps_partially_overlapping_parent_nodes():
    parents = [
        _node("parent_a", depth=3),
        _node("parent_b", depth=3),
    ]
    result = SourceAssignmentResult(
        source_refs_by_node={
            "parent_a": [
                _source_ref("child_a"),
                _source_ref("child_b"),
                _source_ref("child_c"),
            ],
            "parent_b": [
                _source_ref("child_b"),
                _source_ref("child_c"),
                _source_ref("child_d"),
            ],
        }
    )

    layer_nodes, active_nodes, filtered_result = _reject_nodes_with_insufficient_refs(
        parents,
        parents,
        result,
        min_sources=3,
        depth=3,
    )

    assert [node.node_id for node in active_nodes] == ["parent_a", "parent_b"]
    assert [node.status for node in layer_nodes] == ["active", "active"]
    assert set(filtered_result.source_refs_by_node) == {"parent_a", "parent_b"}


def test_parent_layer_rejects_same_children_with_different_facet_support():
    parents = [
        _node("parent_a", depth=3),
        _node("parent_b", depth=3),
    ]
    result = SourceAssignmentResult(
        source_refs_by_node={
            "parent_a": [
                _source_ref("child_a", "facet_a"),
                _source_ref("child_b", "facet_a"),
                _source_ref("child_c", "facet_a"),
            ],
            "parent_b": [
                _source_ref("child_a", "facet_b"),
                _source_ref("child_b", "facet_b"),
                _source_ref("child_c", "facet_b"),
            ],
        }
    )

    layer_nodes, active_nodes, filtered_result = _reject_nodes_with_insufficient_refs(
        parents,
        parents,
        result,
        min_sources=3,
        depth=3,
    )

    assert [node.node_id for node in active_nodes] == ["parent_a"]
    assert [node.status for node in layer_nodes] == ["active", "rejected"]
    assert set(filtered_result.source_refs_by_node) == {"parent_a"}


def test_child_node_can_have_multiple_parent_nodes():
    child = _node("child_a", depth=2)
    parents = [
        _node("parent_a", depth=3).model_copy(update={"child_node_ids": ["child_a"]}),
        _node("parent_b", depth=3).model_copy(update={"child_node_ids": ["child_a"]}),
    ]

    linked_nodes = _assign_parent_node_links([child], parents)

    assert linked_nodes[0].parent_node_ids == ["parent_a", "parent_b"]


def _assignment_result(node_id: str, child_node_ids: list[str]) -> SourceAssignmentResult:
    refs = [_source_ref(child_node_id) for child_node_id in child_node_ids]
    return SourceAssignmentResult(
        source_refs_by_node={node_id: refs},
    )


def _node(node_id: str, depth: int) -> WikiNode:
    return WikiNode(
        node_id=node_id,
        title=node_id.replace("_", " ").title(),
        depth=depth,
        scope="Supported topic.",
    )


def _source_ref(doc_id: str, facet_id: str | None = None) -> SourceRef:
    return SourceRef(
        ref_id=doc_id,
        ref_type="wiki_node",
        doc_id=doc_id,
        resource_uri=f"viking://wiki/nodes/{doc_id}/",
        card_uri=f"viking://wiki/nodes/{doc_id}/card.json",
        title=doc_id,
        support_scope="Supports node.",
        matched_facet_ids=[facet_id] if facet_id else [],
    )
