"""Prompt builders for Wiki generation."""

from __future__ import annotations

import json
import re

from openviking.prompts.manager import PromptManager

from .schemas import (
    DocumentCard,
    DocumentFacetSet,
    GeneratedNodeContext,
    NodeDocument,
    ResourceDocument,
    WikiNode,
)

_PROMPT_MANAGER = PromptManager()
_OPAQUE_TITLE_PREFIX_RE = re.compile(
    r"^(?:dsid_[0-9a-f]{32}|s2_[0-9a-f]{40}|title_[0-9a-f]{20})(?:__)?",
    re.IGNORECASE,
)


def build_document_card_prompt(doc: ResourceDocument) -> str:
    payload = {"content_or_structure": doc.content_or_structure}
    return _render_wiki_prompt("wiki.document_card", payload)


def build_document_facets_prompt(
    doc: ResourceDocument,
    sections: list[dict[str, str]],
) -> str:
    payload = {"source_sections": sections}
    if title := _semantic_title(doc.title, source_id=doc.doc_id):
        payload["document_title"] = title
    return _render_wiki_prompt("wiki.document_facets", payload)


def build_facet_community_prompt(
    facet_sets: list[DocumentFacetSet],
    facet_ids: list[str],
) -> str:
    wanted = set(facet_ids)
    facet_texts = [
        facet.facet_text
        for facet_set in facet_sets
        for facet in facet_set.topic_facets
        if facet.facet_id in wanted
    ]
    payload = {"facet_texts": list(dict.fromkeys(facet_texts))}
    return _render_wiki_prompt("wiki.facet_community", payload)


def build_node_discovery_prompt(
    cards: list[DocumentCard],
    min_sources_per_node: int,
    source_ids: list[str] | None = None,
) -> str:
    if source_ids is not None and len(source_ids) != len(cards):
        raise ValueError("source_ids must have the same length as cards")
    inputs = {
        "source_records": [
            _source_card_payload(
                card,
                source_id=source_ids[index] if source_ids is not None else card.doc_id,
            )
            for index, card in enumerate(cards)
        ],
    }
    return _render_wiki_prompt(
        "wiki.node_discovery",
        inputs,
        min_sources_per_node=min_sources_per_node,
    )


def build_node_card_prompt(node: WikiNode, document: NodeDocument) -> str:
    inputs = {
        "node": node.model_dump(include={"title", "scope"}, mode="json"),
        "document_content": document.content,
    }
    return _render_wiki_prompt("wiki.node_card", inputs)


def build_node_documents_prompt(
    node: WikiNode,
    source_documents: list[dict],
    *,
    max_document_tokens: int = 16000,
) -> str:
    inputs = {
        "node": node.model_dump(include={"title", "scope"}, mode="json"),
        "source_documents": [
            _source_document_payload(source_document)
            for source_document in source_documents
        ],
    }
    return _render_wiki_prompt(
        "wiki.node_documents", inputs, max_document_tokens=max_document_tokens
    )


def build_node_document_outline_prompt(node: WikiNode, source_cards: list[DocumentCard]) -> str:
    inputs = {
        "node": node.model_dump(include={"title", "scope"}, mode="json"),
        "source_cards": [
            _source_card_semantic_payload(card)
            for card in source_cards
        ],
    }
    return _render_wiki_prompt("wiki.node_document_outline", inputs)


def build_node_document_refine_prompt(
    node: WikiNode,
    current_markdown: str,
    source_documents: list[dict],
    *,
    initial: bool,
    max_document_tokens: int = 16000,
) -> str:
    inputs = {
        "node": node.model_dump(include={"title", "scope"}, mode="json"),
        "current_markdown": current_markdown,
        "new_source_documents": [
            _source_document_payload(source_document)
            for source_document in source_documents
        ],
    }
    return _render_wiki_prompt(
        "wiki.node_document_refine",
        inputs,
        phase="initial" if initial else "refine",
        max_document_tokens=max_document_tokens,
    )


def build_next_layer_decision_prompt(
    child_nodes: list[GeneratedNodeContext],
    min_child_nodes_per_parent: int = 3,
) -> str:
    inputs = {
        "child_nodes": [_child_node_payload(context) for context in child_nodes],
    }
    return _render_wiki_prompt(
        "wiki.next_layer_decision",
        inputs,
        min_child_nodes_per_parent=min_child_nodes_per_parent,
    )


def _child_node_payload(context: GeneratedNodeContext) -> dict:
    return {
        "node": context.node.model_dump(include={"title", "scope"}, mode="json"),
        "card": context.card.model_dump(
            include={"summary", "important_terms", "candidate_topics"},
            mode="json",
        ),
    }


def _source_document_payload(
    source_document: dict,
) -> dict:
    sections = [
        {"content": str(section.get("content") or "")}
        for section in source_document.get("sections", [])
        if str(section.get("content") or "")
    ]
    payload = {"sections": sections}
    title = _semantic_title(
        str(source_document.get("title") or ""),
        source_id=str(source_document.get("source_id") or ""),
    )
    if title:
        payload["title"] = title
    return payload


def _source_card_payload(card: DocumentCard, *, source_id: str) -> dict:
    payload = {
        "source_id": source_id,
        "summary": card.summary,
        "candidate_topics": card.candidate_topics,
    }
    if title := _semantic_title(card.title, source_id=card.doc_id):
        payload["title"] = title
    return payload


def _source_card_semantic_payload(card: DocumentCard) -> dict:
    payload = {
        "summary": card.summary,
        "candidate_topics": card.candidate_topics,
    }
    if title := _semantic_title(card.title, source_id=card.doc_id):
        payload["title"] = title
    return payload


def _semantic_title(title: str, *, source_id: str) -> str:
    normalized = title.strip()
    if not normalized or normalized == source_id:
        return ""
    return _OPAQUE_TITLE_PREFIX_RE.sub("", normalized).lstrip("_- " )


def _render_wiki_prompt(prompt_id: str, payload: object, **extra_vars: object) -> str:
    variables = {
        "input_json": json.dumps(payload, ensure_ascii=False, indent=2),
        **extra_vars,
    }
    return _PROMPT_MANAGER.render(prompt_id, variables)
