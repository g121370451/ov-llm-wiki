"""Generate document facets from complete, section-batched source documents."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections import defaultdict
from typing import Any

from pydantic import ValidationError

from openviking.models.embedder.base import embed_compat

from .llm import WikiLLMRunner
from .prompts import build_document_facets_prompt
from .schemas import (
    DocumentFacetBatchResponse,
    DocumentFacetSet,
    FacetRelation,
    ResourceDocument,
    TopicFacet,
)

_SPACE_RE = re.compile(r"\s+")


class DocumentFacetGenerator:
    """Extract facets without truncating away tail sections of long documents."""

    def __init__(
        self,
        llm: WikiLLMRunner,
        *,
        max_batch_chars: int = 30000,
        max_concurrent: int = 10,
        embedder: Any | None = None,
        dedup_score_threshold: float = 0.92,
    ):
        self.llm = llm
        self.max_batch_chars = max(1, max_batch_chars)
        self.max_concurrent = max(1, max_concurrent)
        self.embedder = embedder
        self.dedup_score_threshold = dedup_score_threshold

    async def generate(self, docs: list[ResourceDocument]) -> list[DocumentFacetSet]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results: list[DocumentFacetSet | None] = [None] * len(docs)

        async def generate_one(index: int, doc: ResourceDocument) -> None:
            async with semaphore:
                results[index] = await self.generate_one(doc)

        await asyncio.gather(*(generate_one(index, doc) for index, doc in enumerate(docs)))
        if any(result is None for result in results):
            raise RuntimeError("document facet generation did not produce all facet sets")
        return [result for result in results if result is not None]

    async def generate_one(self, doc: ResourceDocument) -> DocumentFacetSet:
        batches = _batch_source_sections(doc, max_batch_chars=self.max_batch_chars)
        responses: list[tuple[list[TopicFacet], list[FacetRelation]]] = []
        for batch_index, sections in enumerate(batches, start=1):
            prompt_sections, evidence_ref_map = _alias_source_refs(sections)
            prompt = build_document_facets_prompt(doc, prompt_sections)
            result = await self._complete_batch(
                prompt,
                step=f"document_facets_batch_{batch_index}",
                sections=prompt_sections,
            )
            responses.append(_restore_source_refs(result, evidence_ref_map))
        evidence_by_ref = {
            section.section_uri: section.content for section in doc.source_sections
        }
        if not evidence_by_ref and doc.content_or_structure.strip():
            evidence_by_ref[doc.resource_uri] = doc.content_or_structure
        return await self._merge_batches(doc.doc_id, responses, evidence_by_ref)

    async def _complete_batch(
        self, prompt: str, *, step: str, sections: list[dict[str, str]]
    ) -> DocumentFacetBatchResponse:
        last_error: Exception | None = None
        attempt_prompt = prompt
        allowed_refs = sorted(section["section_uri"] for section in sections)
        for attempt in range(1, 4):
            try:
                result = await self.llm.complete_json(
                    step=step if attempt == 1 else f"{step}_retry",
                    prompt=attempt_prompt,
                    schema=DocumentFacetBatchResponse.model_json_schema(),
                    prompt_version="document_facets_v2",
                )
                response = DocumentFacetBatchResponse.model_validate(result)
                _validate_batch_response(response, sections)
                return response
            except (RuntimeError, ValidationError) as exc:
                last_error = exc
                attempt_prompt = (
                    f"{prompt}\n\n"
                    "Your previous response failed validation. Return a corrected JSON object.\n"
                    f"Allowed source_refs for this batch: {allowed_refs}\n"
                    "Do not return any source_ref outside this exact list."
                )
        assert last_error is not None
        raise last_error

    async def _merge_batches(
        self,
        doc_id: str,
        responses: list[tuple[list[TopicFacet], list[FacetRelation]]],
        evidence_by_ref: dict[str, str],
    ) -> DocumentFacetSet:
        raw_facets: list[TopicFacet] = []
        raw_relations: list[FacetRelation] = []
        for batch_index, (topic_facets, facet_relations) in enumerate(responses, start=1):
            local_to_global: dict[str, str] = {}
            for local_index, facet in enumerate(topic_facets, start=1):
                temporary_id = f"batch_{batch_index}:{local_index}"
                local_to_global[facet.facet_id] = temporary_id
                raw_facets.append(facet.model_copy(update={"facet_id": temporary_id}))
            for relation in facet_relations:
                raw_relations.append(
                    relation.model_copy(
                        update={
                            "source_facet_id": local_to_global[relation.source_facet_id],
                            "target_facet_id": local_to_global[relation.target_facet_id],
                        }
                    )
                )

        clusters = await _deduplicate_facets(
            raw_facets,
            embedder=self.embedder,
            threshold=self.dedup_score_threshold,
        )
        merged_facets: list[TopicFacet] = []
        old_to_new: dict[str, str] = {}
        for cluster in clusters:
            representative = max(cluster, key=lambda item: (len(item.facet_text), item.facet_text))
            source_refs = list(
                dict.fromkeys(ref for facet in cluster for ref in facet.source_refs)
            )
            facet_id = stable_facet_id(
                doc_id,
                representative.facet_text,
                source_refs,
                evidence_by_ref=evidence_by_ref,
            )
            merged_facets.append(
                TopicFacet(
                    facet_id=facet_id,
                    facet_text=representative.facet_text,
                    source_refs=source_refs,
                )
            )
            for facet in cluster:
                old_to_new[facet.facet_id] = facet_id

        merged_relations: dict[tuple[str, str, str], FacetRelation] = {}
        for relation in raw_relations:
            source_id = old_to_new[relation.source_facet_id]
            target_id = old_to_new[relation.target_facet_id]
            if source_id == target_id:
                continue
            relation_text = _normalize_text(relation.relation_text)
            key = (source_id, target_id, relation_text.casefold())
            previous = merged_relations.get(key)
            refs = list(
                dict.fromkeys(
                    [*(previous.source_refs if previous else []), *relation.source_refs]
                )
            )
            merged_relations[key] = FacetRelation(
                source_facet_id=source_id,
                target_facet_id=target_id,
                relation_text=relation_text,
                source_refs=refs,
            )
        return DocumentFacetSet(
            doc_id=doc_id,
            topic_facets=merged_facets,
            facet_relations=list(merged_relations.values()),
        )


def legacy_facet_set_from_card(card: Any) -> DocumentFacetSet:
    """Adapt an old Document Card into evidence-poor facets for migration."""
    facets = [
        TopicFacet(
            facet_id=stable_facet_id(card.doc_id, topic, []),
            facet_text=topic,
            source_refs=[],
        )
        for topic in dict.fromkeys(card.candidate_topics)
        if str(topic).strip()
    ]
    return DocumentFacetSet(doc_id=card.doc_id, topic_facets=facets)


def stable_facet_id(
    doc_id: str,
    facet_text: str,
    source_refs: list[str],
    *,
    evidence_by_ref: dict[str, str] | None = None,
) -> str:
    normalized = _normalize_text(facet_text).casefold()
    evidence_by_ref = evidence_by_ref or {}
    evidence = "\n".join(
        hashlib.sha256(evidence_by_ref.get(ref, "").encode("utf-8")).hexdigest()
        for ref in sorted(dict.fromkeys(source_refs))
    )
    digest = hashlib.sha256(f"{normalized}\n{evidence}".encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}:{digest}"


def _batch_source_sections(
    doc: ResourceDocument, *, max_batch_chars: int
) -> list[list[dict[str, str]]]:
    sections = [
        {"section_uri": section.section_uri, "content": section.content}
        for section in doc.source_sections
        if section.content.strip()
    ]
    if not sections and doc.content_or_structure.strip():
        sections = [
            {
                "section_uri": doc.resource_uri,
                "content": doc.content_or_structure,
            }
        ]
    batches: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_chars = 0
    for section in sections:
        uri = section["section_uri"]
        content = section["content"]
        pieces = [
            content[start : start + max_batch_chars]
            for start in range(0, len(content), max_batch_chars)
        ] or [content]
        for piece_index, piece in enumerate(pieces, start=1):
            piece_uri = uri
            if current and current_chars + len(piece) > max_batch_chars:
                batches.append(current)
                current = []
                current_chars = 0
            current.append({"section_uri": piece_uri, "content": piece})
            current_chars += len(piece)
    if current:
        batches.append(current)
    return batches or [[]]


def _validate_batch_response(
    response: DocumentFacetBatchResponse, sections: list[dict[str, str]]
) -> None:
    known_refs = {section["section_uri"] for section in sections}
    facet_ids = [facet.facet_id for facet in response.topic_facets]
    if len(set(facet_ids)) != len(facet_ids):
        raise RuntimeError("facet extraction returned duplicate facet_id values")
    missing_evidence = [
        facet.facet_id for facet in response.topic_facets if not facet.source_refs
    ]
    missing_evidence.extend(
        f"{relation.source_facet_id}->{relation.target_facet_id}"
        for relation in response.facet_relations
        if not relation.source_refs
    )
    if missing_evidence:
        raise RuntimeError(
            f"facet extraction returned items without source refs: {missing_evidence}"
        )
    unknown_refs = {
        ref
        for facet in response.topic_facets
        for ref in facet.source_refs
        if ref not in known_refs
    }
    unknown_refs.update(
        ref
        for relation in response.facet_relations
        for ref in relation.source_refs
        if ref not in known_refs
    )
    if unknown_refs:
        raise RuntimeError(f"facet extraction returned unknown source refs: {sorted(unknown_refs)}")
    known_ids = set(facet_ids)
    unknown_endpoints = {
        endpoint
        for relation in response.facet_relations
        for endpoint in (relation.source_facet_id, relation.target_facet_id)
        if endpoint not in known_ids
    }
    if unknown_endpoints:
        raise RuntimeError(
            f"facet extraction returned unknown relation endpoints: {sorted(unknown_endpoints)}"
        )


def _alias_source_refs(
    sections: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """Use the same S0001-style source aliases as the node discovery prompt."""
    aliased: list[dict[str, str]] = []
    ref_map: dict[str, str] = {}
    for index, section in enumerate(sections, start=1):
        alias = f"S{index:04d}"
        aliased.append({"section_uri": alias, "content": section["content"]})
        ref_map[alias] = section["section_uri"]
    return aliased, ref_map


def _restore_source_refs(
    response: DocumentFacetBatchResponse, ref_map: dict[str, str]
) -> tuple[list[TopicFacet], list[FacetRelation]]:
    return (
        [
            TopicFacet(
                facet_id=facet.facet_id,
                facet_text=facet.facet_text,
                source_refs=[ref_map[ref] for ref in facet.source_refs],
            )
            for facet in response.topic_facets
        ],
        [
            FacetRelation(
                source_facet_id=relation.source_facet_id,
                target_facet_id=relation.target_facet_id,
                relation_text=relation.relation_text,
                source_refs=[ref_map[ref] for ref in relation.source_refs],
            )
            for relation in response.facet_relations
        ],
    )


async def _deduplicate_facets(
    facets: list[TopicFacet], *, embedder: Any | None, threshold: float
) -> list[list[TopicFacet]]:
    if len(facets) < 2:
        return [[facet] for facet in facets]
    parent = list(range(len(facets)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    normalized = [_normalize_text(facet.facet_text).casefold() for facet in facets]
    for left in range(len(facets)):
        for right in range(left + 1, len(facets)):
            if normalized[left] == normalized[right]:
                union(left, right)

    unresolved = [index for index in range(len(facets)) if find(index) == index]
    if embedder is not None and len(unresolved) > 1:
        vectors = await asyncio.gather(
            *(embed_compat(embedder, facets[index].facet_text) for index in unresolved)
        )
        dense = [
            _require_dense_vector(result, facets[index].facet_id)
            for result, index in zip(vectors, unresolved)
        ]
        for left_position, left in enumerate(unresolved):
            for right_position in range(left_position + 1, len(unresolved)):
                right = unresolved[right_position]
                if _cosine(dense[left_position], dense[right_position]) >= threshold:
                    union(left, right)

    grouped: dict[int, list[TopicFacet]] = defaultdict(list)
    for index, facet in enumerate(facets):
        grouped[find(index)].append(facet)
    return list(grouped.values())


def _require_dense_vector(result: Any, facet_id: str) -> list[float]:
    vector = getattr(result, "dense_vector", None)
    if not vector:
        raise RuntimeError(f"embedding for facet {facet_id} has no dense vector")
    return [float(value) for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError("facet embedding dimensions do not match")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _normalize_text(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip()
