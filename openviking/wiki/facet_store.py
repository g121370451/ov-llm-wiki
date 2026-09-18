"""Independent, versioned storage for document facets."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from openviking_cli.exceptions import FailedPreconditionError

from .config import WikiConfig
from .schemas import (
    DocumentFacetBatchResponse,
    DocumentFacetManifest,
    DocumentFacetManifestEntry,
    DocumentFacetSet,
    ResourceDocument,
)
from .uri import (
    facet_json_uri,
    facet_manifest_uri,
    facets_dir,
    node_facets_dir,
    node_facets_json_uri,
    node_facets_manifest_uri,
)
from .writer import WikiVikingFSWriter

FACET_MANIFEST_VERSION = 1
FACET_GENERATOR_VERSION = "document_facets_v2"
FACET_PROMPT_VERSION = "document_facets_v2"


class DocumentFacetStore:
    def __init__(
        self,
        *,
        viking_fs: Any,
        writer: WikiVikingFSWriter,
        config: WikiConfig,
        ctx: Any,
    ):
        self.viking_fs = viking_fs
        self.writer = writer
        self.config = config
        self.ctx = ctx

    async def replace(
        self,
        *,
        facet_sets: list[DocumentFacetSet],
        resource_documents: list[ResourceDocument],
        max_batch_chars: int,
        model_provenance: dict[str, Any] | None = None,
    ) -> DocumentFacetManifest:
        facet_uris = {
            facet_set.doc_id: facet_json_uri(self.config, facet_set.doc_id)
            for facet_set in facet_sets
        }
        manifest = self._build_manifest(
            facet_sets=facet_sets,
            resource_documents=resource_documents,
            max_batch_chars=max_batch_chars,
            model_provenance=model_provenance or {},
            facet_uris=facet_uris,
        )
        cache_root = facets_dir(self.config)
        if await self.viking_fs.exists(cache_root, ctx=self.ctx):
            await self.viking_fs.rm(cache_root, recursive=True, ctx=self.ctx)
        await self.writer.ensure_facet_dirs()
        for facet_set in facet_sets:
            await self.writer.write_metadata_json(
                facet_uris[facet_set.doc_id], facet_set
            )
        await self.writer.write_metadata_json(facet_manifest_uri(self.config), manifest)
        return manifest

    async def replace_node_facets(
        self,
        *,
        facet_sets: list[DocumentFacetSet],
        resource_documents: list[ResourceDocument],
        max_batch_chars: int,
        model_provenance: dict[str, Any] | None = None,
    ) -> DocumentFacetManifest:
        facet_uris = {
            facet_set.doc_id: node_facets_json_uri(self.config, facet_set.doc_id)
            for facet_set in facet_sets
        }
        manifest = self._build_manifest(
            facet_sets=facet_sets,
            resource_documents=resource_documents,
            max_batch_chars=max_batch_chars,
            model_provenance=model_provenance or {},
            facet_uris=facet_uris,
        )
        cache_root = node_facets_dir(self.config)
        if await self.viking_fs.exists(cache_root, ctx=self.ctx):
            await self.viking_fs.rm(cache_root, recursive=True, ctx=self.ctx)
        await self.writer.ensure_facet_dirs()
        for facet_set in facet_sets:
            await self.writer.write_metadata_json(
                facet_uris[facet_set.doc_id], facet_set
            )
        await self.writer.write_metadata_json(
            node_facets_manifest_uri(self.config), manifest
        )
        return manifest

    async def exists(self) -> bool:
        return await self.viking_fs.exists(facet_manifest_uri(self.config), ctx=self.ctx)

    async def load_validated(
        self, resource_documents: list[ResourceDocument]
    ) -> list[DocumentFacetSet]:
        uri = facet_manifest_uri(self.config)
        if not await self.viking_fs.exists(uri, ctx=self.ctx):
            self._fail([f"facet manifest is missing: {uri}"])
        try:
            raw = await self.viking_fs.read_file(uri, ctx=self.ctx)
            manifest = DocumentFacetManifest.model_validate_json(raw)
        except FailedPreconditionError:
            raise
        except Exception as exc:
            self._fail([f"facet manifest is invalid: {exc}"])

        reasons: list[str] = []
        if manifest.version != FACET_MANIFEST_VERSION:
            reasons.append(
                f"manifest version changed: cached={manifest.version} "
                f"current={FACET_MANIFEST_VERSION}"
            )
        if manifest.generator_version != FACET_GENERATOR_VERSION:
            reasons.append("facet generator version changed")
        if manifest.prompt_version != FACET_PROMPT_VERSION:
            reasons.append("facet prompt version changed")
        if manifest.schema_hash != _schema_hash():
            reasons.append("DocumentFacetSet schema changed")
        if manifest.max_batch_chars != self.config.limits.max_facet_batch_chars:
            reasons.append("facet extraction batch size changed")

        docs_by_id = _unique_by_doc_id(resource_documents, "resource documents", reasons)
        entries_by_id = _unique_by_doc_id(manifest.entries, "facet manifest entries", reasons)
        if set(docs_by_id) != set(entries_by_id):
            reasons.append("facet cache document IDs do not match source documents")

        facet_sets: list[DocumentFacetSet] = []
        for doc in resource_documents:
            entry = entries_by_id.get(doc.doc_id)
            if entry is None:
                continue
            if entry.source_hash != resource_document_hash(doc):
                reasons.append(f"facet source input changed: {doc.doc_id}")
            try:
                raw = await self.viking_fs.read_file(entry.facet_json_uri, ctx=self.ctx)
                facet_set = DocumentFacetSet.model_validate_json(raw)
            except Exception as exc:
                reasons.append(f"facet set is missing or invalid for {doc.doc_id}: {exc}")
                continue
            if facet_set.doc_id != doc.doc_id:
                reasons.append(f"facet set identity changed: {doc.doc_id}")
            if _json_hash(facet_set.model_dump(mode="json")) != entry.facet_hash:
                reasons.append(f"facet content hash changed: {doc.doc_id}")
            facet_sets.append(facet_set)

        if reasons:
            self._fail(reasons)
        return facet_sets

    def _build_manifest(
        self,
        *,
        facet_sets: list[DocumentFacetSet],
        resource_documents: list[ResourceDocument],
        max_batch_chars: int,
        model_provenance: dict[str, Any],
        facet_uris: dict[str, str],
    ) -> DocumentFacetManifest:
        reasons: list[str] = []
        docs_by_id = _unique_by_doc_id(resource_documents, "resource documents", reasons)
        facets_by_id = _unique_by_doc_id(facet_sets, "facet sets", reasons)
        if set(docs_by_id) != set(facets_by_id):
            reasons.append("generated facet IDs do not match resource document IDs")
        if reasons:
            self._fail(reasons)
        return DocumentFacetManifest(
            version=FACET_MANIFEST_VERSION,
            generator_version=FACET_GENERATOR_VERSION,
            prompt_version=FACET_PROMPT_VERSION,
            schema_hash=_schema_hash(),
            max_batch_chars=max_batch_chars,
            model_provenance=model_provenance,
            entries=[
                DocumentFacetManifestEntry(
                    doc_id=doc.doc_id,
                    source_hash=resource_document_hash(doc),
                    facet_hash=_json_hash(facets_by_id[doc.doc_id].model_dump(mode="json")),
                    facet_json_uri=facet_uris[doc.doc_id],
                )
                for doc in resource_documents
            ],
        )

    @staticmethod
    def _fail(reasons: list[str]) -> None:
        raise FailedPreconditionError(
            "Document facet cache is missing or stale; rebuild facets before facet discovery",
            details={"reasons": reasons},
        )


def resource_document_hash(document: ResourceDocument) -> str:
    payload = document.model_dump(
        include={"doc_id", "resource_uri", "title", "source_sections"},
        mode="json",
    )
    return _json_hash(payload)


def _unique_by_doc_id(items: list[Any], label: str, reasons: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    duplicates: list[str] = []
    for item in items:
        if item.doc_id in result:
            duplicates.append(item.doc_id)
        result[item.doc_id] = item
    if duplicates:
        reasons.append(f"duplicate {label}: {sorted(set(duplicates))}")
    return result


def _schema_hash() -> str:
    return _json_hash(
        {
            "facet_set": DocumentFacetSet.model_json_schema(),
            "batch_response": DocumentFacetBatchResponse.model_json_schema(),
        }
    )


def _json_hash(value: Any) -> str:
    return _text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _text_hash(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
