"""Persistent, strictly validated Document Card cache."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from openviking_cli.exceptions import FailedPreconditionError

from .config import WikiConfig
from .prompts import build_document_card_prompt
from .schemas import (
    DocumentCard,
    DocumentCardContent,
    DocumentCardManifest,
    DocumentCardManifestEntry,
    ResourceDocument,
)
from .uri import card_json_uri, card_manifest_uri, cards_dir
from .writer import WikiVikingFSWriter

CARD_MANIFEST_VERSION = 2
DOCUMENT_CARD_PROMPT_VERSION = "doc_card_v3"
LEGACY_CARD_MANIFEST_VERSION = 1
LEGACY_DOCUMENT_CARD_PROMPT_VERSION = "doc_card_v1"
LEGACY_DOCUMENT_CARD_SCHEMA_HASHES = {
    "sha256:9a14daaa99dc90e92c3dfdba4af2721db035022e4c7b7c66fa02ef294aadfc30"
}
LEGACY_CARD_FIELDS = {"main_points", "markdown"}


class DocumentCardStore:
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
        cards: list[DocumentCard],
        resource_documents: list[ResourceDocument],
        resource_uris: list[str],
        card_input_mode: str,
        max_card_input_chars: int,
        model_provenance: dict[str, Any] | None = None,
    ) -> DocumentCardManifest:
        manifest = self._build_manifest(
            cards=cards,
            resource_documents=resource_documents,
            resource_uris=resource_uris,
            card_input_mode=card_input_mode,
            max_card_input_chars=max_card_input_chars,
            model_provenance=model_provenance or {},
        )

        cache_root = cards_dir(self.config)
        if await self.viking_fs.exists(cache_root, ctx=self.ctx):
            await self.viking_fs.rm(cache_root, recursive=True, ctx=self.ctx)
        await self.writer.ensure_card_dirs()

        for card in cards:
            await self.writer.write_json(card_json_uri(self.config, card.doc_id), card)

        # The manifest is the commit marker. A partial write is never reusable.
        await self.writer.write_json(card_manifest_uri(self.config), manifest)
        return manifest

    async def read_manifest(self) -> DocumentCardManifest:
        uri = card_manifest_uri(self.config)
        if not await self.viking_fs.exists(uri, ctx=self.ctx):
            self._fail([f"card manifest is missing: {uri}"])
        try:
            raw = await self.viking_fs.read_file(uri, ctx=self.ctx)
            payload = json.loads(raw)
            if isinstance(payload, dict) and payload.get("version") == LEGACY_CARD_MANIFEST_VERSION:
                for entry in payload.get("entries", []):
                    if isinstance(entry, dict):
                        entry.pop("card_markdown_uri", None)
            return DocumentCardManifest.model_validate(payload)
        except FailedPreconditionError:
            raise
        except Exception as exc:
            self._fail([f"card manifest is invalid: {exc}"])

    async def load_validated(
        self,
        *,
        manifest: DocumentCardManifest,
        resource_documents: list[ResourceDocument],
        resource_uris: list[str],
    ) -> list[DocumentCard]:
        reasons: list[str] = []
        legacy_manifest = (
            manifest.version == LEGACY_CARD_MANIFEST_VERSION
            and manifest.prompt_version == LEGACY_DOCUMENT_CARD_PROMPT_VERSION
            and manifest.schema_hash in LEGACY_DOCUMENT_CARD_SCHEMA_HASHES
        )
        if manifest.version != CARD_MANIFEST_VERSION and not legacy_manifest:
            reasons.append(
                f"manifest version changed: cached={manifest.version} "
                f"current={CARD_MANIFEST_VERSION}"
            )
        if manifest.pipeline_version != self.config.pipeline_version:
            reasons.append(
                f"pipeline version changed: cached={manifest.pipeline_version} "
                f"current={self.config.pipeline_version}"
            )
        current_schema_hash = _schema_hash()
        if manifest.schema_hash != current_schema_hash and not legacy_manifest:
            reasons.append("DocumentCard schema changed")
        if manifest.prompt_version != DOCUMENT_CARD_PROMPT_VERSION and not legacy_manifest:
            reasons.append("document-card prompt version changed")
        if manifest.resource_uris != resource_uris:
            reasons.append("resource roots changed")

        docs_by_id = _unique_by_doc_id(resource_documents, "resource documents", reasons)
        entries_by_id = _unique_by_doc_id(manifest.entries, "card manifest entries", reasons)
        if set(docs_by_id) != set(entries_by_id):
            missing = sorted(set(docs_by_id) - set(entries_by_id))
            stale = sorted(set(entries_by_id) - set(docs_by_id))
            if missing:
                reasons.append(f"cards missing for document IDs: {missing}")
            if stale:
                reasons.append(f"stale card document IDs: {stale}")

        cards: list[DocumentCard] = []
        for doc in resource_documents:
            entry = entries_by_id.get(doc.doc_id)
            if entry is None:
                continue
            if entry.resource_uri != doc.resource_uri or entry.title != doc.title:
                reasons.append(f"document identity changed: {doc.doc_id}")
            if (
                not legacy_manifest
                and entry.prompt_hash != _text_hash(build_document_card_prompt(doc))
            ):
                reasons.append(f"document-card input or prompt changed: {doc.doc_id}")
            try:
                raw = await self.viking_fs.read_file(entry.card_json_uri, ctx=self.ctx)
                payload = json.loads(raw)
                if _json_hash(payload) != entry.card_hash:
                    reasons.append(f"card content hash changed: {doc.doc_id}")
                if legacy_manifest and isinstance(payload, dict):
                    payload = dict(payload)
                    for field in LEGACY_CARD_FIELDS:
                        payload.pop(field, None)
                card = DocumentCard.model_validate(payload)
            except Exception as exc:
                reasons.append(f"card is missing or invalid for {doc.doc_id}: {exc}")
                continue
            if (
                card.doc_id != entry.doc_id
                or card.resource_uri != entry.resource_uri
                or card.title != entry.title
            ):
                reasons.append(f"card identity changed: {doc.doc_id}")
            cards.append(card)

        if reasons:
            self._fail(reasons)
        return cards

    def _build_manifest(
        self,
        *,
        cards: list[DocumentCard],
        resource_documents: list[ResourceDocument],
        resource_uris: list[str],
        card_input_mode: str,
        max_card_input_chars: int,
        model_provenance: dict[str, Any],
    ) -> DocumentCardManifest:
        reasons: list[str] = []
        docs_by_id = _unique_by_doc_id(resource_documents, "resource documents", reasons)
        cards_by_id = _unique_by_doc_id(cards, "document cards", reasons)
        if set(docs_by_id) != set(cards_by_id):
            reasons.append("generated card IDs do not match resource document IDs")
        if reasons:
            self._fail(reasons)

        entries = []
        for doc in resource_documents:
            card = cards_by_id[doc.doc_id]
            entries.append(
                DocumentCardManifestEntry(
                    doc_id=doc.doc_id,
                    resource_uri=doc.resource_uri,
                    title=doc.title,
                    prompt_hash=_text_hash(build_document_card_prompt(doc)),
                    card_hash=_json_hash(card.model_dump(mode="json")),
                    card_json_uri=card_json_uri(self.config, doc.doc_id),
                )
            )
        return DocumentCardManifest(
            version=CARD_MANIFEST_VERSION,
            pipeline_version=self.config.pipeline_version,
            prompt_version=DOCUMENT_CARD_PROMPT_VERSION,
            schema_hash=_schema_hash(),
            card_input_mode=card_input_mode,
            max_card_input_chars=max_card_input_chars,
            resource_uris=resource_uris,
            model_provenance=model_provenance,
            entries=entries,
        )

    @staticmethod
    def _fail(reasons: list[str]) -> None:
        raise FailedPreconditionError(
            "Document Card cache is missing or stale; run build_cards before build_wiki",
            details={"reasons": reasons},
        )


def _unique_by_doc_id(items: list[Any], label: str, reasons: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    duplicates: list[str] = []
    for item in items:
        doc_id = str(item.doc_id)
        if doc_id in result:
            duplicates.append(doc_id)
        result[doc_id] = item
    if duplicates:
        reasons.append(f"duplicate {label}: {sorted(set(duplicates))}")
    return result


def _schema_hash() -> str:
    return _json_hash(DocumentCardContent.model_json_schema())


def _json_hash(value: Any) -> str:
    return _text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _text_hash(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
