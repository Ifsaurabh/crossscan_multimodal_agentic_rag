"""pipeline: ingests ONE document that passed the intake check.

    extract (Docling, Tesseract only for image-only pages)  -> quality check (too empty / lost table: rejected)
    -> sections and tables (references dropped)             -> guardrails (personal data redacted, unsafe dropped)
    -> parent/child chunks                                  -> text embeddings (children, tables)
    -> pictures: stored in the bucket under images/<name>/, embedded with CLIP
    -> domain (the upload folder; a new folder is a new domain; the fingerprint of the document is stored)
    -> ONE database transaction: the old version of the document is replaced by the new one, the manifest is
       updated last.

The document's name is `<domain>/<file>` (see doc_storage). Every dependency that is heavy or external (the models,
the bucket, the database) is a constructor argument, so the whole flow runs in tests with small fakes. The models
load on first use, once per process. A bad document raises errors.DocumentRejected; any other exception means "try
again later".

Every stage is timed and, where a model reads text, its tokens are counted (Prompt Guard in the guardrails, the
chunker's tokenizer, the embedding model's tokenizer). Each stage prints one line as it finishes, and the whole
attempt (ingested, rejected or failed) is saved by ingestion_reports, so no attempt is lost.
"""
import io
import mimetypes
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ingestion import (doc_storage, document_loader, domain_registry, embed_text, extraction, extraction_quality,
                       ingestion_manifest, ingestion_reports)
from ingestion.chunk_documents import chunk_document, count_tokens
from ingestion.chunking_config import CHUNKING_VERSION
from ingestion.errors import DocumentRejected
from ingestion.ingestion_guardrails import guard_document
from ingestion.prepare_documents import prepare_document
from ingestion.table_text import table_embedding_text
from shared.embedding_config import EMBEDDING_VERSION

PIPELINE_VERSION = "p1"   # raise it when the order or the rules of the stages change
IMAGES_PREFIX = doc_storage.IMAGES_PREFIX
IMAGE_CACHE_CONTROL = "public, max-age=86400"
NO_DOMAIN = "no_domain"
NO_TEXT = "no_text"
UNSAFE_CONTENT = "unsafe_content"


@dataclass
class IngestReport:
    source_pdf: str
    content_hash: str
    version_status: str = None
    domain: str = None
    domain_created: bool = False
    ocr_used: bool = False
    pages: int = 0
    parents: int = 0
    children: int = 0
    tables: int = 0
    images: int = 0
    images_skipped: int = 0
    image_bytes: int = 0
    pii_redactions: int = 0
    sections_dropped: int = 0
    tables_dropped: int = 0
    injection_flags: int = 0
    cache_entries_removed: int = 0
    stages: list = field(default_factory=list)  # [{"stage", "seconds", "tokens", "token_kind", "detail"}], in order

    @property
    def stage_seconds(self) -> dict:
        return {s["stage"]: s["seconds"] for s in self.stages}

    def to_dict(self) -> dict:
        return {**asdict(self), "stage_seconds": self.stage_seconds}


def load_default_text_embedder():
    """(embed, count): the embedding function, and a function giving the tokens the model really reads for a list of
    texts (each cut at the model's limit) and how many texts were cut."""
    model = embed_text.load_text_model()
    limit = model.max_seq_length or 512

    def embed(texts):
        return [list(map(float, v)) for v in embed_text.embed_texts(model, texts)]

    def count(texts):
        lengths = [len(ids) for ids in model.tokenizer(texts, truncation=False)["input_ids"]]
        return sum(min(n, limit) for n in lengths), sum(1 for n in lengths if n > limit)

    return embed, count


def load_default_image_embedder():
    """(embed_one, embed_batch): a picture's vector from its bytes, and the vectors of many pictures in batches
    (None in the place of a picture that cannot be decoded)."""
    from PIL import Image

    from ingestion import embed_images

    model, processor = embed_images.load_image_model()

    def embed_batch(datas: list) -> list:
        decoded, positions = [], []
        for position, data in enumerate(datas):
            try:
                image = Image.open(io.BytesIO(data))
                image.load()
            except Exception:
                continue  # undecodable: its place in the result stays None
            decoded.append(image)
            positions.append(position)
        result = [None] * len(datas)
        for position, vector in zip(positions, embed_images.embed_pil_images(model, processor, decoded)):
            result[position] = vector
        return result

    def embed_one(data: bytes):
        vector = embed_batch([data])[0]
        if vector is None:
            raise ValueError("the picture could not be decoded")
        return vector

    return embed_one, embed_batch


def default_guard_token_counter(text: str):
    from shared import query_guardrail

    return query_guardrail.count_guard_tokens(text)


def _default_connect():
    from shared.db import connection

    return connection()


def _live_line(name: str, entry: dict) -> str:
    parts = [f"[{name}] {entry['stage']:<14} {entry['seconds']:>8.2f}s"]
    if entry.get("tokens") is not None:
        parts.append(f"{entry['tokens']:,} {entry.get('token_kind') or ''} tokens")
    if entry.get("detail"):
        parts.append(", ".join(f"{k}={v}" for k, v in entry["detail"].items()))
    return "  ".join(parts)


def _print_live(line: str) -> None:
    print(line, flush=True)


class Pipeline:
    def __init__(self, storage, connect=None, text_embedder=None, text_counter=None, image_embedder=None, extract=None,
                 image_lister=None, image_batch_embedder=None, guard_token_counter=default_guard_token_counter,
                 recorder=ingestion_reports.save_report, emit=_print_live, clock=time.perf_counter):
        """storage          doc_storage.DocumentStorage (images are written to it)
        connect          () -> context manager giving a connection (default: the shared pool)
        text_embedder    (list of str) -> list of vectors; loaded on first use when not given
        text_counter     (list of str) -> (tokens the model reads, texts cut at its limit); None = not counted
        image_embedder   (bytes) -> vector; loaded on first use when not given
        image_batch_embedder  ([bytes]) -> [vector or None]; pictures are embedded in batches when this is given or loaded
        extract          (path, intake_result) -> extraction.ExtractionResult
        image_lister     (path) -> [{"page", "filename", "ext", "bytes"}]
        guard_token_counter  (text) -> Prompt Guard tokens, or None when Prompt Guard is not available
        recorder         ingestion_reports.save_report, or None to keep no record
        emit             where the live line of each stage goes (default: standard output)"""
        self.storage = storage
        self.connect = connect or _default_connect
        self._text_embedder = text_embedder
        self._text_counter = text_counter
        self._image_embedder = image_embedder
        self._image_batch_embedder = image_batch_embedder
        self.extract = extract or extraction.extract_document
        self._image_lister = image_lister
        self.guard_token_counter = guard_token_counter
        self.recorder = recorder
        self.emit = emit
        self.clock = clock

    # models are loaded when first needed, so a run that only rejects documents never pays for them
    def _load_text_model(self):
        if self._text_embedder is None:
            self._text_embedder, self._text_counter = load_default_text_embedder()

    def embed_texts(self, texts: list) -> list:
        if not texts:
            return []
        self._load_text_model()
        return self._text_embedder(texts)

    def count_text_tokens(self, texts: list):
        """(tokens, truncated) or (None, None) when the embedder in use cannot count."""
        if not texts or self._text_counter is None:
            return None, None
        return self._text_counter(texts)

    def embed_images(self, datas: list) -> list:
        """The vector of every picture (None for one that cannot be embedded), in batches when a batch embedder is available."""
        if self._image_embedder is None and self._image_batch_embedder is None:
            self._image_embedder, self._image_batch_embedder = load_default_image_embedder()
        if self._image_batch_embedder is not None:
            return self._image_batch_embedder(datas)
        vectors = []
        for data in datas:
            try:
                vectors.append(self._image_embedder(data))
            except Exception:  # a picture that cannot be decoded is skipped, it never fails the document
                vectors.append(None)
        return vectors

    def list_images(self, path) -> list:
        if self._image_lister is None:
            from ingestion.extract_images import extract_pdf_images
            self._image_lister = extract_pdf_images
        return self._image_lister(path)

    def _store_images(self, path, intake_result, stem: str, report: IngestReport) -> list:
        if intake_result.file_type != "pdf":
            return []
        stored = []
        images = self.list_images(path)
        for image, vector in zip(images, self.embed_images([i["bytes"] for i in images])):
            if vector is None:  # a picture that cannot be decoded is skipped, it never fails the document
                report.images_skipped += 1
                continue
            content_type = mimetypes.types_map.get(f".{image['ext']}", "application/octet-stream")
            self.storage.upload_bytes(f"{IMAGES_PREFIX}{stem}/{image['filename']}", image["bytes"], content_type,
                                      IMAGE_CACHE_CONTROL)
            report.image_bytes += len(image["bytes"])
            stored.append({"image_file": f"{stem}/{image['filename']}", "page": image["page"], "embedding": vector})
        return stored

    def _guard_tokens(self, texts: list):
        counts = [self.guard_token_counter(t) for t in texts]
        return None if any(c is None for c in counts) else sum(counts)

    def _save(self, report, status, started_at, reason_code=None, reason=None):
        """Keeps the record of this attempt. A failure to save it never changes what happened to the document."""
        if self.recorder is None:
            return
        try:
            with self.connect() as conn:
                self.recorder(conn, report, status=status, started_at=started_at, finished_at=datetime.now(timezone.utc),
                              reason_code=reason_code, reason=reason, chunking_version=CHUNKING_VERSION,
                              embedding_version=EMBEDDING_VERSION, pipeline_version=PIPELINE_VERSION)
        except Exception as e:
            self.emit(f"[{report.source_pdf}] WARNING the report of this attempt could not be saved: {type(e).__name__}: {e}")

    def __call__(self, path, intake_result) -> IngestReport:
        report = IngestReport(source_pdf=intake_result.file_name, content_hash=intake_result.content_hash,
                              version_status=intake_result.version_status)
        started_at = datetime.now(timezone.utc)
        try:
            self._ingest(path, intake_result, report)
        except DocumentRejected as rejected:
            self._save(report, ingestion_reports.REJECTED, started_at, rejected.reason_code, rejected.reason)
            raise
        except Exception as e:
            self._save(report, ingestion_reports.FAILED, started_at, type(e).__name__, str(e))
            raise
        self._save(report, ingestion_reports.INGESTED, started_at)
        return report

    def _ingest(self, path, intake_result, report: IngestReport) -> None:
        name = intake_result.file_name  # `<domain>/<file>`
        domain, _, file_name = name.partition("/")
        if not domain or not file_name:
            raise DocumentRejected(NO_DOMAIN, f"{name!r} is not in the form <domain>/<file>.")
        stem = f"{domain}/{Path(file_name).stem}"  # chunk ids and picture folders: unique across domains

        def stage(label, fn, measure=None):
            """Runs fn, times it, records the stage (with tokens and details from `measure(result)`), prints its line."""
            started = self.clock()
            try:
                value = fn()
            except Exception as e:
                entry = {"stage": label, "seconds": round(self.clock() - started, 2), "tokens": None, "token_kind": None,
                         "detail": {"error": f"{type(e).__name__}: {e}"[:300]}}
                report.stages.append(entry)
                self.emit(_live_line(name, entry))
                raise
            entry = {"stage": label, "seconds": round(self.clock() - started, 2), "tokens": None, "token_kind": None,
                     "detail": {}}
            if measure is not None:  # counting is not part of the stage's own time
                entry.update(measure(value))
            report.stages.append(entry)
            self.emit(_live_line(name, entry))
            return value

        extracted = stage("extract", lambda: self.extract(path, intake_result), lambda r: {"detail": {
            "pages": r.pages, "ocr_pages": intake_result.image_only_pages if r.ocr_used else 0, "blocks": len(r.blocks),
            "tables_found": r.tables_found, "tables_failed": len(r.tables_failed)}})
        report.ocr_used, report.pages = extracted.ocr_used, extracted.pages

        quality = stage("quality", lambda: extraction_quality.check_extraction(extracted),
                        lambda q: {"detail": {"passed": q.passed, **q.metrics}})
        if not quality.passed:
            raise DocumentRejected(quality.reason_code, quality.reason, quality.metrics)

        prepared = stage("prepare", lambda: prepare_document(extracted.blocks), lambda p: {"detail": {
            "sections": len(p["sections"]), "tables": len(p["tables"]), "references_dropped": p["references_dropped"]}})
        if not prepared["sections"]:
            raise DocumentRejected(NO_TEXT, "No text sections were left after the references were removed.")

        # guard_document redacts in place, so what Prompt Guard will read is captured first
        guard_inputs = [s["text"] for s in prepared["sections"]] + [t["text"] for t in prepared["tables"]]
        guarded, entry = stage("guardrails", lambda: guard_document(
            {"source_pdf": name, "sections": prepared["sections"], "tables": prepared["tables"]}), lambda r: {
                "tokens": self._guard_tokens(guard_inputs), "token_kind": "prompt_guard", "detail": {
                    "pii_redactions": r[1]["pii_redactions"], "sections_dropped": len(r[1]["sections_dropped"]),
                    "tables_dropped": len(r[1]["tables_dropped"]), "injection_flags": len(r[1]["injection_flags"])}}
        )
        report.pii_redactions = entry["pii_redactions"]
        report.sections_dropped, report.tables_dropped = len(entry["sections_dropped"]), len(entry["tables_dropped"])
        report.injection_flags = len(entry["injection_flags"])
        if not guarded["sections"]:
            raise DocumentRejected(UNSAFE_CONTENT, "Every section was dropped by the safety guardrails.",
                                   {"sections_dropped": entry["sections_dropped"]})

        def chunk_tokens(chunked):
            parents_list = chunked["parents"]
            child_tokens = sum(count_tokens(c["text"]) for p in parents_list for c in p["children"])
            return {"tokens": child_tokens, "token_kind": "tiktoken", "detail": {
                "parents": len(parents_list), "children": sum(len(p["children"]) for p in parents_list),
                "parent_tokens": sum(count_tokens(p["text"]) for p in parents_list)}}

        parents = stage("chunk", lambda: chunk_document(file_name, guarded["sections"], stem=stem), chunk_tokens)["parents"]
        children = [child for parent in parents for child in parent["children"]]
        tables = guarded.get("tables", [])
        if not children:
            raise DocumentRejected(NO_TEXT, "The text could not be cut into chunks.")

        texts = [embed_text.child_embedding_text(c) for c in children] + [table_embedding_text(t) for t in tables]

        def embed_measure(_):
            tokens, truncated = self.count_text_tokens(texts)
            return {"tokens": tokens, "token_kind": "bge", "detail": {"texts": len(texts), "truncated": truncated}}

        def embed_all():
            vectors = self.embed_texts(texts)
            return vectors[:len(children)], vectors[len(children):]

        child_vectors, table_vectors = stage("embed_text", embed_all, embed_measure)
        images = stage("images", lambda: self._store_images(path, intake_result, stem, report), lambda stored: {
            "detail": {"stored": len(stored), "skipped": report.images_skipped, "bytes": report.image_bytes}})

        fingerprint = domain_registry.fingerprint(child_vectors)

        with self.connect() as conn:
            assignment = stage("domain", lambda: domain_registry.assign_domain(conn, name, domain, fingerprint),
                               lambda a: {"detail": {"domain": a.name, "created": a.created}})
            counts = stage("replace", lambda: document_loader.replace_document(
                conn, source_pdf=name, stem=stem, parents=parents, child_vectors=child_vectors, tables=tables,
                table_vectors=table_vectors, images=images, domain=assignment.name,
                embedding_version=EMBEDDING_VERSION, clear_cache=True),
                lambda c: {"detail": {"parents": c.parents, "children": c.children, "tables": c.tables,
                                      "images": c.images, "cache_entries_removed": c.cache_entries_removed}})
            # last; commits everything
            stage("manifest_commit", lambda: ingestion_manifest.mark_ingested(conn, name, intake_result.content_hash))

        if intake_result.file_type == "pdf":  # pictures of an older version that the new one no longer has
            self.storage.delete_prefix(f"{IMAGES_PREFIX}{stem}/",
                                       keep=[f"{IMAGES_PREFIX}{i['image_file']}" for i in images])

        report.domain, report.domain_created = assignment.name, assignment.created
        report.parents, report.children, report.tables, report.images = (
            counts.parents, counts.children, counts.tables, counts.images)
        report.cache_entries_removed = counts.cache_entries_removed
