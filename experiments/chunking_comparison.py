"""Chunking strategy comparison - evidence for why one strategy was chosen.

Compares three ways of cutting the same 12 papers into chunks:

  A  current   : sections from Docling section headers (prepare_documents.py), then the parent/child
                 recursive splitter (chunk_documents.py). This is what the pipeline does today.
  B  markdown  : the same extracted blocks written as markdown, split on "#" and "##" with
                 MarkdownHeaderTextSplitter, then the SAME parent/child recursive splitter.
  C  docling   : Docling's own HybridChunker (token-aware, follows the document tree), children of
                 the same size, parents built from the chunks that share a heading.

A and B run from the saved text in data/text (no extraction). C needs Docling's document object,
which data/text does not hold, so the PDFs are converted once into a cache folder (data is not
touched). Nothing here changes the pipeline, writes to data/, calls Gemini or uses any API.

Two kinds of results:
  1. Structure: section/chunk counts, sizes, whether B cuts the text differently from A, heading levels.
  2. Retrieval on the golden set with the local embedding model, dense search only, top-k as in the
     pipeline: source hit and how much of the gold passage the retrieved text covers.

Run from the project root:
    venv-rag-new/Scripts/python.exe experiments/chunking_comparison.py --variants A,B,C
"""
import argparse
import collections
import datetime
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
from langchain_text_splitters import MarkdownHeaderTextSplitter

import chunk_documents as cd
from chunking_config import CHILD_MAX_TOKENS, CHILD_OVERLAP_TOKENS, PARENT_MAX_TOKENS, PARENT_OVERLAP_TOKENS
from embedding_config import TEXT_MODEL_NAME
from prepare_documents import is_references_heading, split_into_sections
from retrieval_config import VECTOR_TOP_K

TEXT_DIR = ROOT / "data" / "text"
GOLDEN_PATH = ROOT / "data" / "golden_set.jsonl"
RAW_DIR = ROOT / "data" / "raw"
SHINGLE = 5


# ---------- helpers ----------

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def shingles(text: str, n: int = SHINGLE) -> set:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def load_blocks() -> dict:
    docs = {}
    for path in sorted(TEXT_DIR.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        docs[d["source_pdf"]] = d["blocks"]
    return docs


def load_golden() -> list:
    rows = []
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ---------- variant A: current sections ----------

def sections_current(blocks):
    sections, _tables = split_into_sections(blocks)
    return [s for s in sections if s["text"] and not is_references_heading(s["heading"])]


# ---------- variant B: markdown split ----------

_HASH_LINE = re.compile(r"^\s{0,3}#{1,6}\s")


PARA_MARK = "<<PARA>>"


def blocks_to_markdown(blocks, keep_paragraphs=False):
    """Markdown from the extracted blocks. Tables are left out, as the pipeline keeps them apart.
    Also counts body-text lines that start with '#', which a markdown splitter would misread as headers."""
    lines, stray = [], 0
    for b in blocks:
        if b["label"] == "table":
            continue
        if b["label"] == "section_header":
            lines.append("#" * max(1, int(b["level"] or 1)) + " " + b["text"])
        else:
            if any(_HASH_LINE.match(ln) for ln in b["text"].splitlines()):
                stray += 1
            lines.append(b["text"])
        lines.append(PARA_MARK if keep_paragraphs else "")
    return "\n".join(lines), stray


def sections_markdown(blocks, keep_paragraphs=False):
    """keep_paragraphs: put a marker line between blocks before the markdown split and turn it back into a
    blank line afterwards, because MarkdownHeaderTextSplitter drops blank lines."""
    md, stray = blocks_to_markdown(blocks, keep_paragraphs)
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")], strip_headers=True)
    out = []
    for d in splitter.split_text(md):
        h1, h2 = d.metadata.get("h1"), d.metadata.get("h2")
        if is_references_heading(h1) or is_references_heading(h2):
            continue
        text = d.page_content
        if keep_paragraphs:
            text = text.replace("\n" + PARA_MARK + "\n", "\n\n").replace(PARA_MARK, "")
        text = text.strip()
        if text:
            out.append({"heading": h2 or h1, "text": text})
    return out, stray


# ---------- parent/child from sections (A and B) ----------

def build_text_variant(sections_by_doc: dict) -> dict:
    parents, children = [], []
    for src, sections in sections_by_doc.items():
        for s in sections:
            for ptext in cd.split_text(s["text"], PARENT_MAX_TOKENS, PARENT_OVERLAP_TOKENS):
                pid = len(parents)
                parents.append({"source_pdf": src, "section": s["heading"], "text": ptext})
                for ctext in cd.split_text(ptext, CHILD_MAX_TOKENS, CHILD_OVERLAP_TOKENS):
                    children.append({"source_pdf": src, "section": s["heading"], "text": ctext, "parent_id": pid})
    return {"parents": parents, "children": children}


# ---------- variant C: Docling's own chunker ----------

def build_docling_variant(cache_dir: Path, only_papers: set = None) -> dict:
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from docling_core.types.doc import DoclingDocument
    from transformers import AutoTokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(TEXT_MODEL_NAME), max_tokens=CHILD_MAX_TOKENS,
    )
    chunker = HybridChunker(tokenizer=tokenizer)
    converter = None
    parents, children = [], []
    extra = {"conversion_seconds": {}, "table_chunks_excluded": 0, "references_chunks_excluded": 0,
             "chunks_total": 0, "chunks_with_page": 0}

    for pdf in sorted(RAW_DIR.glob("*.pdf")):
        if only_papers is not None and pdf.name not in only_papers:
            continue
        cache = cache_dir / (pdf.stem + ".docling.json")
        if cache.exists():
            doc = DoclingDocument.load_from_json(cache)
        else:
            if converter is None:
                converter = DocumentConverter()
            start = time.perf_counter()
            doc = converter.convert(str(pdf)).document
            extra["conversion_seconds"][pdf.name] = round(time.perf_counter() - start, 1)
            doc.save_as_json(cache)
            print(f"   converted {pdf.name} in {extra['conversion_seconds'][pdf.name]}s", flush=True)

        group_key, group = None, []

        def flush_group():
            """Packs consecutive chunks that share a heading into parents of up to PARENT_MAX_TOKENS."""
            nonlocal group
            if not group:
                return
            buf, buf_tokens, buf_children = [], 0, []
            for text, ch in group:
                t = cd.count_tokens(text)
                if buf and buf_tokens + t > PARENT_MAX_TOKENS:
                    emit(buf, buf_children)
                    buf, buf_tokens, buf_children = [], 0, []
                buf.append(text)
                buf_tokens += t
                buf_children.append(ch)
            emit(buf, buf_children)
            group = []

        def emit(texts, kids):
            pid = len(parents)
            parents.append({"source_pdf": pdf.name, "section": kids[0]["section"], "text": "\n\n".join(texts)})
            for k in kids:
                k["parent_id"] = pid
                children.append(k)

        for chunk in chunker.chunk(dl_doc=doc):
            extra["chunks_total"] += 1
            items = chunk.meta.doc_items or []
            labels = {str(getattr(i, "label", "")) for i in items}
            headings = tuple(chunk.meta.headings or ())
            if items and all(l.endswith("table") for l in labels):
                extra["table_chunks_excluded"] += 1
                continue
            if headings and is_references_heading(headings[-1]):
                extra["references_chunks_excluded"] += 1
                continue
            if any(getattr(i, "prov", None) for i in items):
                extra["chunks_with_page"] += 1
            if headings != group_key:
                flush_group()
                group_key = headings
            group.append((chunk.text, {
                "source_pdf": pdf.name, "section": headings[-1] if headings else None, "text": chunk.text,
            }))
        flush_group()
    return {"parents": parents, "children": children, "extra": extra}


# ---------- statistics ----------

def token_stats(texts: list) -> dict:
    counts = [cd.count_tokens(t) for t in texts]
    if not counts:
        return {"n": 0}
    return {
        "n": len(counts), "mean": round(statistics.mean(counts), 1), "median": statistics.median(counts),
        "min": min(counts), "max": max(counts), "under_100": sum(1 for c in counts if c < 100),
    }


def variant_stats(variant: dict) -> dict:
    return {
        "parents": token_stats([p["text"] for p in variant["parents"]]),
        "children": token_stats([c["text"] for c in variant["children"]]),
    }


# ---------- retrieval check (dense only, local model, no API) ----------

def evaluate(variant: dict, golden: list, model) -> dict:
    children = variant["children"]
    texts = [f"Document: {c['source_pdf']} | Section: {c['section']}\n\n{c['text']}" for c in children]
    start = time.perf_counter()
    vecs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    embed_s = round(time.perf_counter() - start, 1)

    rows = []
    for g in golden:
        q = model.encode(g["input"], normalize_embeddings=True)
        top = np.argsort(-(vecs @ q))[:VECTOR_TOP_K]
        top_children = [children[i] for i in top]
        parent_ids = list(dict.fromkeys(c["parent_id"] for c in top_children))
        parent_texts = [variant["parents"][p]["text"] for p in parent_ids]
        gold = shingles(" ".join(g["context"]))
        if not gold:
            continue
        child_cov = len(gold & shingles(" ".join(c["text"] for c in top_children))) / len(gold)
        parent_cov = len(gold & shingles(" ".join(parent_texts))) / len(gold)
        rows.append({
            "question": g["input"], "source_pdf": g["source_pdf"],
            "source_hit": float(any(c["source_pdf"] == g["source_pdf"] for c in top_children)),
            "child_cov": child_cov, "parent_cov": parent_cov,
            "context_tokens": sum(cd.count_tokens(t) for t in parent_texts),
        })

    def mean(key):
        return round(statistics.mean(r[key] for r in rows), 3)

    return {
        "n_questions": len(rows), "embed_seconds": embed_s,
        "source_hit": mean("source_hit"), "child_coverage": mean("child_cov"),
        "parent_coverage": mean("parent_cov"), "context_tokens": round(statistics.mean(r["context_tokens"] for r in rows)),
        "rows": rows,
    }


# ---------- A against B ----------

def compare_ab(a: dict, b: dict, sections_a: dict, sections_b: dict) -> dict:
    a_children = collections.Counter(norm(c["text"]) for c in a["children"])
    b_children = collections.Counter(norm(c["text"]) for c in b["children"])
    same_sections = sum(
        1 for src in sections_a
        if [norm(s["text"]) for s in sections_a[src]] == [norm(s["text"]) for s in sections_b.get(src, [])]
    )
    return {
        "papers_with_identical_sections": same_sections, "papers": len(sections_a),
        "sections_a": sum(len(v) for v in sections_a.values()), "sections_b": sum(len(v) for v in sections_b.values()),
        "children_a": sum(a_children.values()), "children_b": sum(b_children.values()),
        "identical_chunk_sets": a_children == b_children,
        "children_in_both": sum((a_children & b_children).values()),
        "paragraph_breaks_a": sum(s["text"].count("\n\n") for v in sections_a.values() for s in v),
        "paragraph_breaks_b": sum(s["text"].count("\n\n") for v in sections_b.values() for s in v),
    }


# ---------- report ----------

def md_report(res: dict) -> str:
    L = []
    L.append(f"# Chunking strategy comparison ({res['stamp']})\n")
    L.append("Purpose: evidence for the choice of chunking strategy. Produced by `experiments/chunking_comparison.py`; "
             "re-run it to reproduce. Nothing in the pipeline or in `data/` was changed, and no Gemini or other API was used.\n")
    L.append("## What was compared\n")
    L.append("| Variant | How the text is cut |\n|---|---|")
    L.append("| A current | Sections from Docling section headers, then parent (1,800 tokens) and child (400 tokens) recursive splitting. The pipeline today. |")
    L.append("| B markdown | The same extracted blocks written as markdown, split on `#` and `##`, then the same parent/child splitting. |")
    L.append("| B2 markdown, paragraph breaks kept | As B, but the blank lines the markdown splitter removes are put back before the parent/child splitting. |")
    L.append("| C docling | Docling's HybridChunker (token-aware, follows the document tree), children up to 400 tokens, parents built from chunks sharing a heading. |\n")
    L.append(f"Papers: {res['n_papers']}. Golden questions: {res['n_golden']}. Embedding model: `{TEXT_MODEL_NAME}`. Top-k: {VECTOR_TOP_K}.\n")

    L.append("## 1. Heading levels in the extracted papers\n")
    L.append("| Paper | Section headers by level |\n|---|---|")
    for src, lv in res["heading_levels"].items():
        L.append(f"| {src[:60]} | {lv} |")
    L.append("")

    L.append("## 2. Chunk statistics (tokens, counted with the pipeline's tokenizer)\n")
    L.append("| Variant | Parents | Parent mean / max | Children | Child mean / median / max | Children under 100 tokens |\n|---|---|---|---|---|---|")
    for name, v in res["variants"].items():
        p, c = v["stats"]["parents"], v["stats"]["children"]
        L.append(f"| {name} | {p['n']} | {p.get('mean')} / {p.get('max')} | {c['n']} | {c.get('mean')} / {c.get('median')} / {c.get('max')} | {c.get('under_100')} |")
    L.append("")

    if "ab" in res:
        ab = res["ab"]
        L.append("## 3. Does the markdown split cut the text differently from the current sections?\n")
        L.append(f"- Papers whose sections are identical in A and B: **{ab['papers_with_identical_sections']} of {ab['papers']}**")
        L.append(f"- Sections: A {ab['sections_a']}, B {ab['sections_b']}")
        L.append(f"- Child chunks: A {ab['children_a']}, B {ab['children_b']}; present in both: {ab['children_in_both']}")
        L.append(f"- Identical chunk sets: **{ab['identical_chunk_sets']}**")
        L.append(f"- Paragraph breaks (blank lines) kept inside the section text: A {ab['paragraph_breaks_a']}, B {ab['paragraph_breaks_b']}. "
                 "The markdown splitter removes them, and the recursive splitter's first choice of cut is a paragraph break, so chunk boundaries move.")
        L.append(f"- Body-text blocks containing a line that starts with `#` (a markdown splitter would misread it as a header): {res['stray_hash_blocks']}\n")

    if "ab2" in res:
        ab = res["ab2"]
        L.append("### With the paragraph breaks put back (B2)\n")
        L.append(f"- Papers whose sections are identical in A and B2: **{ab['papers_with_identical_sections']} of {ab['papers']}**; paragraph breaks A {ab['paragraph_breaks_a']}, B2 {ab['paragraph_breaks_b']}")
        L.append(f"- Child chunks: A {ab['children_a']}, B2 {ab['children_b']}; present in both: {ab['children_in_both']}")
        L.append(f"- Identical chunk sets: **{ab['identical_chunk_sets']}**\n")

    L.append("## 4. Retrieval on the golden set (dense search, local model, no LLM)\n")
    L.append("- Source hit: a top-k chunk comes from the paper that holds the answer.")
    L.append(f"- Coverage: share of the gold passage's {SHINGLE}-word sequences found in the retrieved text. Child = the matched chunks alone. Parent = the larger passages the LLM would read.")
    L.append("- Context tokens: size of the passages the LLM would read per question.\n")
    L.append("| Variant | Source hit | Child coverage | Parent coverage | Context tokens |\n|---|---|---|---|---|")
    for name, v in res["variants"].items():
        e = v.get("eval")
        if e:
            note = " (same chunks as A)" if v.get("eval_reused") else ""
            L.append(f"| {name}{note} | {e['source_hit']} | {e['child_coverage']} | {e['parent_coverage']} | {e['context_tokens']} |")
    L.append("")

    if "docling_extra" in res:
        x = res["docling_extra"]
        L.append("## 5. Docling chunker details\n")
        L.append(f"- Chunks produced: {x['chunks_total']}; table-only chunks left out (the pipeline keeps tables apart): {x['table_chunks_excluded']}; references chunks left out: {x['references_chunks_excluded']}")
        share = round(100 * x["chunks_with_page"] / max(1, x["chunks_total"]))
        L.append(f"- Chunks that carry page numbers in their metadata: {x['chunks_with_page']} ({share}%)")
        if x["conversion_seconds"]:
            L.append(f"- PDF conversion needed for this variant (not stored in data/text): {sum(x['conversion_seconds'].values()):.0f} s in total for {len(x['conversion_seconds'])} papers\n")

    L.append("## Limits of this comparison\n")
    L.append("- The golden contexts were generated from the current chunks, so variant A is the most natural fit. Coverage uses word sequences, so it does not depend on where a chunk starts, but A still has a small advantage.")
    L.append("- 36 questions are few. A difference of a few hundredths is noise.")
    L.append("- Dense search only. The pipeline also uses keyword (hybrid) search and a reranker, which are not part of this check.")
    L.append("- The Docling chunker counts tokens with the embedding model's tokenizer, the others with the pipeline's `cl100k_base` tokenizer, so sizes are close but not identical.")
    L.append("- Page numbers cannot be checked for B: a markdown export holds none.\n")
    return "\n".join(L)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--out-dir", default=str(ROOT / "reports"))
    ap.add_argument("--cache-dir", default=str(ROOT / "experiments" / "_docling_cache"))
    ap.add_argument("--no-eval", action="store_true", help="structure only, skip embeddings")
    ap.add_argument(
        "--papers", default=None,
        help="limit to some papers: a number (the first N) or comma-separated parts of file names. "
             "Golden questions are limited to the same papers. Use this for a trial run.",
    )
    args = ap.parse_args()
    wanted = [v.strip().upper() for v in args.variants.split(",")]

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    blocks_by_doc = load_blocks()
    golden = load_golden()
    if args.papers:
        if args.papers.isdigit():
            keep = list(blocks_by_doc)[:int(args.papers)]
        else:
            parts = [p.strip().lower() for p in args.papers.split(",")]
            keep = [s for s in blocks_by_doc if any(p in s.lower() for p in parts)]
        blocks_by_doc = {s: blocks_by_doc[s] for s in keep}
        golden = [g for g in golden if g["source_pdf"] in blocks_by_doc]
        print(f"Limited to {len(blocks_by_doc)} paper(s), {len(golden)} golden question(s): {keep}", flush=True)
    res = {"stamp": stamp, "n_papers": len(blocks_by_doc), "n_golden": len(golden), "variants": {}}
    res["heading_levels"] = {
        src: dict(collections.Counter(b["level"] for b in blocks if b["label"] == "section_header"))
        for src, blocks in blocks_by_doc.items()
    }

    variants = {}
    sections_a = {src: sections_current(b) for src, b in blocks_by_doc.items()}
    if "A" in wanted:
        print("Building A (current)...", flush=True)
        variants["A current"] = build_text_variant(sections_a)
    if "B" in wanted:
        print("Building B (markdown split)...", flush=True)
        sections_b, stray = {}, 0
        for src, b in blocks_by_doc.items():
            sections_b[src], s = sections_markdown(b)
            stray += s
        res["stray_hash_blocks"] = stray
        variants["B markdown"] = build_text_variant(sections_b)
        if "A current" in variants:
            res["ab"] = compare_ab(variants["A current"], variants["B markdown"], sections_a, sections_b)
    if "B2" in wanted:
        print("Building B2 (markdown split, paragraph breaks kept)...", flush=True)
        sections_b2 = {src: sections_markdown(b, keep_paragraphs=True)[0] for src, b in blocks_by_doc.items()}
        variants["B2 markdown, paragraph breaks kept"] = build_text_variant(sections_b2)
        if "A current" in variants:
            res["ab2"] = compare_ab(variants["A current"], variants["B2 markdown, paragraph breaks kept"], sections_a, sections_b2)
    if "C" in wanted:
        print("Building C (Docling chunker; converts PDFs the first time)...", flush=True)
        c = build_docling_variant(Path(args.cache_dir), set(blocks_by_doc))
        res["docling_extra"] = c.pop("extra")
        variants["C docling"] = c

    model = None
    if not args.no_eval:
        from sentence_transformers import SentenceTransformer
        print(f"Loading {TEXT_MODEL_NAME}...", flush=True)
        model = SentenceTransformer(TEXT_MODEL_NAME)

    for name, v in variants.items():
        entry = {"stats": variant_stats(v)}
        if model is not None:
            ab_key = {"B markdown": "ab", "B2 markdown, paragraph breaks kept": "ab2"}.get(name)
            if ab_key and res.get(ab_key, {}).get("identical_chunk_sets"):
                entry["eval"] = res["variants"]["A current"]["eval"]
                entry["eval_reused"] = True
            else:
                print(f"Embedding and evaluating {name} ({len(v['children'])} chunks)...", flush=True)
                entry["eval"] = evaluate(v, golden, model)
        res["variants"][name] = entry

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"chunking_comparison_{stamp}.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (out / f"chunking_comparison_{stamp}.md").write_text(md_report(res), encoding="utf-8")
    print(f"saved {out / ('chunking_comparison_' + stamp + '.md')}")


if __name__ == "__main__":
    main()
