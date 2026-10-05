# Chunking strategy comparison (20261002_185025)

Purpose: evidence for the choice of chunking strategy. Produced by `experiments/chunking_comparison.py`; re-run it to reproduce. Nothing in the pipeline or in `data/` was changed, and no Gemini or other API was used.

## What was compared

| Variant | How the text is cut |
|---|---|
| A current | Sections from Docling section headers, then parent (1,800 tokens) and child (400 tokens) recursive splitting. The pipeline today. |
| B markdown | The same extracted blocks written as markdown, split on `#` and `##`, then the same parent/child splitting. |
| B2 markdown, paragraph breaks kept | As B, but the blank lines the markdown splitter removes are put back before the parent/child splitting. |
| C docling | Docling's HybridChunker (token-aware, follows the document tree), children up to 400 tokens, parents built from chunks sharing a heading. |

Papers: 2. Golden questions: 6. Embedding model: `BAAI/bge-base-en-v1.5`. Top-k: 5.

## 1. Heading levels in the extracted papers

| Paper | Section headers by level |
|---|---|
| 1-s2.0-S1877050925029461-main.pdf | {1: 14} |
| 2510.26923v1.pdf | {1: 15} |

## 2. Chunk statistics (tokens, counted with the pipeline's tokenizer)

| Variant | Parents | Parent mean / max | Children | Child mean / median / max | Children under 100 tokens |
|---|---|---|---|---|---|
| A current | 26 | 442.2 / 1568 | 48 | 253.5 / 269.0 / 401 | 7 |
| B markdown | 26 | 445.7 / 1575 | 49 | 250.0 / 265 / 401 | 7 |
| B2 markdown, paragraph breaks kept | 26 | 442.2 / 1568 | 48 | 253.5 / 269.0 / 401 | 7 |

## 3. Does the markdown split cut the text differently from the current sections?

- Papers whose sections are identical in A and B: **2 of 2**
- Sections: A 26, B 26
- Child chunks: A 48, B 49; present in both: 44
- Identical chunk sets: **False**
- Paragraph breaks (blank lines) kept inside the section text: A 137, B 0. The markdown splitter removes them, and the recursive splitter's first choice of cut is a paragraph break, so chunk boundaries move.
- Body-text blocks containing a line that starts with `#` (a markdown splitter would misread it as a header): 0

### With the paragraph breaks put back (B2)

- Papers whose sections are identical in A and B2: **2 of 2**; paragraph breaks A 137, B2 137
- Child chunks: A 48, B2 48; present in both: 48
- Identical chunk sets: **True**

## 4. Retrieval on the golden set (dense search, local model, no LLM)

- Source hit: a top-k chunk comes from the paper that holds the answer.
- Coverage: share of the gold passage's 5-word sequences found in the retrieved text. Child = the matched chunks alone. Parent = the larger passages the LLM would read.
- Context tokens: size of the passages the LLM would read per question.

| Variant | Source hit | Child coverage | Parent coverage | Context tokens |
|---|---|---|---|---|

## Limits of this comparison

- The golden contexts were generated from the current chunks, so variant A is the most natural fit. Coverage uses word sequences, so it does not depend on where a chunk starts, but A still has a small advantage.
- 36 questions are few. A difference of a few hundredths is noise.
- Dense search only. The pipeline also uses keyword (hybrid) search and a reranker, which are not part of this check.
- The Docling chunker counts tokens with the embedding model's tokenizer, the others with the pipeline's `cl100k_base` tokenizer, so sizes are close but not identical.
- Page numbers cannot be checked for B: a markdown export holds none.
