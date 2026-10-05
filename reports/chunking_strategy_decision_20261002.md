# Why the project keeps its current chunking strategy

Date: 2026-10-02. Decision: **keep the current strategy** (sections from Docling section headers, then parent and child chunks cut with `RecursiveCharacterTextSplitter`). Markdown header splitting and Docling's own chunker were tested and rejected.

This report explains the choice, and gives the evidence and its limits. The tests were run from `experiments/chunking_comparison.py`; the two raw outputs are saved next to this file.

## The question

Should ingestion cut papers into chunks the way it does today, or switch to one of two alternatives?

| Variant | How the text is cut |
|---|---|
| **A, current** | Sections start at each Docling section header (references sections dropped). Each section is cut into parent chunks of up to 1,800 tokens, and each parent into child chunks of up to 400 tokens with 20% overlap, using `RecursiveCharacterTextSplitter`. Children are searched, parents are read by the model. |
| **B, markdown split** | The same extracted text written as markdown and split on `#` and `##` headers, then cut into parents and children with the same splitter. |
| **B2, markdown split with paragraph breaks kept** | As B, with the blank lines the markdown splitter removes put back before the parent and child cutting. |
| **C, Docling's HybridChunker** | Docling's own token-aware chunker that follows the document tree, children up to 400 tokens, parents built from chunks that share a heading. |

## How it was tested

- Nothing in the pipeline or in `data/` was changed. No Gemini or other API was used. Retrieval ran on the local embedding model (`BAAI/bge-base-en-v1.5`).
- A, B and B2 ran on the extracted text already saved in `data/text`. C cannot run from that text, because the saved files hold only text blocks, not Docling's document object. The PDFs were converted again into a scratch cache for C.
- Retrieval check, dense search only, top 5 chunks, as in the pipeline: for each golden-set question, the share of the gold passage's 5-word sequences found in the retrieved text. "Child" is the matched chunks alone, "parent" is the larger passages the model would read.
- **Scope: a trial on 2 papers and 6 golden questions** (`1-s2.0-S1877050925029461-main.pdf`, 10 pages, and `2510.26923v1.pdf`, 5 pages). A full run on all 12 papers was not done, by choice.

## Results

### 1. Heading levels: the papers have no heading hierarchy

Checked on all 12 papers from the saved text. Every section header in every paper is level 1.

| Paper | Section headers (all level 1) |
|---|---|
| 1-s2.0-S1877050925029461-main.pdf | 14 |
| 2404.03936v2.pdf | 49 |
| 2510.26923v1.pdf | 15 |
| 2602.10481v1.pdf | 51 |
| AI-Powered Lung Cancer Detection (VGG16 and CNN) | 30 |
| Assessment of Machine Learning Algorithms for Land Cover | 25 |
| Classification of NSCLC subtypes using lung microbiome | 24 |
| Deep learning-based approach to diagnose lung cancer | 39 |
| Land-Cover Classification Using Deep Learning | 18 |
| LungPaper.pdf | 30 |
| Utilizing Sentinel-2 Satellite Imagery for LULC and NDVI | 22 |
| fmed-12-1567119.pdf | 29 |

So a `#` against `##` split has no hierarchy to use. It cuts at every header, which is exactly where the current pipeline cuts.

### 2. Chunk statistics (2 papers, tokens counted with the pipeline's tokenizer)

| Variant | Parents | Children | Child mean / median / max tokens | Children under 100 tokens |
|---|---|---|---|---|
| A current | 26 | 48 | 253.5 / 269 / 401 | 7 |
| B markdown | 26 | 49 | 250.0 / 265 / 401 | 7 |
| B2 markdown, paragraph breaks kept | 26 | 48 | 253.5 / 269 / 401 | 7 |
| C Docling | 26 | 49 | 248.9 / 256 / 396 | 7 |

- All variants found the same 26 sections.
- **B differs from A only because of one side effect.** LangChain's markdown splitter removes every blank line, so the paragraph breaks fell from 137 to 0. The recursive splitter's first choice of where to cut is a paragraph break, so B's chunks moved: only 44 of its 49 chunks match A.
- **With the paragraph breaks put back (B2), the chunk set is identical to A:** 48 of 48 children in both, on both papers.

### 3. Retrieval on 6 golden questions (dense search, top 5)

| Variant | Source hit | Child coverage | Parent coverage | Context tokens per question |
|---|---|---|---|---|
| A current | 1.0 | 0.499 | 0.844 | 2,578 |
| B markdown | 1.0 | 0.454 | 0.844 | 2,553 |
| C Docling | 1.0 | 0.442 | 0.832 | 2,474 |

Per question, parent coverage was 1.00 for five of the six questions in all three variants. **One question (the SACL performance improvements one) scored 0.06 in all three variants,** so that miss is not caused by the chunking. The averages are driven by that single question.

### 4. Docling's chunker: cost and metadata

- It needs the PDFs converted again, because the saved text is not enough. Converting these 2 papers took 419 s (370 s for the 10-page one, which probably included warm-up, and 49 s for the 5-page one). Across all 218 pages of the corpus I estimate roughly 35 minutes to 2 hours on this CPU.
- 49 of its 69 chunks (71%) carried page numbers in their metadata. The pipeline's citation check needs a page range for every chunk.
- It produced 4 table-only chunks (the pipeline keeps tables apart) and 16 references chunks, which had to be left out.

## What this means

1. **The current strategy is already structure-aware.** It cuts at every section header, drops the references, and puts `Document | Section` in front of each chunk before embedding. The published advantages of header-based chunking over fixed-size slicing are already in the pipeline.
2. **A markdown split recreates the same sections.** Because the papers have flat headings, it finds the same boundaries. Its only visible difference came from the removed paragraph breaks, which can be fixed, and fixing it gives chunks identical to A. It would add a conversion step and a workaround, lose page numbers, and risk body lines starting with `#` being read as headers, with no gain.
3. **Docling's chunker showed no gain.** Retrieval was level with A (the differences are within noise), only 71% of its chunks carried page numbers, and it needs the PDFs converted again.
4. **Retrieval did not separate the variants.** Parent coverage of 0.84, 0.84 and 0.83 is a tie at this sample size.

## Limits of these results

- Two papers and six questions is a small trial. A difference of a few hundredths is noise, and the source hit of 1.0 is trivial because only two papers were in the pool.
- The golden-set passages were generated from the current chunks, so A has a small natural advantage. The coverage measure uses word sequences, so it does not depend on where a chunk starts, but the advantage is not zero. A's higher child coverage on one question (0.99 against 0.72 for B and 0.64 for C) probably reflects this.
- Dense search only. The pipeline also uses keyword (hybrid) search and a reranker, which were not part of this check.
- The Docling chunker counted tokens with the embedding model's tokenizer, the others with the pipeline's tokenizer, so sizes are close but not identical.
- Page numbers could not be checked for B, because a markdown export holds none.
- The 2-paper identity of A and B2 was not repeated on the other 10 papers. It is expected to hold, since all headings are level 1, but it was not measured.

## When to revisit

Change the strategy only if one of these happens:
- A full run on all 12 papers, or a larger golden set, shows B2 or C clearly ahead, for example by more than 0.05 on parent coverage.
- Future documents have real heading hierarchy (levels 2 and 3), which a heading-path prefix could use. The current papers have none.
- A future retrieval review shows many misses caused by where chunks are cut.

## Files

- `experiments/chunking_comparison.py`: the test script. Reproduce with `venv-rag-new/Scripts/python.exe experiments/chunking_comparison.py --papers "2510.26923,1-s2.0-S1877050925029461" --variants A,B,B2,C`. Without `--papers` it runs the whole corpus, which takes much longer because of the Docling conversion.
- `reports/chunking_comparison_20261002_182849.md` and `.json`: raw output for A, B and C with the retrieval check.
- `reports/chunking_comparison_20261002_185025.md` and `.json`: raw output for A, B and B2, structure only.
