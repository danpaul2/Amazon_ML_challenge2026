# ML Challenge 2026: Business Entity Resolution

**Team Name:** Null Variance
**Team Members:** Daniel Verghese Paul, Hitesh N, K Siddharth, Adithyadev J  
**Submission Date:** 2nd October 2026

## 1. Problem statement

The task is to link each Source 1 business entity to zero or more matching records in Source 2 and Source 3. The submitted result includes one row for every Source 1 test entity and an empty match list for predicted singletons. The matching code and supplied task description use the precision-weighted macro F_0.5 metric.

## 2. Inputs and outputs

The pipeline reads UTF-8 TSV source files containing entity ID, business name, business address, and country. Its training workflow also reads the provided ground-truth TSV. The final outputs are `output/matching_results.tsv` and `output/candidate_pairs.tsv`; both use a Source 1 ID plus a comma-separated list of Source 2/3 IDs. The candidate output is the final candidate set sent to the scoring stage.

## 3. Preprocessing

Names are transliterated with Unidecode, lowercased, mapped through a small leetspeak translation table, stripped of selected web-domain suffixes and punctuation, and whitespace-normalized. Address text is transliterated, lowercased, stripped of punctuation, and whitespace-normalized. The blocking code also uses a repeat-character squeeze transformation. Separate stop-token sets exclude common legal name suffixes and generic address terms from selected features and blocking keys.

## 4. Blocking and candidate generation

Inference is partitioned by the country values found in Source 1. Within each country, an inverted index is built over Source 2 and Source 3 using name tokens, squeezed name tokens, a compressed name prefix, first name-token bigrams, address-number-plus-token keys, and address token bigrams. Posting lists longer than 200 records are removed. Candidates are scored by the sum of inverse posting-list lengths; when more than 300 candidates remain, the top 300 are retained. The retained list for every Source 1 record is written to `candidate_pairs.tsv` before matching decisions are made.

## 5. Matching approach

The production script computes 21 features per candidate pair: four direct name similarities, two squeezed-name similarities, maximum name similarity, name-token overlap and Jaccard similarity, a compound-name substring indicator, four address similarities, an empty-address indicator, address-token overlap and Jaccard similarity, digit overlap, exact-address evidence, name/address-conflict evidence, and a combined name/address score. RapidFuzz implements the string similarity measures.

A bundled LightGBM binary booster scores these features. The bundled threshold file is read at inference. The decision layer also applies a precision gate: it requires a name signal or strong address evidence, rejects low-name-similarity records with completely conflicting addresses, and includes a narrowly defined high-name-similarity rescue near the threshold.

## 6. Training and threshold selection

`run_pipeline.py --mode train` samples training Source 1 records, forms positives from ground truth and negative samples from blocked candidates, and trains a 600-round LightGBM model. Its tuning routine selects a threshold by macro F_0.5 over a training holdout while applying the precision gate. The submitted package contains the pre-existing model and threshold artifacts used to produce the supplied outputs; it does not claim that rerunning training will bitwise reproduce them.

## 7. Output generation and validation

The production script emits headers exactly as required and writes every Source 1 test ID, including records without candidates or matches. The supplied project progress record reports that the existing outputs passed the included validator: 1,732,544 rows in each file, no wrong prefixes, no self-matches, and all final matched IDs present in the candidate set. The submission packaging process separately syntax-checked the Python source and re-ran the provided validator for the matching file without the memory-intensive ID-existence option.

## 8. Reproducibility

The `code/business_entity_resolution/` directory contains the source, pinned dependencies, README, and model artifacts. Dataset files are excluded from the submission package and must be placed in `dataset/train` and `dataset/test` alongside `code/` as described in the README. The recommended entry point is `python code/business_entity_resolution/src/fast_predict.py` run from the unpacked submission root.

## 9. Limitations and computational considerations

Blocking can miss true matches that share none of its retained keys, and the 300-candidate cap can remove lower-ranked candidates. The model and threshold were created from a sampled training workflow, so their validation figures do not establish performance on hidden test labels. Inference intentionally loads and indexes Source 2/3 records one country at a time to bound memory, but it still requires the complete test data and can take substantial time on a large dataset.
