# Business Entity Resolution

This package generates entity matches from a Source 1 business file to Source 2 and Source 3 business files. The supplied production entry point is `src/fast_predict.py`; it uses the included LightGBM booster and decision threshold. `src/run_pipeline.py` provides the equivalent train, tune, and prediction workflow.

## Environment

Use Python 3.11 (the package was syntax-checked with Python 3.11). Install the pinned runtime dependencies from the directory containing this README:

```bash
python -m pip install -r requirements.txt
```

## Input layout

 After unzipping, place the supplied files beside `code/` as follows:

```text
<submission-root>/
  dataset/
    train/
      train_source1.tsv
      train_source2.tsv
      train_source3.tsv
      train_ground_truth.tsv
    test/
      test_source1.tsv
      test_source2.tsv
      test_source3.tsv
```

All inputs are UTF-8 TSVs with `entity_id`, `business_name`, `business_address`, and `country` fields. Training also uses the ground-truth file's `source1_entity_id` and comma-separated `matched_entity_ids` fields.

## Reproduce the supplied prediction workflow

Run from the submission root. The included `models/lgbm_model.txt` and `models/threshold.pkl` are required for prediction.

```bash
python code/business_entity_resolution/src/fast_predict.py
```

The program processes each country separately, normalizes names and addresses, builds an inverted index over Source 2 and Source 3, scores its candidates with the bundled 21-feature LightGBM model, and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

`candidate_pairs.tsv` contains the final ranked blocking candidates presented to model inference. `matching_results.tsv` contains the candidates retained by the model threshold and precision gate.

For the slower general pipeline, use:

```bash
python code/business_entity_resolution/src/run_pipeline.py --mode predict
```

To tune the included model's threshold on training data, run `python code/business_entity_resolution/src/run_pipeline.py --mode tune`. To train a replacement model and then predict, run `python code/business_entity_resolution/src/run_pipeline.py --mode full`. Training uses a sampled subset defined in `run_pipeline.py` and overwrites the model and threshold artifacts; retain copies of the shipped artifacts if the provided predictions must be reproduced exactly.

## Implementation details

Names are transliterated with Unidecode, lowercased, leetspeak-normalized, stripped of selected domain suffixes and punctuation, and repeat-character-squeezed for alternate blocking keys. Addresses are transliterated, lowercased, and tokenized. Blocking keys are country-prefixed name tokens, squeezed name tokens, a compressed name prefix, name bigrams, address number-plus-token keys, and address token bigrams. Posting lists over 200 are dropped; more than 300 candidates are ranked by inverse posting-list size and truncated.

The scorer computes 21 name, address, overlap, and conflict features using RapidFuzz similarities. A LightGBM score is compared to the bundled threshold. The precision gate requires a name signal or strong address agreement and adds protections for conflicting addresses.

## Validate outputs

The competition-provided `utils/validate_submission.py` can be run from the original resource folder:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

It checks headers, one output row per Source 1 ID, ID prefixes, duplicates, and that final matches are candidates. Add `--check-ids` when sufficient memory is available to verify all Source 2/3 IDs.

## Common issues

- Run commands from the submission root; the scripts resolve `dataset/` and `output/` relative to that location.
- Do not open or rewrite the TSV outputs in spreadsheet software, which can alter tabs or long IDs.
- Full inference needs the complete test files and can take substantial time; the final validated outputs are already included in `output/`.
