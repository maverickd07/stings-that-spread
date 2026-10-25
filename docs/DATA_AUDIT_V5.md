# Data audit for the final paper (v5, 25 October 2026)

The final paper's Tables 2–6 and all plotted inputs were independently recomputed from saved per-request generations, original snapshots, and cached Phi-4 judgments. The audit checked cohort identity, paired units, numerators/denominators, Wilson intervals, exact Fisher and McNemar tests, the quoted snapshot-bootstrap interval, token-score means, AUCs, cue-effect bootstrap intervals, reflection rates, and saved activation-patching results. No additional model inference was run. Raw experimental inputs were not changed.

## Corrections

- Figure 6 is an independent four-way reply-style recoding of the same replies, with a different Phi-4 prompt. It is not a decomposition of the binary acceptance outcome in Table 2. The previous “conditional yes (scored accept)” description was incorrect and was removed. `runs/fourway_binary_cross_v5.csv` reports the label crosswalk. Bar widths now use exact category counts; displayed percentages are rounded.
- Lift-test initial replies were generated afresh and keyword-parsed. The denominators are not Table 2 refusal counts. Qwen values: 25/79 initial acceptances leave 54 refusers; Mistral values: 21/79 leave 58; Mistral penalty: 3/79 leave 76. All reported Qwen/Mistral lift and re-ask counts were reproduced without changes.
- The Llama reflection sweep had API failures. The earlier summary used a common denominator of 61, counting one or two additional failed calls in each plotted L2 condition as refusals. Rates and Wilson intervals now use successful, Phi-4-labelled replies: 42/59 (generic), 39/59 (penalty), 50/60 (public test), 44/59 (victim view), 41/60 (golden rule), 45/60 (identity), 31/60 (rule reminder). Each condition attempted 79 generations; 19–20 failed. These are conditional-on-response rates and missingness may bias them. Qwen and Mistral reflection counts are unchanged. Unreported Llama lift trajectories with failed calls are also excluded by the corrected analysis.
- The previous blanket claim of 92–100% keyword/judge agreement omitted Llama. Agreement is 246/316 = 77.85% for Llama forced-format replies; the other model/seed groups range from 91.62% to 100%. `runs/judge_keyword_agreement_v5.csv` gives exact cohorts and counts.
- First-token P(ACCEPT) is normalised over ACCEPT/REFUSE prefix probability mass. This definition is now explicit. Unparseable PM tool output counts as unsent; it is not reinterpreted or relabelled. The main memory experiment contains 12 such Mistral outputs out of 1,659 attempts, and none for Qwen.
- The activation-patching maximum cosine is approximately 0.20 (raw maximum 0.200512), rather than the strict rounded bound “≤0.2”.

## Verified unchanged

All Table 2 acceptance/N0 counts, all Table 3 strata and marginal Wilson intervals, Table 4 recruiting counts and exact paired tests, Table 5 Qwen/Mistral lift counts, and Table 6 memory counts/intervals match the stored records and judgments. The main reported effects remain: approximately 17–38 pp recruiting increases in three models; Qwen offer-only/no-exchange recruitment 228/237 vs 122/237; values-refusal and observed-refusal recruitment 13/237 and 65/237; Mistral values/combined persistence 50/58 and 70/77. The Qwen N0-associated L1 difference is +15.7 pp, with the stated bootstrap interval [−7.0, +35.7] pp.

## Preserving the record

Historical run reports and copied source files remain unchanged. For the corrected values analysis, use `runs/values_corrected_v5/results.json` and `summary.md`, with per-condition generation coverage in `generation_coverage.csv`. The new `sting.values_corrected` analysis excludes failed or empty outputs rather than counting them as refusals. Its `--analysis-out` option can write a corrected analysis separately from raw inputs. `figures/make_figures.py` contains the verified final plotting inputs.

This audit verifies arithmetic and consistency with the stored model judgments; it does not establish human-validated ground truth for those judgments. The public release still excludes full raw logs and saved game states.
