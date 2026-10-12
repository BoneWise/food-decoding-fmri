# Decoding results

Whole-brain fMRI patterns predict high versus low subjective food value in participants who were not used for training. CNN and linear SVM decoding performed above the 50% chance level at all three input settings.

## Task and evaluation

- **Target:** current desire to eat the pictured food. Ratings 1 and 4 define low and high subjective value.
- **Participants:** 30 participants in ten-fold cross-validation, with three participants held out in each fold.
- **Inputs:** k = 1 uses single-trial beta maps. For k = 3 or 6, non-overlapping groups of trials with the same rating are averaged within each participant; incomplete groups are discarded.
- **Models:** a 3D-CNN and a linear SVM evaluated using matched inputs and participant splits within each k setting.

## Cross-participant performance

Balanced accuracy (BA) is the average of the two class recalls. BA is calculated for each held-out participant, and participants receive equal weight in the group summary. SD is the sample standard deviation across participants; confidence intervals are participant-bootstrap 95% intervals.

| k | Inputs | Model | Mean BA ± SD (%) | 95% CI (%) |
|---:|---:|---|---:|---:|
| 1 | 7,038 | CNN | 53.71 ± 6.56 | 51.26–55.87 |
| 1 | 7,038 | SVM | 57.87 ± 5.96 | 55.73–59.92 |
| 3 | 2,326 | CNN | 54.58 ± 8.43 | 51.47–57.38 |
| 3 | 2,326 | SVM | 63.80 ± 10.33 | 60.13–67.38 |
| 6 | 1,147 | CNN | 57.24 ± 10.60 | 53.38–60.84 |
| 6 | 1,147 | SVM | 63.33 ± 15.76 | 57.59–68.65 |

These results support cross-participant generalization of value-related information under the study's task and stimulus set. The k = 3 and k = 6 estimates describe classification of same-rating averaged inputs.

![Cross-participant decoding](../figures/reproduced/decoding_performance.png)

## Within-participant decoding

A separate whole-brain SVM analysis leaves out one scanning session at a time. Mean BA is 62.41% across 30 participants. This analysis evaluates generalization across sessions within a participant.

## Result files

| File or directory | Contents |
|---|---|
| [decoding_performance.csv](../results/reference/tables/decoding_performance.csv) | Group BA, SD, confidence intervals, and input counts. |
| [decoding_participants.csv](../results/reference/tables/decoding_participants.csv) | Participant-level decoding metrics. |
| [decoding_folds.csv](../results/reference/tables/decoding_folds.csv) | Fold-level metrics. |
| [CNN predictions and summaries](../results/reference/decoding/cnn/) | Saved held-out predictions and CNN statistics. |
| [SVM predictions and summaries](../results/reference/decoding/svm/) | Saved held-out predictions and SVM statistics for each k. |
| [session_holdout_summary.csv](../results/reference/tables/session_holdout_summary.csv) | Within-participant, session-held-out decoding summary. |

To regenerate the tables, run `python scripts/reproduce_results.py` from the repository root. Means and SDs are recalculated from the saved predictions; confidence intervals are loaded from the saved analysis results. See [figure reproduction](README.md) for plotting commands and output files.
