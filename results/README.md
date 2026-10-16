# Results files

Every number in the paper comes from these files (copies of `MyDrive/redraw/.../results.json` written
by the notebooks):

| file | written by | split |
|---|---|---|
| `round1.json` | `notebooks/round1.ipynb` | validation |
| `round2.json` | `notebooks/round2.ipynb` | validation |
| `diagnostics.json` | `notebooks/diagnostics.ipynb` | validation |
| `round3.json` | `notebooks/round3.ipynb` | validation |
| `final.json` | `notebooks/final.ipynb` | test (predicted once) |

Accuracies are fractions, not percentages.

`paper_numbers.json` holds every number of the paper that comes from these files, recomputed by
`python -m redraw.paper_numbers` (paired comparisons across training runs with the exact signed-rank
test, ties handled). The further experiments (`notebooks/rev*`) write their results to
`MyDrive/redraw/revision`; copies of their result archives go in `results/revision/`, including the
per-image test predictions of every final network (`per_image/test_predictions.csv`, and the final-output
probabilities in `per_image/probabilities/`).
