import json
from pathlib import Path

import pytest

from redraw.paper_numbers import final_numbers, reduced_data_numbers, robustness_numbers

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def final():
    return json.loads((RESULTS / "final.json").read_text())


def test_headline_numbers_come_out_of_the_results_file(final):
    t = final_numbers(final)
    assert round(t["control"]["tta_logits"]["mean"], 2) == 97.83 and round(t["control"]["tta_logits"]["sd"], 2) == 0.07
    assert round(t["ours"]["tta_logits"]["mean"], 2) == 97.89 and round(t["ours"]["tta_logits"]["sd"], 2) == 0.06
    assert t["ours_minus_control_tta"]["images_per_seed"] == [18, 14, -9]
    assert t["ours"]["seed_average_tta"]["errors"] == 264 and t["control"]["seed_average_tta"]["errors"] == 271
    assert (t["ours"]["check_1view"]["wrong_to_right"], t["ours"]["check_1view"]["right_to_wrong"]) == (11, 14)


def test_paired_runs_with_ties(final):
    r = reduced_data_numbers(final)
    assert (r["all (12 pairs)"]["ahead"], r["all (12 pairs)"]["n"]) == (10, 12)
    assert r["all (12 pairs)"]["exact_wilcoxon_p_ties"] == pytest.approx(0.0083, abs=1e-4)
    assert r["reduced (9 pairs)"]["exact_wilcoxon_p_ties"] == pytest.approx(0.0273, abs=1e-4)
    assert r["full (3 pairs)"]["exact_wilcoxon_p_ties"] >= 0.25


def test_robustness_means(final):
    assert robustness_numbers(final)["mean_over_14_conditions"] == {"control": 94.11, "ours": 94.01, "coupled": 93.97}
