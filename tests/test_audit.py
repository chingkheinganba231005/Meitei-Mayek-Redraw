import pytest

from redraw.audit import audit, expected_live, markdown, mismatches


@pytest.mark.parametrize("name, route", [("redraw_hdr_hr", "honest"), ("redraw_hdr", "coupled")])
def test_gradient_contract_matches_the_intended_routing(name, route):
    table = audit(name, tiny=True, batch=4, num_classes=55)
    assert mismatches(table, expected_live(route)) == []
    # the fit loss teaches only the true classes' templates; margin and discrimination teach the rivals too
    labels = set(range(4))
    assert set(table["fit"]["template_classes"]) <= labels
    assert set(table["margin"]["template_classes"]) - labels
    assert "| loss |" in markdown(table)


def test_classification_never_reaches_stroke_geometry_under_honest_routing():
    table = audit("rt_both", tiny=True, batch=2, num_classes=55)
    for loss in ("classification (fused)", "classification (stroke reader)"):
        assert table[loss]["decoder: queries and stroke outputs"]["status"] != "live"
        assert table[loss]["backbone"]["status"] == "live"
    assert table["drawing"]["backbone"]["status"] != "live"


ROUTED = ["redraw_hdr_hr", "redraw_hdr", "redraw_hdr_hr_g25", "rt_none", "rt_backbone", "rt_geometry", "rt_both",
          "rd_geometry", "rd_features", "rd_grid", "rd_free", "sequential", "control_m"]


@pytest.mark.parametrize("name", ROUTED)
def test_every_ablation_trains_what_its_routing_says(name):
    from redraw.audit import expected_for
    from redraw.model import CONFIGS
    table = audit(name, tiny=True, batch=2, num_classes=55)
    assert mismatches(table, expected_for(CONFIGS[name])) == []


def test_expected_table_agrees_with_the_paper_table():
    from redraw.audit import expected_for
    from redraw.model import CONFIGS
    assert expected_for(CONFIGS["redraw_hdr_hr"]) == expected_live("honest")
    assert expected_for(CONFIGS["redraw_hdr"]) == expected_live("coupled")


def test_factorial_corners_differ_only_where_intended():
    from redraw.audit import expected_for
    from redraw.model import CONFIGS
    e = {n: expected_for(CONFIGS[n]) for n in ("rt_none", "rt_backbone", "rt_geometry", "rt_both")}
    assert "backbone" in e["rt_none"]["drawing"] and "backbone" in e["rt_geometry"]["drawing"]
    assert "backbone" not in e["rt_backbone"]["drawing"] and "backbone" not in e["rt_both"]["drawing"]
    geometry = "decoder: queries and stroke outputs"
    assert geometry in e["rt_none"]["classification (stroke reader)"]
    assert geometry in e["rt_backbone"]["classification (stroke reader)"]
    assert geometry not in e["rt_geometry"]["classification (stroke reader)"]
    assert geometry not in e["rt_both"]["classification (stroke reader)"]


def test_frozen_backbone_learns_nothing():
    table = audit("sequential", tiny=True, batch=2, num_classes=55)
    for row in table.values():
        assert row["backbone"]["status"] == "none" and row["image classifier"]["status"] == "none"
    assert table["classification (stroke reader)"]["stroke reader"]["status"] == "live"
