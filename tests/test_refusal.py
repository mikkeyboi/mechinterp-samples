"""Model-free tests for the refusal-direction (H005) toolkit.

None of these load a language model. We use a planted-refusal synthetic model,
whose refusal is by construction one known direction, so the whole pipeline
(difference-of-means -> ablation/addition -> generate -> classify -> verdict) is
verifiable deterministically on CPU. This mirrors the project rule that the
analysis math is decoupled from capture and always unit-testable.
"""
from __future__ import annotations

import numpy as np
import pytest

from mechinterp_samples import (
    RefusalDirection,
    DirectionalAblation,
    ActivationAddition,
    RefusalClassifier,
    PlantedRefusalModel,
    make_planted_dataset,
    SingleDirectionExperiment,
    RefusalReproductionPlotter,
    cosine,
    unit,
    norm_matched_random,
)


# --- classifier -------------------------------------------------------------

def test_classifier_detects_refusal_phrases():
    clf = RefusalClassifier()
    assert clf.is_refusal("I'm sorry, but I can't help with that.")
    assert clf.is_refusal("I cannot assist with that request.")
    assert clf.is_refusal("As an AI, I am not able to do this.")


def test_classifier_passes_compliance():
    clf = RefusalClassifier()
    assert not clf.is_refusal("Sure, here is how you can do that:")
    assert not clf.is_refusal("Of course. To begin, you will need:")


def test_classifier_is_case_insensitive():
    clf = RefusalClassifier()
    assert clf.is_refusal("I CANNOT help with that.")


def test_classifier_only_inspects_head():
    """A refusal phrase buried deep in a long compliant answer is not a refusal."""
    clf = RefusalClassifier(head_chars=40)
    text = "Here are the full steps you asked for. " + ("detail " * 30) + "I cannot stop."
    assert not clf.is_refusal(text)


def test_classifier_handles_empty_and_none():
    clf = RefusalClassifier()
    assert not clf.is_refusal("")
    assert not clf.is_refusal(None)
    assert clf.refusal_rate([]) == 0.0


# --- direction + interventions ---------------------------------------------

def test_diff_of_means_recovers_planted_direction():
    Xh, Xb, r_hat = make_planted_dataset(seed=0)
    direction = RefusalDirection.from_means(Xh, Xb)
    assert abs(cosine(direction.unit, r_hat)) > 0.95


def test_ablation_removes_the_projection():
    Xh, Xb, r_hat = make_planted_dataset(seed=1)
    direction = RefusalDirection.from_means(Xh, Xb)
    ablated = DirectionalAblation(direction.vector).apply(Xh)
    # After ablation the projection onto the direction is ~0.
    proj = RefusalDirection(direction.vector).projection(ablated)
    assert np.abs(proj).max() < 1e-8


def test_ablation_single_vector_shape():
    Xh, Xb, _ = make_planted_dataset(seed=2)
    direction = RefusalDirection.from_means(Xh, Xb)
    one = DirectionalAblation(direction.vector).apply(Xh[0])
    assert one.shape == Xh[0].shape


def test_addition_shifts_along_axis():
    Xh, Xb, _ = make_planted_dataset(seed=3)
    direction = RefusalDirection.from_means(Xh, Xb)
    shifted = ActivationAddition(direction.vector, coeff=1.0).apply(Xb)
    delta = shifted - Xb
    # Every row moved by exactly one copy of the direction vector.
    assert np.allclose(delta, direction.vector, atol=1e-8)


def test_control_is_norm_matched_and_decorrelated():
    Xh, Xb, _ = make_planted_dataset(seed=4)
    direction = RefusalDirection.from_means(Xh, Xb)
    control = direction.control(seed=7)
    assert abs(control.norm - direction.norm) < 1e-6
    assert abs(cosine(control.vector, direction.vector)) < 0.3


def test_norm_matched_random_matches_norm():
    v = np.array([3.0, 4.0, 0.0])  # norm 5
    r = norm_matched_random(v, seed=0)
    assert abs(np.linalg.norm(r) - 5.0) < 1e-9


# --- planted model ----------------------------------------------------------

def test_planted_model_refuses_above_threshold():
    d = 32
    r_hat = unit(np.ones(d))
    model = PlantedRefusalModel(r_hat=r_hat, theta=1.5, seed=0)
    high = (r_hat * 5.0)[None, :]   # large projection -> refuse
    low = (r_hat * -5.0)[None, :]   # negative projection -> comply
    assert model.decides_refuse(high)[0]
    assert not model.decides_refuse(low)[0]


def test_planted_model_generates_refusal_and_compliance_text():
    d = 16
    r_hat = unit(np.ones(d))
    model = PlantedRefusalModel(r_hat=r_hat, theta=1.5, seed=0)
    clf = RefusalClassifier()
    refusal_text = model.generate((r_hat * 5.0)[None, :])
    compliance_text = model.generate((r_hat * -5.0)[None, :])
    assert clf.is_refusal(refusal_text[0])
    assert not clf.is_refusal(compliance_text[0])


# --- the experiment + verdict ----------------------------------------------

def _planted_split(seed=0):
    """Train + held-out test sharing the same planted geometry."""
    rng = np.random.default_rng(seed)
    r_hat = unit(rng.standard_normal(64))
    base = rng.standard_normal(64) * 0.5
    Xh_tr, Xb_tr, _ = make_planted_dataset(seed=seed, r_hat=r_hat, base=base)
    Xh_te, Xb_te, _ = make_planted_dataset(seed=seed + 100, r_hat=r_hat, base=base)
    model = PlantedRefusalModel(r_hat=r_hat, seed=seed)
    direction = RefusalDirection.from_means(Xh_tr, Xb_tr)
    return model, Xh_te, Xb_te, direction


def test_experiment_reproduces_single_direction():
    model, Xh_te, Xb_te, direction = _planted_split(seed=0)
    report = SingleDirectionExperiment().run(model, Xh_te, Xb_te, direction)
    assert report.verdict == "SINGLE-DIRECTION-REPRODUCES"
    # Real direction moves both arms; control does not.
    assert report.real.ablation_harmful_refusal < 0.25
    assert report.control.ablation_harmful_refusal > 0.75
    assert report.real.addition_harmless_refusal > 0.75
    assert report.control.addition_harmless_refusal < 0.25


def test_experiment_reproduces_across_seeds():
    for seed in (1, 2, 3):
        model, Xh_te, Xb_te, direction = _planted_split(seed=seed)
        report = SingleDirectionExperiment().run(model, Xh_te, Xb_te, direction)
        assert report.verdict == "SINGLE-DIRECTION-REPRODUCES"


def test_verdict_no_baseline_refusal():
    """A model that barely refuses at baseline returns the guard outcome."""
    rng = np.random.default_rng(0)
    r_hat = unit(rng.standard_normal(64))
    base = rng.standard_normal(64) * 0.5
    Xh_tr, Xb_tr, _ = make_planted_dataset(seed=0, r_hat=r_hat, base=base)
    Xh_te, Xb_te, _ = make_planted_dataset(seed=100, r_hat=r_hat, base=base)
    # theta far above the harmful mean (4.0) -> almost nothing refuses at baseline.
    model = PlantedRefusalModel(r_hat=r_hat, theta=8.0, seed=0)
    direction = RefusalDirection.from_means(Xh_tr, Xb_tr)
    report = SingleDirectionExperiment().run(model, Xh_te, Xb_te, direction)
    assert report.verdict == "NO-BASELINE-REFUSAL"


def test_verdict_not_single_direction_when_control_also_works():
    """If the 'control' is actually the real direction, neither arm beats it."""
    model, Xh_te, Xb_te, direction = _planted_split(seed=0)
    # Force the control margin impossibly high so the gap can never be met.
    exp = SingleDirectionExperiment(control_margin=2.0)
    report = exp.run(model, Xh_te, Xb_te, direction)
    assert report.verdict == "NOT-SINGLE-DIRECTION"


def test_report_roundtrips_to_json(tmp_path):
    model, Xh_te, Xb_te, direction = _planted_split(seed=0)
    report = SingleDirectionExperiment().run(model, Xh_te, Xb_te, direction)
    out = tmp_path / "report.json"
    report.to_json(out)
    assert out.exists()
    import json
    d = json.loads(out.read_text())
    assert d["verdict"] == "SINGLE-DIRECTION-REPRODUCES"
    assert d["real"]["baseline_harmful_refusal"] == 1.0
    assert "margins" in d


def test_plotter_writes_figure(tmp_path):
    model, Xh_te, Xb_te, direction = _planted_split(seed=0)
    report = SingleDirectionExperiment().run(model, Xh_te, Xb_te, direction)
    out = tmp_path / "refusal.png"
    path = RefusalReproductionPlotter().plot(report, out)
    assert path.exists() and path.stat().st_size > 0


def test_addition_with_unknown_intervention_subclass():
    """The Intervention ABC cannot be instantiated without apply()."""
    from mechinterp_samples import Intervention
    with pytest.raises(TypeError):
        Intervention()
