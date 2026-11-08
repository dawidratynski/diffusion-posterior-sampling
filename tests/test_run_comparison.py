"""The comparison driver's gating and cache-invalidation logic.

No generation runs here: these check WHICH models run, WHAT command each gets,
and WHEN a cached result is reused. Those are the decisions that silently
corrupt a comparison if wrong, since every run still produces plausible output.
"""
import csv
import importlib.util
import json
import math
import os
import sys
from types import SimpleNamespace

import pytest

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, os.path.abspath(_SCRIPTS))

SPEC = importlib.util.spec_from_file_location(
    "run_comparison", os.path.join(_SCRIPTS, "run_comparison.py"))
rc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rc)

CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")


def make_args(tmp_path, gan=False, wgan=False, rs_models="auto",
              stage="final"):
    paths = {}
    for name, present in (("gan", gan), ("wgan", wgan)):
        p = tmp_path / f"uvcgan_{name}"
        if present:
            p.mkdir(parents=True, exist_ok=True)
            # What uvcgan2 writes; the driver checks the loss matches the flag.
            (p / "config.json").write_text(
                json.dumps({"loss": "wgan" if name == "wgan" else "lsgan"}))
        paths[name] = str(p)
    return SimpleNamespace(
        uvcgan_gan_path=paths["gan"], uvcgan_wgan_path=paths["wgan"],
        rs_models=rs_models, stage=stage,
        data_root=str(tmp_path / "data"),
        model_config=os.path.join(CONFIGS, "crystal_model_config.yaml"),
        diffusion_config=os.path.join(CONFIGS, "crystal_diffusion_config.yaml"),
        task_config=os.path.join(CONFIGS, "crystal_cyclegan_config.yaml"),
        scale_dps_analytic=2.0, scale_dps_uvcgan_gan=1.0,
        scale_dps_uvcgan_wgan=1.5)


def labels(models):
    return [m["label"] for m in models]


def test_five_models_with_distinct_labels():
    assert labels(rc.MODELS) == [
        "uvcgan_gan", "uvcgan_wgan", "dps_uvcgan_gan", "dps_uvcgan_wgan",
        "dps_analytic"]


@pytest.mark.parametrize("gan,wgan,expected", [
    (False, False, ["dps_analytic"]),
    (True, False, ["uvcgan_gan", "dps_uvcgan_gan", "dps_analytic"]),
    (False, True, ["uvcgan_wgan", "dps_uvcgan_wgan", "dps_analytic"]),
    (True, True, ["uvcgan_gan", "uvcgan_wgan", "dps_uvcgan_gan",
                  "dps_uvcgan_wgan", "dps_analytic"]),
])
def test_each_uvcgan_gates_only_its_own_pair(tmp_path, gan, wgan, expected):
    usable, *_ = rc.available(make_args(tmp_path, gan, wgan))
    assert labels(usable) == expected


def test_empty_path_counts_as_absent(tmp_path):
    """The notebook passes "" for missing weights; that must not be truthy."""
    args = make_args(tmp_path)
    args.uvcgan_gan_path = ""
    usable, paths, *_ = rc.available(args)
    assert paths["gan"] is None
    assert labels(usable) == ["dps_analytic"]


def test_missing_measurement_builder_fails_early(tmp_path):
    args = make_args(tmp_path, gan=False, wgan=True, rs_models="uvcgan_gan")
    with pytest.raises(ValueError, match="GAN UVCGAN"):
        rc.available(args)


def test_every_dps_label_has_its_own_scale_flag():
    """stage_final looks scales up by label; a missing flag would crash late."""
    sys.argv = ["run_comparison.py", "--stage", "final", "--data_root", "d",
                "--model_config", "m", "--diffusion_config", "d",
                "--task_config", "t", "--out_dir", "o"]
    args = rc.parse_args()
    for label in rc.DPS_LABELS:
        assert hasattr(args, f"scale_{label}")


def capture_cmd(monkeypatch):
    seen = []

    def fake_run(cmd):
        # The real generator creates its out_dir; the driver relies on that.
        # (The evaluator command has none.)
        if flag(cmd, "--out_dir"):
            os.makedirs(flag(cmd, "--out_dir"), exist_ok=True)
        seen.append(cmd)

    monkeypatch.setattr(rc, "run", fake_run)
    return seen


def flag(cmd, name):
    return cmd[cmd.index(name) + 1] if name in cmd else None


def test_builder_and_model_get_their_own_weights(tmp_path, monkeypatch):
    """GAN builds y, WGAN is under test: the two paths must not be conflated."""
    args = make_args(tmp_path, gan=True, wgan=True, rs_models="uvcgan_gan")
    _, paths, *_ = rc.available(args)
    seen = capture_cmd(monkeypatch)
    model = next(m for m in rc.MODELS if m["label"] == "dps_uvcgan_wgan")
    rc.generate(model, str(tmp_path / "out"), args, paths, "roundtrip",
                "uvcgan_gan", 1.5, 2, 3)
    cmd = seen[0]
    assert flag(cmd, "--rs_framework") == "uvcgan2"
    assert flag(cmd, "--rs_path") == paths["gan"]
    assert flag(cmd, "--uvcgan_path") == paths["wgan"]
    assert flag(cmd, "--dps_framework") == "uvcgan2"
    assert flag(cmd, "--scale") == "1.5"


def test_analytic_gets_no_weights_in_synth_mode(tmp_path, monkeypatch):
    args = make_args(tmp_path, gan=True, wgan=True, rs_models="uvcgan_gan")
    _, paths, *_ = rc.available(args)
    seen = capture_cmd(monkeypatch)
    model = next(m for m in rc.MODELS if m["label"] == "dps_analytic")
    rc.generate(model, str(tmp_path / "out"), args, paths, "synth", None, 2.0,
                2, 3)
    cmd = seen[0]
    assert "--uvcgan_path" not in cmd
    assert "--rs_path" not in cmd and "--rs_framework" not in cmd


def sig(tmp_path, label, mode, rs=None, **kw):
    args = make_args(tmp_path, **kw)
    _, paths, *_ = rc.available(args)
    model = next(m for m in rc.MODELS if m["label"] == label)
    return rc.run_signature(model, args, paths, mode, rs, 2.0, 4, 50)


def test_weights_arriving_keeps_analytic_synth_cache(tmp_path):
    """Synth-mode dps_analytic depends on no UVCGAN; new weights must not
    force a 1.6 h regeneration of it."""
    before = sig(tmp_path / "a", "dps_analytic", "synth")
    after = sig(tmp_path / "b", "dps_analytic", "synth", gan=True, wgan=True)
    # tmp paths differ between the two calls but neither enters the signature.
    assert before == after


def test_changing_builder_invalidates_roundtrip(tmp_path):
    a = sig(tmp_path, "dps_analytic", "roundtrip", "uvcgan_gan",
            gan=True, wgan=True)
    b = sig(tmp_path, "dps_analytic", "roundtrip", "uvcgan_wgan",
            gan=True, wgan=True)
    assert a != b


@pytest.mark.parametrize("gan,wgan,expected", [
    (False, False, ["spectral"]),
    (True, False, ["uvcgan_gan"]),
    (False, True, ["uvcgan_wgan"]),
    (True, True, ["uvcgan_gan", "uvcgan_wgan"]),
])
def test_auto_builders_are_every_available_uvcgan(tmp_path, gan, wgan,
                                                   expected):
    _, _, rs_models = rc.available(make_args(tmp_path, gan, wgan))
    assert rs_models == expected


def test_swapped_gan_wgan_dirs_are_refused(tmp_path):
    """Both runs produce identically named uvcgan2 dirs; a swap must not run."""
    args = make_args(tmp_path, gan=True, wgan=True)
    gan_cfg = os.path.join(args.uvcgan_gan_path, "config.json")
    with open(gan_cfg, "w") as fh:
        json.dump({"loss": "wgan"}, fh)
    with pytest.raises(ValueError, match="swapped"):
        rc.available(args)


def test_dir_without_config_is_refused(tmp_path):
    """E.g. the uvcgan2 training outdir instead of the model dir inside it."""
    args = make_args(tmp_path, gan=True)
    os.remove(os.path.join(args.uvcgan_gan_path, "config.json"))
    with pytest.raises(FileNotFoundError, match="config.json"):
        rc.available(args)


def test_sweep_needs_no_builder_weights(tmp_path):
    """The sweep runs on synth inputs, so a missing builder must not block it."""
    args = make_args(tmp_path, gan=False, wgan=True, rs_models="uvcgan_gan",
                     stage="sweep")
    usable, *_ = rc.available(args)
    assert "dps_uvcgan_wgan" in [m["label"] for m in usable]


# ------------------------------------------------------- scale selection ----

def row(scale, sig=0.01, kid=0.02, kid_std=0.002, nn=1.13, floor=1.11,
        prom=1200.0, real_prom=1500.0, div=0.2, angle=0.5):
    """A sweep-table row. `sig` is the median SPACING error (the label-validity
    measure the rule uses; the signature error is deliberately not used)."""
    return {"scale": str(scale), "spacing_err_median": str(sig), "angle_err_median": str(angle),
            "signature_err_median": "0.9",   # ignored by the rule on purpose
            "kid": "" if kid is None else str(kid),
            "kid_std": "" if kid_std is None else str(kid_std),
            "nn_dist_median": str(nn), "nn_floor": str(floor),
            "prominence_median": str(prom),
            "real_prominence_median": str(real_prom),
            "diversity_rmse": str(div)}


def test_lowest_kid_wins_when_clearly_better():
    pick = rc.select_scale([row(0.3, kid=0.05), row(1, kid=0.02),
                            row(3, kid=0.04)])
    assert pick["scale"] == 1.0
    assert pick["warnings"] == []


def test_inadmissible_scale_never_chosen_despite_best_kid():
    """Label validity is decisive: drifted lattices = mislabelled data."""
    pick = rc.select_scale([row(0.3, sig=0.20, kid=0.001), row(1, kid=0.02),
                            row(3, kid=0.03)])
    assert pick["scale"] == 1.0


def test_wrong_orientation_is_inadmissible():
    """Right spacing at the wrong angle is still a mislabelled lattice."""
    pick = rc.select_scale([row(0.3, kid=0.001, angle=30.0), row(1, kid=0.02),
                            row(3, kid=0.03)])
    assert pick["scale"] == 1.0


def test_signature_error_does_not_block_admissibility():
    """Realistic outputs score signature ~0.3-0.6 with the lattice intact
    (harmonics blurred away), so the rule must not use it."""
    pick = rc.select_scale([row(0.3, kid=0.03), row(1, kid=0.01),
                            row(3, kid=0.04)])
    assert pick["scale"] == 1.0 and not any("NO scale" in w
                                            for w in pick["warnings"])


def test_memorising_scale_never_chosen():
    pick = rc.select_scale([row(0.3, kid=0.03), row(1, kid=0.001, nn=1.0),
                            row(3, kid=0.04)])
    assert pick["scale"] == 0.3


def test_kid_ties_within_noise_go_to_better_label_validity():
    pick = rc.select_scale([row(0.3, kid=0.020, sig=0.030),
                            row(1, kid=0.021, sig=0.010),
                            row(3, kid=0.040, sig=0.001)])
    # 3 has the best signature error but is clearly worse on KID.
    assert pick["scale"] == 1.0


def test_no_kid_falls_back_to_prominence():
    pick = rc.select_scale([row(0.3, kid=None, prom=500),
                            row(1, kid=None, prom=1400),
                            row(3, kid=None, prom=4000)])
    assert pick["scale"] == 1.0


def test_none_admissible_falls_back_with_warning():
    pick = rc.select_scale([row(0.3, sig=0.3), row(1, sig=0.1),
                            row(3, sig=0.2)])
    assert pick["scale"] == 1.0
    assert any("NO scale" in w for w in pick["warnings"])


def test_edge_and_collapse_warnings():
    pick = rc.select_scale([row(0.3, kid=0.05), row(1, kid=0.04),
                            row(3, kid=0.01, div=0.01)])
    assert pick["scale"] == 3.0
    assert any("edge" in w for w in pick["warnings"])
    assert any("collapsed" in w for w in pick["warnings"])


@pytest.mark.parametrize("chosen,expected", [
    (1, [0.548, 1.73]),     # interior: geometric midpoints
    (10, [5.48, 33.3]),     # top edge: midpoint below, one step beyond
    (0.1, [0.0333, 0.173]),  # bottom edge
])
def test_refine_scales(chosen, expected):
    assert rc.refine_scales([0.1, 0.3, 1, 3, 10], chosen) == expected


def test_sweep_uses_synth_train_and_picks_per_model(tmp_path, monkeypatch):
    """End to end through stage_sweep with generation and evaluation faked:
    inputs come from synth/train, every DPS model gets its own choice, and an
    earlier choice for a model not swept this time is kept."""
    args = make_args(tmp_path, gan=True, wgan=False, stage="sweep")
    args.out_dir = str(tmp_path / "sweep")
    args.scales, args.refine = "0.3,1,3", True
    args.sweep_limit, args.sweep_samples = 3, 2
    args.fid_kid, args.lpips = True, False
    os.makedirs(args.out_dir)
    with open(os.path.join(args.out_dir, "chosen_scales.json"), "w") as fh:
        json.dump({"dps_uvcgan_wgan": {"scale": 7.0}}, fh)

    best = {"dps_analytic": 3.0, "dps_uvcgan_gan": 0.3}
    gens = []

    def fake_run(cmd):
        if cmd[0].endswith("generate_augmented.py"):
            os.makedirs(flag(cmd, "--out_dir"), exist_ok=True)
            gens.append(cmd)
            return
        # evaluate.py: write a summary where KID is lowest at the model's best.
        dirs = [cmd[i + 1] for i, c in enumerate(cmd) if c == "--result_dir"]
        label = os.path.basename(os.path.dirname(dirs[0]))
        rows = []
        for d in dirs:
            s = float(os.path.basename(d).removeprefix("scale"))
            rows.append(row(s, kid=0.01 + abs(math.log(s / best[label]))))
        with open(flag(cmd, "--out_summary_csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    monkeypatch.setattr(rc, "run", fake_run)
    rc.stage_sweep(args)

    assert all(flag(c, "--synth_root").endswith(os.path.join("synth", "train"))
               for c in gens)
    assert all("--rs_framework" not in c for c in gens)
    with open(os.path.join(args.out_dir, "chosen_scales.json")) as fh:
        chosen = json.load(fh)
    assert chosen["dps_analytic"]["scale"] == 3.0
    assert chosen["dps_uvcgan_gan"]["scale"] == 0.3
    assert chosen["dps_uvcgan_wgan"] == {"scale": 7.0}   # kept, not erased
    # Edge winners got a step beyond the grid in the refinement round.
    assert 9.0 in chosen["dps_analytic"]["scales_tried"]


def test_unknown_builder_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        rc.available(make_args(tmp_path, rs_models="uvcgan_gna"))


def test_final_runs_one_roundtrip_per_builder_then_synth(tmp_path,
                                                         monkeypatch):
    args = make_args(tmp_path, gan=True, wgan=True)
    args.out_dir = str(tmp_path / "final")
    args.samples, args.limit = 2, 3
    args.no_figures, args.fid_kid, args.lpips = True, False, False
    seen = capture_cmd(monkeypatch)
    rc.stage_final(args)
    gens = [c for c in seen if c[0].endswith("generate_augmented.py")]
    by_exp = {}
    for c in gens:
        exp = os.path.basename(os.path.dirname(flag(c, "--out_dir")))
        by_exp.setdefault(exp, []).append(c)
    assert sorted(by_exp) == ["roundtrip_uvcgan_gan", "roundtrip_uvcgan_wgan",
                              "synth"]
    for exp, cmds in by_exp.items():
        assert len(cmds) == 5, exp
    # Within one round trip every model sees the same builder.
    for c in by_exp["roundtrip_uvcgan_wgan"]:
        assert flag(c, "--rs_path").endswith("uvcgan_wgan")


# --------------------------------------------- cache validity and resume ----

def fake_generator(monkeypatch, die_after=None):
    """Stands in for generate_augmented.py: writes manifest rows + images,
    honouring --resume, optionally 'disconnecting' after N new samples."""
    calls = []

    def fake_run(cmd):
        if not cmd[0].endswith("generate_augmented.py"):
            return
        out = flag(cmd, "--out_dir")
        limit = int(flag(cmd, "--limit"))
        k_max = int(flag(cmd, "--samples_per_input") or 1)
        resume = "--resume" in cmd
        calls.append({"resume": resume})
        os.makedirs(os.path.join(out, "generated"), exist_ok=True)
        man = os.path.join(out, "manifest.csv")
        rows = []
        if resume and os.path.exists(man):
            with open(man) as fh:
                rows = list(csv.DictReader(fh))
        done = {r["output_image"] for r in rows}
        new = 0
        for n in range(limit):
            for k in range(k_max):
                f = f"{n:05d}_{k:02d}.png"
                if f in done:
                    continue
                if die_after is not None and new >= die_after:
                    _write(man, rows)
                    raise RuntimeError("disconnected")
                open(os.path.join(out, "generated", f), "wb").close()
                rows.append({"output_image": f})
                new += 1
        _write(man, rows)

    monkeypatch.setattr(rc, "run", fake_run)
    return calls


def _write(man, rows):
    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["output_image"])
        w.writeheader()
        w.writerows(rows)


def _gen(tmp_path, args, paths, label="dps_analytic"):
    model = next(m for m in rc.MODELS if m["label"] == label)
    return rc.generate(model, str(tmp_path / "out"), args, paths, "synth",
                       None, 2.0, 2, 3)


def _setup(tmp_path, **kw):
    args = make_args(tmp_path, **kw)
    _, paths, _ = rc.available(args)
    return args, paths


def test_rerun_of_complete_run_is_cached(tmp_path, monkeypatch):
    args, paths = _setup(tmp_path)
    calls = fake_generator(monkeypatch)
    _gen(tmp_path, args, paths)
    _gen(tmp_path, args, paths)
    assert len(calls) == 1


def test_interrupted_run_resumes_instead_of_restarting(tmp_path, monkeypatch):
    args, paths = _setup(tmp_path)
    fake_generator(monkeypatch, die_after=4)
    with pytest.raises(RuntimeError):
        _gen(tmp_path, args, paths)
    calls = fake_generator(monkeypatch)
    _gen(tmp_path, args, paths)
    assert calls == [{"resume": True}]
    with open(tmp_path / "out" / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6 and len({r["output_image"] for r in rows}) == 6
    assert (tmp_path / "out" / "run_config.json").exists()
    assert not (tmp_path / "out" / "run_config.partial.json").exists()


def test_changed_settings_restart_from_scratch_not_resume(tmp_path, monkeypatch):
    args, paths = _setup(tmp_path)
    fake_generator(monkeypatch, die_after=4)
    with pytest.raises(RuntimeError):
        _gen(tmp_path, args, paths)
    calls = fake_generator(monkeypatch)
    model = next(m for m in rc.MODELS if m["label"] == "dps_analytic")
    rc.generate(model, str(tmp_path / "out"), args, paths, "synth", None,
                3.0, 2, 3)                       # different scale
    assert calls == [{"resume": False}]


def test_comment_only_config_edit_keeps_cache(tmp_path, monkeypatch):
    args, paths = _setup(tmp_path)
    cfg = tmp_path / "task.yaml"
    with open(args.task_config) as src:
        cfg.write_text(src.read())
    args.task_config = str(cfg)
    calls = fake_generator(monkeypatch)
    _gen(tmp_path, args, paths)
    cfg.write_text(cfg.read_text() + "\n# just a comment\n")
    _gen(tmp_path, args, paths)
    assert len(calls) == 1


def test_weights_replaced_in_place_invalidate_cache(tmp_path, monkeypatch):
    args, paths = _setup(tmp_path, gan=True)
    w = os.path.join(paths["gan"], "net_avg_gen_ba.pth")
    with open(w, "wb") as fh:
        fh.write(b"early checkpoint")
    calls = fake_generator(monkeypatch)
    _gen(tmp_path, args, paths, "dps_uvcgan_gan")
    with open(w, "wb") as fh:
        fh.write(b"final weights, different size")
    _gen(tmp_path, args, paths, "dps_uvcgan_gan")
    assert len(calls) == 2


def test_missing_image_invalidates_cache(tmp_path, monkeypatch):
    args, paths = _setup(tmp_path)
    calls = fake_generator(monkeypatch)
    _gen(tmp_path, args, paths)
    os.remove(tmp_path / "out" / "generated" / "00001_01.png")
    _gen(tmp_path, args, paths)
    assert len(calls) == 2


def test_evaluation_skipped_when_inputs_unchanged(tmp_path, monkeypatch):
    args, paths = _setup(tmp_path)
    args.fid_kid, args.lpips = True, False
    fake_generator(monkeypatch)
    _gen(tmp_path, args, paths)
    evals = []

    def fake_eval(cmd):
        evals.append(flag(cmd, "--reference_half"))
        for key in ("--out_csv", "--out_summary_csv"):
            open(flag(cmd, key), "w").close()

    monkeypatch.setattr(rc, "run", fake_eval)
    out = str(tmp_path / "out")
    for _ in range(2):
        rc.evaluate([out], args, str(tmp_path / "p.csv"),
                    str(tmp_path / "s.csv"), paired=False, half="B")
    assert evals == ["B"]



def test_dps_uvcgan_runs_from_the_gray_projection_era_regenerate(tmp_path):
    """Only DPS-through-UVCGAN runs are marked; others keep their caches."""
    args, paths = _setup(tmp_path, gan=True)
    sig = lambda label: rc.run_signature(
        next(m for m in rc.MODELS if m["label"] == label), args, paths,
        "synth", None, 1.0, 2, 3)
    assert sig("dps_uvcgan_gan")["operator_input"] == "unmodified"
    assert "operator_input" not in sig("dps_analytic")
    assert "operator_input" not in sig("uvcgan_gan")


def test_analytic_operator_runs_carry_its_version(tmp_path):
    """Changing the analytic operator must invalidate its cached runs --
    whether it is the DPS operator or the round-trip measurement builder."""
    args, paths = _setup(tmp_path, gan=True)

    def sig(label, mode, rs=None):
        m = next(x for x in rc.MODELS if x["label"] == label)
        return rc.run_signature(m, args, paths, mode, rs, 1.0, 2, 3)

    v = rc.SPECTRAL_OPERATOR_VERSION
    assert sig("dps_analytic", "synth")["spectral_operator"] == v
    assert sig("uvcgan_gan", "roundtrip", "spectral")["spectral_operator"] == v
    assert "spectral_operator" not in sig("dps_uvcgan_gan", "synth")
    assert "spectral_operator" not in sig("uvcgan_gan", "roundtrip",
                                          "uvcgan_gan")
