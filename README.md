# Jev calibration testbed

TypeSafe AI's **Jev** (released 2026-09-15) is a non-autoregressive "System
One" model that returns typed decisions (`noul`/`choice`/`score`) with
probabilities in a single parallel pass, pitched on calibration: *"higher
confidence means higher accuracy."* TypeSafe's own published evals report
accuracy, cost, and latency — no ECE, no reliability diagram, no Brier
score. Independent tests (days old as of this writing) show real
in-distribution calibration and real out-of-distribution degradation,
including one item type that is provably unanswerable from its text where
Jev was confident anyway.

This repo builds **one synthetic data-generating process with an exact,
closed-form posterior** (naive-Bayes emission tables, not estimated), used
twice:

1. **`toy/`** — a small encoder-based "System One" model, trained on an M1
   Max, that reproduces Jev's architectural shape (one encoder pass,
   parallel per-question heads) and the soft-target training bet that
   plausibly explains Jev's own calibration story.
2. **`probe/`** — six pre-registered test suites (S1-S6) against the real
   Jev API, graded against the generator's *known true posterior* rather
   than noisy observed labels — a sharper instrument than any published
   test, including items whose correct answer is provably the prior.

Full rationale: [`/Users/steve/.claude/plans/hi-i-m-curious-about-cosmic-pelican.md`](file:///Users/steve/.claude/plans/hi-i-m-curious-about-cosmic-pelican.md).
Math and transformer background: [`jev/docs/explainers.md`](jev/docs/explainers.md).

## Quick start

```bash
uv sync --dev              # installs everything, including torch (MPS-enabled on Apple Silicon)
uv run pytest tests/ -q    # 80+ tests: generator ground truth, metric oracle/detector checks,
                            # model structure, client/suite parsing, negative controls

# Train the toy model (soft-target vs hard-label ablation, ~5 min/run on an M1 Max)
uv run python scripts/run_toy_experiment.py

# Probe the real Jev API (needs JEV_API_KEY; see probe/client.py)
uv run python -m jev.probe.run_probe --backend live --suites S1,S2,S3,S4,S5,S6 --n 200
```

## Layout

```
jev/
  dgp/
    slots.py       # emission tables: P(surface realization | label) -- the only place
                    # "ground truth" is authored; everything downstream derives from these
    generator.py    # exact Bayes posterior arithmetic, sampling, rendering to text
    scenarios.py     # family builders (resolvable/ambiguous/unanswerable), compositional
                      # train/test splits, negative controls (label permutation, paraphrase
                      # leakage classifier)
  eval/
    metrics.py       # ECE + noise floor, Brier/Murphy decomposition, temperature scaling,
                      # posterior KL, risk-coverage/AURC, bootstrap & Wilson CIs
    report.py         # reliability diagrams, risk-coverage plots
  toy/
    model.py           # encoder + per-concept cross-attention heads + ordinal head
    data.py              # Item -> tokenized tensors with BOTH soft and hard targets
    train.py              # soft-vs-hard training loop, evaluation
  probe/
    client.py             # Jev API client: live / cache-replay / hand-written fixtures
    suites.py               # S1-S6, each with a pre-registered hypothesis
    run_probe.py             # CLI
    fixtures/                 # hand-written example request/response pairs (schema dev only)
  docs/
    explainers.md              # calibration math + transformer background, with LaTeX
scripts/
  run_toy_experiment.py         # the actual soft-vs-hard experiment + report
tests/                           # 80+ tests -- see "Verification" below
outputs/                          # generated: results.json, reliability diagrams, logs
```

## The core trick

`dgp/slots.py` defines, by hand, tables of the form `P(surface phrase |
label)`. Because generation runs label → text through these tables (never
the other way), Bayes' rule inverts them *exactly*:

```
P(billing | "I can't access my account") = P(phrase|billing)*P(billing) / Z
```

No annotator, no LLM judge, no estimation error — the training target and
the evaluation ground truth are the same closed-form number. Three
overlap regimes give the three item families used throughout:

| Family | Emission overlap | True posterior |
|---|---|---|
| resolvable | disjoint | one-hot |
| ambiguous | partial (tunable) | any value you dial in, e.g. exactly 0.7 |
| unanswerable | zero slots (no textual trace at all) | exactly the prior |

See `docs/explainers.md` for the derivation and the negative controls that
guard it (label-independent paraphrasing, a leakage classifier, an exact
Bayes-error-rate ceiling no model may cross).

## Verification (see the plan for full ordering/rationale)

1. Generator ground truth validated by Monte Carlo (`tests/test_generator.py`).
2. Metric oracle/detector checks: a perfectly-calibrated source reads as
   calibrated; a *known* miscalibration is recovered with the right sign and
   magnitude (`tests/test_metrics.py`).
3. Toy model structural correctness (ordinal head validity, forward-pass
   shapes) (`tests/test_toy_model.py`).
4. Client/suite parsing verified against controlled test doubles with
   injected, known calibration properties (`tests/test_client.py`,
   `tests/test_suites.py`) — never against real Jev output.
5. `scripts/run_toy_experiment.py` runs the actual soft-vs-hard ablation and
   a shuffled-label negative control that must land at chance.

## Status

- Generator, metrics, toy model, and probe harness: built and tested (82
  tests, `uv run pytest tests/ -q`).
- Toy soft-vs-hard experiment: complete. Full numbers in
  `outputs/toy_experiment/results.json`; log in `outputs/toy_experiment_run.log`.
- Live Jev probe: not yet run (no API key at time of writing) — the harness
  is offline-first (fixture/replay backends) precisely so it's ready to run
  the moment one exists; reference cost for a comparable published run was
  ~\$0.06 for 4,621 calls.

### Toy experiment: what the soft-vs-hard ablation actually showed

30k generated items, compositional 80/20 split on `category` (test items use
slot-value combinations never seen in training), `all-MiniLM-L6-v2` encoder,
3 epochs, identical architecture/optimizer/schedule/seed between the two
runs — only the training target differs (see `docs/explainers.md`,
"soft targets vs. label smoothing").

**Resolution is shared; reliability is not.** On the `category` head
(held-out split), soft and hard training reach essentially the same
accuracy (0.714 vs 0.713) and an *overlapping* risk-coverage curve
(AURC 0.081 for both — see `risk_coverage_category.png`): the two models
rank confident-vs-unsure items identically well. But mean posterior KL
divergence from the true label distribution is **25x lower for soft
training** (0.0002 vs 0.0050), and on the deliberately-ambiguous item family
built via `build_ambiguous` — where the true posterior is genuinely
non-degenerate, not near one-hot — the gap is the same order (0.0004 vs
0.0101, again ~25x) while accuracy is identical (0.498 vs 0.498). This is
the Brier decomposition's reliability/resolution split
(`docs/explainers.md`) showing up empirically, not just as theory: the same
architecture, same accuracy, same *ranking* of confident vs. unsure
predictions — but the hard-label model's stated probabilities are
substantially further from the truth. (One honest wrinkle that is itself a
good illustration of the material: the two reliability-diagram PNGs report
ECE values that *disagree* with the table above, because the plotting call
used different bin settings — 10 equal-mass bins there vs. 15 equal-width
bins in `evaluate()`. That's ECE's known bin-sensitivity, from the
explainer, showing up unprompted in this project's own output. The
posterior-KL comparison is binning-independent and is the one to trust.)

**The unanswerable concept (`hidden_priority`) was solved correctly by
both.** True posterior is exactly the prior everywhere (Bayes-optimal
accuracy 0.400). Both soft- and hard-trained models land at accuracy
0.386–0.408 and mean KL ≤0.002 across every split — both learned to report
close to the prior on a question the text cannot answer, essentially
matching the Bayes-optimal ceiling. This makes sense in hindsight (with
enough data, even hard-label NLL is minimized by predicting the class
marginal when there's truly no signal — see the explainer) and it's a
useful negative result: it suggests the real Jev's published failure on an
analogous unanswerable question (44.7% accuracy, 0.74 mean confidence) is
*not* an inevitable property of this architecture/loss family under clean
training — something about Jev's actual training distribution or objective
is producing that overconfidence, not the mere fact of being a fast, non-
autoregressive decision model. Suite S3 is built to test this directly
against the real API.

**Honest limitation: the `urgency` head underfit.** Its accuracy (0.46–0.48)
sits well below its Bayes-optimal ceiling (0.56–0.60), and mean KL (~0.20)
is far higher than category's — the ordinal cumulative-link head likely
needs more capacity, more epochs, or a less restrictive parameterization
than a single latent utility can offer for this posterior shape. Soft vs.
hard show no clear gap on this head; both are too far from converged for a
target-choice effect to show through the noise. Worth revisiting before
trusting any urgency-specific conclusion — flagged rather than glossed over.

**Shuffled-label negative control passed.** Accuracy 0.253 against a
uniform-random baseline of 0.25, with reliability ≈0 — the pipeline learns
nothing from permuted labels, as it must.
