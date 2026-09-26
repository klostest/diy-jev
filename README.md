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
   than noisy observed labels — including items whose correct answer is
   provably the prior. See "Prior art" below: several independent projects
   test Jev against synthetic ground truth now; this one's specific edge is
   a *tunable, exact* posterior plus a provably-unanswerable control, tested
   against the real hosted API, which as of this writing nothing else combines.

Full rationale: [`/Users/steve/.claude/plans/hi-i-m-curious-about-cosmic-pelican.md`](file:///Users/steve/.claude/plans/hi-i-m-curious-about-cosmic-pelican.md).
Math and transformer background: [`jev/docs/explainers.md`](jev/docs/explainers.md).
How the pieces connect end to end, with a worked example: [`EXPERIMENTAL_SETUP.md`](EXPERIMENTAL_SETUP.md).

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

## Prior art

Jev launched 2026-09-15. This project started 2026-09-20 (first commit). By
2026-09-25 the ecosystem had grown fast enough that "test Jev against
synthetic ground truth" is no longer a novel idea on its own — worth saying
plainly rather than repeating TypeSafe's own habit of overclaiming ("nobody
has published that"). Closest work found, checked directly rather than
assumed:

- **[jev-from-scratch](https://github.com/Maverick-Ansh/jev-from-scratch)**
  is the closest match: a synthetic world with an exact, enumerable
  naive-Bayes posterior, a toy non-autoregressive model trained against it,
  and a soft-vs-improper-objective ablation with the same qualitative
  finding as this repo's toy experiment (an improper training signal
  degrades calibration while accuracy holds). Two confirmed differences:
  it never calls the real hosted Jev API (their own README: "a rebuild...
  trained on a synthetic world," tested only against their own
  reimplementation), and none of its questions are *independent* of the
  rendered text by construction — no unanswerable-by-design control, no
  analog of this repo's `HIDDEN_PRIORITY` result. (For what it's worth, it
  also doesn't appear to have gotten much traction — no citations found in
  any of the roundup articles or discussion threads checked, unlike e.g.
  JevBench or poorjev.)
- **[jev-calibration-probe](https://github.com/heyimMarc/jev-calibration-probe)**
  does test the real API, with ground truth "correct by construction" (does
  a code diff weaken a test suite) — but binary/deterministic, not a
  tunable probability, and `noul` only.
- **[jev-exploration](https://github.com/SamuelSacco/jev-exploration)** asks
  the same question this project's `HIDDEN_PRIORITY`/S3 suite answers
  ("does calibration survive difficulty, or only track accuracy?") but is
  proposal-stage, proposing human-labeled difficulty tiers rather than a
  generator with a known posterior.
- Everything else found (**poorjev**, **JevBench**, and several other
  from-scratch reimplementations) uses real or hand-labeled ground truth,
  or composite scoring — not exact posteriors.

None of this changes what's in `outputs/probe_results.json` — it changes how
the results should be framed: as a sharper instrument on one specific,
checkable axis (exact tunable posterior + provably-unanswerable control +
the real API, together), not as the first calibration test of Jev.

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

- Generator, metrics, toy model, and probe harness: built and tested (86
  tests, `uv run pytest tests/ -q`).
- Toy soft-vs-hard experiment: complete, **3 seeds/mode** (100/101/102), one
  fixed data pool/split. Per-seed data in `outputs/toy_experiment/results.json`,
  seed-aggregated mean±std in `outputs/toy_experiment/summary.json`; log in
  `outputs/toy_experiment_run.log`.
- Live Jev probe: complete, 2026-09-26, via OpenRouter (`typesafe/jev-1.13-20260917`,
  TypeSafe's own direct signup was paused on demand at the time). Run twice —
  once, then again after fixing a `CATEGORY` slot-vocabulary coverage gap
  found post-hoc (see "Live probe" below); ~1,400 calls/run across S1-S6,
  n=200/suite, actual cost well under \$0.05/run. Current results in
  `outputs/probe_results.json` / `outputs/probe_run.log`; first run archived
  at `outputs/probe_results_run1_20260925.json` / `outputs/probe_run_run1_20260925.log`.

### Toy experiment: what the soft-vs-hard ablation actually showed

30k generated items (one fixed pool/split, generated once), compositional
80/20 split on `category` (test items use slot-value combinations never
seen in training), `all-MiniLM-L6-v2` encoder, 3 epochs, identical
architecture/optimizer/schedule between the two modes — only the training
target differs (see `docs/explainers.md`, "soft targets vs. label
smoothing"). **Trained across 3 seeds per mode (100/101/102), mean±std
below** — the plan's own research-practices rule required this
("report across ≥3 seeds for any training claim") and the original write-up
only had one; every number here now has a standard deviation attached, and
it's tight. (Numbers below post-date a fix to the `CATEGORY` slot
vocabulary — see "Prior art"-adjacent note in the live-probe section below —
so the accuracy baseline is a bit higher than an earlier single-seed run
that used a narrower vocabulary; the qualitative finding is unchanged and
now much better attested.)

**Resolution is shared; reliability is not.** On the `category` head
(held-out split), soft and hard training reach essentially the same
accuracy (0.8664±0.0007 vs 0.8687±0.0017) and near-identical AURC
(0.0174±0.0002 vs 0.0176±0.0001 — see `risk_coverage_category.png`): the
two models rank confident-vs-unsure items identically well. But mean
posterior KL divergence from the true label distribution is **16.5x lower
for soft training** (0.0002±0.0000 vs 0.0033±0.0003), and on the
deliberately-ambiguous item family built via `build_ambiguous` — where the
true posterior is genuinely non-degenerate, not near one-hot — the gap
widens to **23.5x** (0.0004±0.0001 vs 0.0094±0.0003) while accuracy is
identical (0.498±0.0000 for both, exactly, every seed). This is the Brier
decomposition's reliability/resolution split (`docs/explainers.md`) showing
up empirically: same architecture, same accuracy, same *ranking* of
confident vs. unsure predictions — but the hard-label model's stated
probabilities are substantially and consistently further from the truth,
replicated across three independent training runs with standard deviations
small enough that this isn't a one-seed artifact. (One honest wrinkle that
is itself a good illustration of the material: the two reliability-diagram
PNGs, illustrative single-seed plots, report ECE values that *disagree*
with the aggregated table above, because the plotting call uses different
bin settings — 10 equal-mass bins there vs. 15 equal-width bins in
`evaluate()`. That's ECE's known bin-sensitivity from the explainer,
showing up unprompted in this project's own output. The posterior-KL
comparison is binning-independent and is the one to trust.)

**The unanswerable concept (`hidden_priority`) was solved correctly by
both, consistently.** True posterior is exactly the prior everywhere
(Bayes-optimal accuracy 0.400). Both soft- and hard-trained models land at
accuracy exactly 0.4037±0.0000 / 0.3860±0.0000 across every seed — both
learned to report close to the prior on a question the text cannot answer.
One reproducible nuance visible only with multiple seeds: soft training's
mean KL is essentially exact (0.0000±0.0000), while hard's is small but
consistently nonzero (0.0018–0.0024±0.0002–0.0004) — hard gets very close
to the prior but not quite as exactly as soft does, reliably, across every
seed. Both still comfortably beat the real Jev's published failure on an
analogous unanswerable question (44.7% accuracy, 0.74 mean confidence) —
suggesting that failure is *not* an inevitable property of this
architecture/loss family under clean training, something about Jev's
actual training distribution or objective is producing it. Suite S3 tests
this directly against the real API (see below).

**Honest limitation: the `urgency` head underfit.** Its accuracy
(0.44–0.48) sits well below its Bayes-optimal ceiling, and mean KL (~0.20)
is far higher than category's — the ordinal cumulative-link head likely
needs more capacity, more epochs, or a less restrictive parameterization
than a single latent utility can offer for this posterior shape. Soft vs.
hard show only a small gap on this head (0.2012±0.0001 vs 0.2076±0.0003 on
held-out — real and consistent, but nowhere near category's 16–23x) —
consistent with both modes being too far from converged for the full
target-choice effect to show through. Worth revisiting before trusting any
urgency-specific conclusion — flagged rather than glossed over.

**Shuffled-label negative control passed.** Accuracy 0.236 against a
uniform-random baseline of 0.25 (single run — this is a sanity check, not a
headline claim needing seed replication) — the pipeline learns nothing from
permuted labels, as it must.

### Live probe: what S1–S6 found against the real Jev API

`typesafe/jev-1.13-20260917`, via OpenRouter, n=200/suite. Organized by the
plan's own pre-registered split: S2 and S3 are the primary endpoints; S1,
S4, S5, S6 are exploratory. **Run twice** — an initial run, then a second
after fixing a coverage gap in the `CATEGORY` slot vocabulary (see below);
numbers here are the second, corrected run. The first run's raw output is
kept for the record at `outputs/probe_results_run1_20260925.json` /
`outputs/probe_run_run1_20260925.log` rather than deleted.

**S2 (known-ambiguous) — falsified, and *why* is the actual finding.**
`enumerate_binary_combos(CATEGORY, ["billing"])` showed the achievable
true-posterior grid had nothing between 0.37 and 1.0 — every
`complaint_phrase` realization was either label-exclusive (posterior → 1.0)
or nearly identical across labels (posterior stuck ≈0.3–0.4), so the
original sweep could only ever probe the *underconfident* side of Jev's
errors. Fixed by adding two new realizations giving billing a real but
non-overwhelming edge over its closest competitor (technical) — grid now
spans {..., 0.466, 0.533, 0.584, 0.647, 0.714, 0.761, 1.0}. Re-run: mean
posterior KL **3.66 nats** (target was <0.15), median jumped from 0.27
(first run) to **3.20** — not noise, a real, different, now-visible
failure mode:

| Target | True P(billing) | Jev's mean reported P | Direction |
|---|---|---|---|
| 0.3 | 0.345 | 0.011 | Underconfident — 18/50 literal near-zeros |
| 0.5 | 0.533 | 0.307 | Underconfident, moderate |
| 0.7 | 0.714 | **0.997** | **Overconfident** — rounds to near-certain |
| 0.9 | 1.000 | 1.000 | Correct (trivially — fully resolved) |

Jev appears to round toward the nearest extreme rather than report the
actual intermediate value, with the under→over crossover somewhere between
true-P 0.53 and 0.71 (n=50/level — that range is defensible, a single exact
crossover value isn't, yet). Both directions are the same underlying
symptom in calibration terms (extremizing, needing T>1 to fix) and match
the independently-published finding's *direction*, but this pins it to an
exact, located crossover that requires an exact tunable posterior to see at
all — no binary or hand-labeled ground truth could produce this table. The
narrower first run's mean KL (3.04) came entirely from the underconfident
side; widening the grid didn't just add data points, it found the opposite-
signed failure the narrow grid structurally couldn't reach.

**S3 (unanswerable) — confirmed cleanly, again.** On `HIDDEN_PRIORITY`
items (true posterior exactly the prior, zero evidence in the text by
construction): accuracy **0.265**, *below* the 0.40 Bayes-optimal ceiling
for guessing the prior; mean confidence **0.672**; overconfidence gap
**+0.272**. Directly replicates the published 44.7%-accuracy/0.74-
confidence finding — on an independently-built, provably-unanswerable item
with a different exact prior. Twice now, on two different constructions,
Jev is confident on a question it structurally cannot know the answer to.

**S1 (resolvable), exploratory:** accuracy 0.91, mean confidence 0.943 —
Jev is genuinely good on easy items, as expected.

**S4 (type consistency), exploratory:** the same `HIDDEN_PRIORITY` draw
(true value exactly 0.300), asked three ways: `noul` says 0.288, `choice`
says 0.366, `score` says 0.363. Choice and score agree closely with each
other (0.039 apart) but both disagree with noul by >0.15 — same question,
three framings, three different answers around a known target.

**S5 (confidence vs. probability), exploratory:** correlation 0.9998 between
`confidence` and max(`probabilities`); AURC essentially identical (0.1339 vs
0.1341). `confidence` appears to be a restatement of max-probability, not a
distinct signal.

**S6 (prior sensitivity), exploratory:** given an explicitly restated base
rate in the question instructions, Jev shifted its answer in the
Bayes-correct *direction* on 200/200 items, but by only ~68% of the correct
*magnitude* (observed shift 0.389 vs. Bayes-correct 0.568) — it listens to
an explicit prior, but underweights it.
