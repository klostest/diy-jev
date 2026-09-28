# diy-jev

Companion repo for the blog post **"DIY Jev"** (link to be added on publication).
The post is the readable story; this repo has the full details: code, data,
methods, and every number.

TypeSafe AI's **Jev** (released 2026-09-15) is a non-autoregressive "System
One" model that returns typed decisions (`noul`/`choice`/`score`) with
probabilities in a single parallel pass, pitched on calibration: *"higher
confidence means higher accuracy."* TypeSafe's own published evals report
accuracy, cost, and latency — no ECE, no reliability diagram, no Brier
score.

This repo builds **one synthetic data-generating process with an exact,
closed-form posterior** (naive-Bayes emission tables, not estimated), and
uses it three ways:

1. **`probe/`** — six pre-registered test suites (S1–S6) against the real
   Jev API, graded against the generator's *known true posterior* rather
   than noisy observed labels.
2. **`toy/`** — a small encoder-based model, trained on an M1 Max, with
   Jev's architectural shape (one encoder pass, parallel per-question
   heads), used for a soft-vs-hard-label training comparison.
3. **`scripts/run_head_to_head.py`** — Jev as is, Jev recalibrated on your
   own labels, and the toy model trained on your own (hard) labels, all on
   the same 6,440 emails, with learning curves over the number of labels.

> **Deprecated: the `hidden_priority` concept (suite S3, and the priority
> parts of the head-to-head).** It was meant as a provably unanswerable
> question: priority is drawn independently of the text. But it shares its
> labels (low/medium/high/critical) with urgency, and the question we asked
> Jev ("What priority level should this ticket be assigned?") gives no
> policy, so treating urgent wording as high priority is a reasonable
> reading, not a calibration failure. Those results test our hidden rule
> more than Jev. The code and results are kept for the record; see
> `RESULTS_AND_DISCUSSION.md`. The S4 consistency result still stands: it
> uses priority questions, but only checks whether Jev's three answer
> formats agree with each other, which doesn't depend on the hidden labels.

How the pieces connect end to end, with a worked example: [`EXPERIMENTAL_SETUP.md`](EXPERIMENTAL_SETUP.md).
What the experiments found, and what it does and doesn't say about Jev: [`RESULTS_AND_DISCUSSION.md`](RESULTS_AND_DISCUSSION.md).
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
  run_head_to_head.py           # Jev (raw / recalibrated) vs. own model on the same 6,440 emails
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

## Verification

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
below** — the project's own research-practices rule required this
("report across ≥3 seeds for any training claim") and the original write-up
only had one; every number here now has a standard deviation attached, and
it's tight. (Numbers below post-date a fix to the `CATEGORY` slot
vocabulary — see "Prior art"-adjacent note in the live-probe section below —
so the accuracy baseline is a bit higher than an earlier single-seed run
that used a narrower vocabulary; the qualitative finding is unchanged and
now much better attested.)

**Resolution is shared; reliability is not.** On the `category` head
(held-out split), soft and hard training reach essentially the same
accuracy (0.867±0.001 vs 0.869±0.002) and near-identical AURC
(0.0173 vs 0.0175 — see `risk_coverage_category.png`): the
two models rank confident-vs-unsure items identically well. But mean
posterior KL divergence from the true label distribution is **8x lower
for soft training** (0.0005±0.0001 vs 0.0037±0.0001), and on the
deliberately-ambiguous item family built via `build_ambiguous` — where the
true posterior is genuinely non-degenerate, not near one-hot — the gap
is **6x** (0.0022±0.0005 vs 0.0134±0.0051) while accuracy is
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
identical accuracy on every eval set and every seed (0.404 held-out, 0.386 on
the ambiguous set — the sets differ only in which hard labels were sampled) — both
learned to report close to the prior on a question the text cannot answer.
One reproducible nuance visible only with multiple seeds: soft training's
mean KL is near-exact (about 0.0002), while hard's is small but
consistently larger (0.002–0.003) — hard gets very close
to the prior but not quite as exactly as soft does, reliably, across every
seed. Both still comfortably beat the real Jev's published failure on an
analogous unanswerable question (44.7% accuracy, 0.74 mean confidence) —
suggesting that failure is *not* an inevitable property of this
architecture under clean training. (It doesn't show what Jev's training
did: the toy model was trained on this world and Jev wasn't — see
`RESULTS_AND_DISCUSSION.md`.) Suite S3 tests
this directly against the real API (see below).

**The `urgency` head: underfit, diagnosed, fixed.** An earlier version used
an ordinal (cumulative-link) head for `urgency` and stalled at accuracy 0.48
(ceiling 0.56) and mean KL ~0.20. Diagnostic runs showed the ordinal form
could represent the posteriors but trained far too slowly (8 epochs only
reached 0.13); a plain softmax head fit in 3 epochs. With it, held-out
urgency accuracy is 0.558/0.560 (soft/hard) and mean KL 0.0025/0.0080 — the
same soft-beats-hard pattern as category. Details in `RESULTS_AND_DISCUSSION.md`;
the earlier outputs are archived in `outputs/toy_experiment_ordinal_urgency_20260926/`.

**Dropout fix (2026-09-26).** A bug left dropout switched off for the last
two training epochs in earlier runs. With it fixed, soft training's advantage
shrank from 10–30x to 6–8x on `category` (hard training was barely
affected); the numbers above are post-fix. Pre-fix outputs are archived in
`outputs/toy_experiment_dropout_bug_20260926/`.

**Shuffled-label negative control passed.** Accuracy 0.253 against a
uniform-random baseline of 0.25 (single run — this is a sanity check, not a
headline claim needing seed replication) — the pipeline learns nothing from
permuted labels, as it must.

### Live probe: what S1–S6 found against the real Jev API

> **Read with `RESULTS_AND_DISCUSSION.md`.** Re-checking these results found
> caveats not reflected below: S2's 3.66 nats is dominated by Jev's exact-zero
> probabilities under a 1e-12 floor (0.67 nats treating Jev's two-decimal
> zeros as 0.005 — still falsified); all 18 S1 misses are feature requests,
> which the tables label *sales*; S3's low accuracy comes from Jev reading
> urgency wording as priority; and S6's "underweights the stated prior" does
> not hold against Jev's own starting belief.

`typesafe/jev-1.13-20260917`, via OpenRouter, n=200/suite. Organized by the
project's pre-registered split: S2 and S3 are the primary endpoints; S1,
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
for guessing the prior; mean `confidence` field **0.672** (gap +0.272 as
pre-registered), but that field isn't a probability (see S5 below): on the
probability scale Jev's mean top probability is **0.754**, a gap of **+0.35**. Directly replicates the published 44.7%-accuracy/0.74-
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
0.1341). TypeSafe's docs (docs.typesafe.ai/confidence) say confidence is
computed from the probability distribution; for `choice` answers it's exactly
(top − 1/K) / (1 − 1/K), the top probability rescaled so a uniform guess
reads 0 (16,283 recorded `choice` answers with 2–6 options, within about 0.02; see
`scripts/check_confidence_formula.py`). So it's
not a probability and carries no separate signal.

**S6 (prior sensitivity), exploratory:** given an explicitly restated base
rate in the question instructions, Jev shifted its answer in the
Bayes-correct *direction* on 200/200 items, but by only ~68% of the correct
*magnitude* (observed shift 0.389 vs. Bayes-correct 0.568) — it listens to
an explicit prior, but underweights it.
