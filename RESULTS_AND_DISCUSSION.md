# Results and discussion

What happened when the setup in
[`EXPERIMENTAL_SETUP.md`](EXPERIMENTAL_SETUP.md) was actually run. Read
that first: this document introduces no new definitions. Every metric used
here (accuracy and its ceiling, posterior KL and nats, ECE, the
overconfidence gap, AURC, and the pass/fail thresholds) is defined, with
its equation, in Part 3 there. Every number below was recomputed from the
files in `outputs/` for this document, not copied from the README.

## The findings in brief

About Jev (`typesafe/jev-1.13-20260917`, run 2026-09-26; 200 emails per
test in S1–S6):

1. **Asked the same question three ways, Jev gives three different
   answers** (S4). On the same email, its yes/no answer and its
   multiple-choice answer to "is this high priority?" differ by 0.17 on
   average.
2. **Jev reports near-certainty on emails that say outright they're
   uncertain** (S2). An email reading "I think I'm being charged
   incorrectly, though it might be tied to a feature not working" got
   P(billing) ≥ 0.99 on all 50 such emails. Half of all S2 answers put 95%
   or more on a single label.
3. **Jev's `confidence` field is its top probability, rescaled** (S5).
   TypeSafe's docs say it's computed from the probabilities; we checked
   that for `choice` answers it's exactly (top − 1/K) / (1 − 1/K) with K
   options (16,283 answers, 2–6 options), so a uniform guess reads 0. It
   adds no information, and it isn't a probability: a confidence of 0.54
   means a top probability of 0.655.
4. *(Deprecated; see S3.)* **Jev gives no sign when a question can't be answered from the text**
   (S3). Asked a priority question whose answer, in this synthetic world,
   has nothing to do with the email, it answered from the email's urgency
   wording with a median top probability of 0.76, and was right less often
   than blind guessing (26.5% vs. 40%).

About the toy model (Part 2 of the setup):

5. **Accuracy can't see calibration.** Two models trained identically
   except for the loss reach the same accuracy, but one's probabilities
   are 6–8× further from the truth. An accuracy-only evaluation, which
   is what TypeSafe publishes, can't show whether a model's probabilities
   are right.
6. **A model built the way Jev is can say "I don't know" correctly.** On
   the unanswerable question, the toy model reports the base rates almost
   exactly. So Jev's behavior in S3 isn't forced by its architecture.

Jev against a model trained on your own labels (6,440 emails, Part C):

7. **With a few hundred labels, recalibrate Jev; with a few thousand,
   train your own.** Adjusting Jev's probabilities with 200–1,000 labeled
   emails lifts its accuracy from 0.79 to 0.86 and cuts its KL fivefold,
   mostly by teaching it this world's conventions. A model trained from
   scratch on hard labels loses to Jev at 200 labels, ties it at about
   1,000, and beats every version of it at 5,000, including on ranking
   (AURC 0.017 vs. Jev's 0.046 raw, 0.021 at best).

The rest of this document is the evidence for each, and its limits.

---

## Before the numbers: "exact" means exact for *our* world

Part 1 of the setup makes the answer key exact: given the tables in
`slots.py`, the right probability for every email is known to the last
decimal. But those tables describe a world we invented. They say, for
example, that a feature request is always a *sales* email, that vague
emails are equally likely to be billing or technical, and that a ticket's
priority has nothing to do with how urgent it sounds. Jev has never seen
these tables. It brings its own sense of how support email works, and
where that differs from ours, it will disagree with the answer key even if
its probabilities are perfectly honest by its own lights.

The toy model doesn't have this problem, since it was trained on this
world. Jev does, so each Jev result below is labeled:

- **Holds regardless of our world.** A property of Jev's own answers that
  any honest forecaster would avoid, whatever it believed about support
  email. For example, contradicting itself, or being certain about text
  that itself expresses doubt.
- **Depends on our world.** Measures Jev against rules we made up, which
  it couldn't know. Still informative, but a "wrong" answer here may be a
  disagreement about the world rather than bad calibration.

This distinction matters most for the blog post. The first kind can be
stated as facts about Jev. The second kind needs the caveat attached.

---

## Part A — The real Jev

The six suites and their pre-registered predictions are described in Part
2 of the setup; the thresholds are explained in Part 3's
"pre-registration" section. S2 and S3 are the primary tests.

### S2 (primary): does Jev report "in-between" probabilities as in-between?

**The test.** 200 emails at four levels of ambiguity, 50 each, chosen so
that the exact P(billing) is 0.345, 0.533, 0.714 or 1.0. Jev is asked the
`category` question, and its full four-label answer is compared with the
exact four-label posterior.

**One email, and what Jev said.** This is the 0.714 level: the same
`strong_lean` sentence as Part 1's worked example.

```
From: (sender address withheld)
Subject: Quick note

Thanks so much for your help.
I think I'm being charged incorrectly, though it might be tied to a
feature not working as expected.

Thanks,
Maria
```

```json
{"choice": "billing",
 "probabilities": {"billing": 1, "technical": 0, "sales": 0, "spam": 0},
 "confidence": 1}
```

The exact answer is billing 0.714, technical 0.190, sales 0.048, spam
0.048. Jev said billing, with certainty.

**All four levels.** Average of Jev's reported probabilities over the 50
emails at each level, next to the exact values:

| Level | Deciding sentence (realization) | Exact (billing / technical / sales / spam) | Jev, averaged | Jev's P(billing), min–max |
|---|---|---|---|---|
| 0.345 | a vague greeting only (`generic_greeting`) | 0.345 / 0.345 / 0.207 / 0.103 | 0.011 / 0.619 / 0.043 / 0.327 | 0.00–0.06 |
| 0.533 | "might be a charge, might be a feature that broke" (`mixed_lean`) | 0.533 / 0.333 / 0.067 / 0.067 | 0.307 / 0.693 / 0.000 / 0.000 | 0.07–0.75 |
| 0.714 | "I think I'm being charged incorrectly, though…" (`strong_lean`) | 0.714 / 0.190 / 0.048 / 0.048 | 0.997 / 0.003 / 0.000 / 0.000 | 0.99–1.00 |
| 1.0 | "My invoice amount looks wrong" (`invoice_issue`) | 1 / 0 / 0 / 0 | 1.000 / 0 / 0 / 0 | 1.00–1.00 |

**Against the pre-registered test.** Mean posterior KL was **3.66 nats**
(median 3.20), far past the 0.3-nat falsification line. **The prediction
is falsified.** That number needs a caveat, though. Jev rounds to two
decimals, so it often reports exactly 0, and 119 of the 200 answers gave 0
to a label with real probability. Under the harsh 10⁻¹² floor the
pre-registered computation uses, those zeros dominate the total (Part 3,
"What a reported 0 costs"). With the rounding-aware computation, which
reads each of Jev's zeros as 0.005, mean KL is **0.67 nats** (median 0.67).
That's still more than twice the falsification line, so the verdict holds,
but 3.66 overstates the size of the miss about fivefold. **Use 0.67 when
describing it, and mention the 3.66 only with this explanation.** (Per level,
rounding-aware: 1.45, 0.56, 0.65 and 0.02 nats.)

**What it means, level by level.**

- **0.714 — holds regardless of our world.** The email says in so many
  words that it might not be a billing problem. However a forecaster
  weighs that hedge, 99–100% billing, every time, ignores it. It held for
  all three wordings of the sentence, including "This looks like a billing
  problem to me, but it could also be a glitch on your end" (13 emails, all
  0.99–1.00). This is the single clearest calibration failure in the
  project.
- **0.533 — partly world-dependent.** The sentence is written to be
  balanced. Our tables tilt it slightly toward billing (0.533), and Jev
  tilts it toward technical (0.31 billing on average). Neither tilt is
  obviously right. The wide spread across the 50 emails (0.07 to 0.75) is
  almost entirely *which wording* each email got. Jev was consistent within
  each wording but read the three very differently:

  | Wording | Emails | Jev's P(billing), mean (range) |
  |---|---|---|
  | "I'm having trouble with my account, not totally sure if it's a billing thing or a bug." | 20 | 0.13 (0.07–0.23) |
  | "Something's wrong with my account -- might be a charge, might be a feature that broke." | 19 | 0.28 (0.17–0.40) |
  | "There's an issue with my account I can't quite pin down -- possibly a payment problem, possibly something not working right." | 11 | 0.68 (0.58–0.75) |

  To most readers all three say "could be billing, could be a bug," which
  is why our tables give them one answer. Jev seems to weigh particular
  words heavily ("bug" toward technical, "payment problem" toward billing).
  Whether that's a flaw depends on whether the three sentences really are
  equivalent, so treat it as an observation. A smaller part of the spread
  may come from the email's unrelated urgency wording: emails that sounded
  urgent got higher P(billing) (0.43–0.47, 16 emails) than ones that didn't
  (0.22–0.26, 34 emails). In this world urgency has nothing to do with
  category. The groups are small and weren't checked for balance across
  the three wordings, so this is a lead, not a finding.
- **0.345 — world-dependent.** The email says nothing about its topic.
  The right answer is then mostly the base rates, and ours are invented.
  Jev thinks a contentless email is most likely technical (0.62) or spam
  (0.33), and almost never billing (0.01). That's a different set of base
  rates, not a calibration failure we can pin on it.
- **1.0 — correct.** Unambiguous emails get certain answers, as they should.

Across all 200 S2 answers, 51% put 95% or more on a single label. Jev's
probabilities pile up near 0 and 1 even when the text doesn't justify it.

**What about Jev's `confidence` field?** S2 is graded on the full
probability vector, because only a distribution can be compared with the
exact four-label answer. The separate `confidence` number adds nothing,
because it's computed from the top probability (see S5):

| Exact P(billing) | Jev's mean `confidence` | Jev's mean top probability |
|---|---|---|
| 0.345 | 0.558 | 0.668 |
| 0.533 | 0.690 | 0.770 |
| 0.714 | 0.996 | 0.997 |
| 1.0 | 1.000 | 1.000 |

Each row follows confidence = (top − 0.25) / 0.75: for example,
(0.668 − 0.25) / 0.75 = 0.557. The two agree near certainty and differ in
the middle only because of that rescaling. `confidence` isn't a less
overconfident second opinion, and it isn't a probability, so it shouldn't
be compared with the exact answers. On the hedged 0.714 emails both say
near-certainty. Whichever field a user reads, Jev claimed to be sure
about emails that say they aren't.

For the record: the first probe run could only test the 0.345 and 1.0
levels, because the tables couldn't produce anything in between (setup,
Part 3). Filling that gap is what exposed the 0.714 result. The first
run's average errors were all in one direction (too low); the second run
showed errors in both.

### S3 (primary): does Jev admit when it can't know?

> **Deprecated.** This suite's premise doesn't hold up: priority is drawn
> independently of the text, but it shares its labels with urgency, and
> the question gives Jev no priority policy, so reading urgent wording as
> high priority is reasonable rather than a calibration failure. The
> results below test our hidden rule more than Jev, and are kept only as a
> record. (A post-hoc follow-up that told Jev the policy isn't visible,
> `scripts/run_s3_policy_gap.py`, found it hedged somewhat but kept
> reading urgency wording.) S4's consistency result is unaffected.

**The test.** Ordinary emails, with Jev asked "What priority level should
this ticket be assigned?" In this world, priority is drawn by its own coin
flip and never affects the text (setup, Part 1), so the exact answer for
every email is the base rates: low 0.40, medium 0.30, high 0.20, critical
0.10. The best possible strategy is to always guess *low*, which is right
40% of the time, with a top probability of 0.40.

**One email, and what Jev said.**

```
From: sofia.costa@gmail.com
Subject: Need some help

Thanks so much for your help.
Do you have any discount codes available?
```

```json
{"choice": "low",
 "probabilities": {"low": 0.95, "medium": 0.05, "high": 0, "critical": 0},
 "confidence": 0.94}
```

(Its sampled priority happened to be *low*, so this one counted as
correct.)

**Against the pre-registered test.** Accuracy **0.265** (53/200; 95%
interval 0.209–0.330, entirely below the 0.40 that blind guessing would
get). The test was written in terms of Jev's `confidence` field: mean
0.672, an overconfidence gap of +0.27. But `confidence` isn't a
probability (S5), so the like-for-like comparison with the 0.40 accuracy
ceiling uses Jev's top probability: mean **0.754**, a gap of **+0.35**.
Either way, **the prediction is confirmed**: Jev is confident on a question
it can't answer from the text. The corrected numbers make it more so.

**Why it did worse than guessing.** Grouping the emails by the urgency
phrase they happen to contain (from the separate `urgency` concept):

| Urgency phrase in the email | Emails | Jev's average P(high or critical) | Jev's accuracy |
|---|---|---|---|
| none | 70 | 0.26 | 0.34 |
| mild ("no particular rush") | 60 | 0.11 | 0.27 |
| strong ("this is fairly urgent") | 52 | 0.66 | 0.17 |
| extreme ("URGENT -- please treat as a priority") | 18 | 0.77 | 0.22 |

Jev reads urgency wording as evidence of priority. In our world, that
wording is unrelated to priority by construction, so following it makes
Jev worse than ignoring it.

**What it means — mostly world-dependent, with one part that isn't.** A
person asked "what priority should this ticket get?" about an email that
says URGENT would also say *high*. Nothing in the question tells Jev that
priority is unrelated to the wording, so treating the wording as evidence
is a reasonable reading. The claim "Jev is overconfident on an unanswerable
question" therefore depends on our world's rule. What holds more generally
is the *size* of the confidence: a median top probability of 0.76, and
0.98 or more on one answer in nine, from a single stock phrase. And Jev has no way, in its
output, to say "the text doesn't determine this." The README cites an
independent test that found something similar on a different
unanswerable question (44.7% accuracy, 0.74 confidence, as reported
there; whether that "confidence" is the rescaled field isn't stated).

### S1 (exploratory): easy emails

Emails whose exact answer is at least 97% one category. Accuracy **0.91**
(95% interval 0.862–0.942), mean top probability **0.958**, ECE 0.066.
**The prediction is confirmed** (both above 0.9). The pre-registered
computation used Jev's `confidence` field (mean 0.943, ECE 0.061). That
field isn't a probability (S5), but here the difference is small because
most answers are near certainty, where the two agree.

Every one of the 18 misses is the same thing: an email asking for a new
feature ("Could you add a dark mode option?"). Our tables make these
*sales* only (`feature_request` has zero probability under every other
label), and the answer-option description Jev was given for sales, "About
pricing, plan features, or discounts" (see the setup's "How we talk to
Jev"), reasonably covers them. Jev leaned *technical* anyway, presumably
from its own sense that product requests go to a technical team.

But it didn't do so confidently. Across all 26 feature-request emails in
S1, Jev picked technical 18 times and sales 8 times, and its P(sales)
ranged from 0.19 to 0.72. It never came close to certainty either way.
For a label whose definition is genuinely open to two readings, a split,
moderate answer is a reasonable one. It's the opposite of what S2 found
on hedged billing emails.

On every other kind of easy email, Jev was 182 for 182. Read S1 as "Jev is
reliable on clear-cut emails." The feature-request misses are a
world-dependent disagreement about a definition, and Jev handled that
disagreement with appropriate uncertainty.

### S4 (exploratory): the same question, three ways

**The test.** On each email, Jev is asked about priority three ways in a
single request: a yes/no question ("should this be expedited, i.e. high or
critical?"), the multiple-choice priority question, and a 0–3 score. Each
answer converts to the same number, P(high or critical) (setup, Part 3).
Its exact value is 0.30 on every email.

For the discount-codes email above: yes/no 0.06, multiple choice 0.00,
score 0.00. Consistent there, but not in general:

| Framing | Average P(high or critical) |
|---|---|
| yes/no | 0.288 |
| multiple choice | 0.366 |
| 0–3 score | 0.363 |
| *exact* | *0.300* |

Averages hide the per-email disagreement. On the same email, the yes/no
and multiple-choice answers differ by **0.167** on average, and yes/no
and score by 0.155. Multiple choice and score agree more closely (0.039).
**The prediction is confirmed** (disagreement above 0.1).

**What it means — holds regardless of our world.** Whatever Jev believes
about priority, those beliefs are the same across the three questions,
because they're the same question about the same email in the same
request. A consistent forecaster gives the same number all three times.
This is the cleanest evidence in the project that Jev's probabilities
depend on the question format, not only on the email.

### S5 (exploratory): what is the `confidence` field?

Jev returns a `confidence` alongside its probabilities. The pre-registered
question was whether it adds anything beyond them. Across 200 emails
(half easy, half ambiguous), its correlation with Jev's own highest
probability is **0.9998**. Ranking emails by either gives the same AURC
(0.1339 vs 0.1341). **The prediction is confirmed**, and this holds
regardless of our world: `confidence` carries no information that the
probabilities don't already contain.

**It's a formula, and TypeSafe's docs say so.** The docs' confidence page
(docs.typesafe.ai/confidence, checked 2026-09-28) says `confidence` "is a
statistic computed from the probability distribution the answer already
gives you", and shows (3 × largest probability − 1) / 2 as what "this demo
uses to approximate confidence for three options". It gives no general
formula and nothing for `score` questions. The general form for `choice`
questions is

> confidence = (top probability − 1/K) / (1 − 1/K), with K answer options

which is the docs' formula when K = 3. With four options, a flat
25/25/25/25 answer gets confidence 0 and certainty gets 1. What this
project adds is checking that the real API follows it exactly, for any
number of options. `scripts/check_confidence_formula.py` tests this on
every `choice` answer Jev has returned in this project, plus 100 new
emails asked 2-, 3-, 5- and 6-option versions of the category question
(all four S1–S6 questions have 4 options, which can't test the 1/K part):

| Options (K) | Answers | Largest error, the formula | Largest error, same formula with K fixed at 4 | Largest error, best alternative |
|---|---|---|---|---|
| 2 | 100 | 0.010 | 0.333 | 0.010 (top − second, which equals the formula when K = 2) |
| 3 | 100 | 0.010 | 0.083 | 0.23 |
| 4 | 15,883 | 0.023 | — | 0.24 |
| 5 | 100 | 0.018 | 0.047 | 0.13 |
| 6 | 100 | 0.016 | 0.080 | 0.14 |

"Alternatives" means the plain top probability, the gap between the top
two, and one minus normalized entropy. Across 16,283 answers the formula
is never off by more than about 0.02. The K-fixed-at-4 version fails as
soon as K changes, so the dependence on the number of options is real.

The fit is close but not exact to the last digit. The same displayed
probabilities occasionally come with confidences up to 0.04 apart (for
example (0.61, 0.35, 0.03, 0.01) with confidences 0.46–0.50, around a
formula value of 0.48), more than rounding alone explains. The most likely
explanation is that confidence is computed from Jev's unrounded internal
probabilities. Either way, that leftover carries **no** information about
whether an answer is right. On the 6,440 head-to-head emails, adding it to
the top probability doesn't improve prediction of correctness (10-fold
cross-validated log loss changes by 0.00004 on category and 0 on
priority). Among emails with the same displayed top probability, those
whose confidence sits above the formula are no more often right than
those below it (differences of −0.011 and +0.009).

Consequences:

- **`confidence` adds no information.** It's a fixed increasing function
  of the top probability plus meaningless noise, so it ranks answers the
  same way (the AURC difference above is noise) and can't flag an answer
  the probabilities don't already flag. The hedged S2 emails got 0.99+ on
  both.
- **`confidence` is not a probability.** It measures how far the top answer
  is above a random guess. A confidence of 0.54 on a four-option question
  means a top probability of 0.655, not a 54% chance of being right. A user
  who reads it as a probability will under-read what Jev is claiming,
  especially in the middle of the range.
- **It's a reasonable design, but it sits awkwardly with the pitch.**
  Measuring "distance from chance" puts questions with different numbers
  of options on one scale: 0.5 means halfway from a random guess to
  certainty whether there are 2 options or 6. The docs are upfront that
  it's computed from the probabilities. But TypeSafe's launch post
  describes Jev as "Calibrated: higher confidence means higher accuracy",
  and the docs suggest gating actions on confidence thresholds. Since
  confidence is just the top probability re-expressed, those thresholds are
  thresholds on the top probability, and they're only as good as the
  probabilities' calibration, which is what S2 found lacking.

Limits: this covers `choice` questions only. `score` questions report a
`confidence` too, but none of these formulas fits it, and the docs don't
say how it's computed, so what it measures there is undetermined. Yes/no
(`noul`) answers have no confidence field, as the docs also state.
All of this is inferred from Jev's outputs, not from its code. It's a
precise description of what Jev returns, not a statement about how it's
implemented. `outputs/figures/s5_confidence.png` plots confidence against
the top probability with the formula drawn through the points; the check's
numbers are in `outputs/confidence_check/results.json`.

### S6 (exploratory): does Jev use a stated base rate?

**The test.** One ambiguous email design (exact P(spam) = 0.5 under
the world's own base rates), sent twice with the instruction prefixed by
"only about 10% of tickets in this queue turn out to be spam" or "about
60% … are spam." Bayes' rule says P(spam) should move from 0.250 to 0.818,
a shift of +0.568 (setup, Part 3).

**Against the pre-registered test.** Jev moved in the right direction on
**all 200** emails, but by **+0.389** on average (from 0.585 to 0.974).
That's short of +0.568 by more than 0.1, so **the prediction ("Jev
under-reacts") is confirmed** as written.

**Why that verdict shouldn't be the headline.** Jev started much higher
than our world's answer: even when told spam is rare, it put P(spam) at
0.585 on average, where our tables say 0.25. That starting point is a
world-dependent disagreement (Jev finds these vague emails spammier than
our tables do). Starting at 0.585, there's less room to rise. Measuring
the second answer against Jev's *own* first one, Bayes' rule predicts it
should reach about 0.945 when told 60%. Jev said 0.974: if anything
slightly *more* than its own starting belief implies, on 92% of emails.
So the fair summary is that Jev responds to a stated base rate in the
right direction every time, by roughly the right amount relative to its
own beliefs. "Jev under-weights stated priors" is not supported, even
though the pre-registered test technically confirmed it. (Probabilities
this close to 1, rounded to two decimals, make the self-consistency
comparison approximate.)

---

## Part B — The toy model

### Why it exists, and what it can and can't say about Jev

Jev's weights and training data aren't public, so the probe can only
observe what Jev does, not why. The toy model is a small stand-in built the
way TypeSafe describes Jev: one encoder pass over the text, then a separate
small output head per question (setup, Part 2). Because it's small and
fully ours, it can be trained two different ways and the results compared
directly. Everything is identical except the training target:

- **Soft:** trained against the exact probabilities (e.g. "71% billing,
  19% technical, …").
- **Hard:** trained against one sampled label per email ("billing"), the
  way most classifiers are trained.

It can answer two questions that bear on Jev:

1. *Can standard evaluations detect whether a model's probabilities are
   right?* If two models are equally accurate but differ in calibration,
   then no: accuracy numbers alone can't back a calibration claim.
2. *Can a model built like Jev learn to say "I don't know" properly?* If
   it can, Jev's S3 behavior isn't forced by its design.

It **can't** say how Jev was trained or why Jev behaves as it does. The toy
model was trained on this exact synthetic world; Jev was tested cold. So
the two sets of numbers aren't comparable as scores. Even the *worse*
(hard-trained) toy model is far closer to the truth on S2-style items
than Jev. The gap between the toy model and Jev comes from Jev not knowing
our world, not from soft versus hard training. Nothing here supports
saying "Jev was trained with hard labels."

### The headline: equal accuracy, very different honesty

Results on the `category` head, mean ± standard deviation across three
training seeds (a std is omitted where it rounds to zero at the precision
shown):

| Eval set | Accuracy, soft | Accuracy, hard | Posterior KL, soft (nats) | Posterior KL, hard (nats) | Hard ÷ soft |
|---|---|---|---|---|---|
| Held-out test split | 0.867 ± 0.001 | 0.869 ± 0.002 | 0.0005 ± 0.0001 | 0.0037 ± 0.0001 | 8× |
| Ambiguous family | 0.498 | 0.498 | 0.0022 ± 0.0005 | 0.0134 ± 0.0051 | 6× |
| Resolvable family | 1.000 | 1.000 | 0.0001 | 0.0008 ± 0.0002 | 7× |
| Unanswerable family | 0.762 ± 0.001 | 0.761 ± 0.002 | 0.0009 ± 0.0002 | 0.0065 ± 0.0003 | 7× |

**Start with the ambiguous row, where accuracy is exactly equal.** That set
uses one forced combination of clues, so all 1,000 emails have the same
exact answer, with sales-or-spam as the most likely label. Any model that
ranks that label first will pick the same answer for all 1,000 emails. Its
accuracy is then just the fraction of sampled labels that happened to match
that answer: 0.498, for both models, on every seed. Accuracy literally
cannot tell the models apart here, but posterior KL can: 0.0022 nats for
soft against 0.0134 for hard, 6× apart.

**The same pattern holds on every set.** Accuracy and AURC, which measures
how well confident answers are ranked above unsure ones, are essentially
identical between the two models (held-out AURC on `category`: 0.0173
soft, 0.0175 hard). Posterior KL is 6–8× lower for soft training on
`category`, and 3–14× lower on the other three questions (`urgency` 3–4×,
`is_business` 5–7×, `hidden_priority` 12–14×), on every eval set and every
seed. In the setup's vocabulary (Part 3, "reliability and resolution"), the two
models have the same *resolution*: they rank emails equally well. Only the
soft-trained one has good *reliability*: it reports the right numbers.

**Why this matters for Jev.** TypeSafe's published evaluations report
accuracy, cost and latency. Here are two models whose accuracy matches to
the third decimal while one's probabilities are 6–8× further from the
truth. So accuracy numbers, however good, can't establish that a model's
probabilities mean what they say. Only a test against known
probabilities, like S2, can.

### Hard labels are enough; soft labels make it cheaper

It's easy to read the table above as "soft labels are needed for
calibration." They aren't. The hard-trained model never saw a single true
probability, only one observed label per email, which is the situation of
anyone training on historical outcomes. Yet its KL on held-out emails is
0.0037 nats, excellent in absolute terms: well under the 0.028 nats of
reporting 0.60 when the truth is 0.714 (setup, Part 3). The reason is that
cross-entropy is a *proper scoring rule* (`docs/explainers.md`, "Proper
scoring rules"). Across many emails with the same clues, about 71% of the
sampled labels say billing, and the prediction that minimizes cross-entropy
on them is 0.71. Given enough examples per situation, training on outcomes
with a proper loss recovers calibrated probabilities by itself.

Soft labels don't make calibration *possible*; they make it more
*data-efficient*. Each example carries the whole distribution rather than
one noisy draw from it. Here, where the 116 `category` clue combinations
in the training split each appear in a median of 151 training emails (at
least 21), that efficiency shows up as a 6–8× smaller KL, from an already
small base. With fewer examples per situation, the gap would be expected
to grow.

**How big the advantage looks depends on training details.** An earlier run
of this experiment had a bug that switched dropout off for the last two of
three training epochs. Soft training fit the exact targets more tightly
under those conditions, and the soft advantage came out at 10–30× on
`category`. With dropout working as intended, which is standard practice
for fine-tuning, it's 6–8×. The hard-trained model was barely affected
(0.0036 vs. 0.0037 nats held-out). The direction is robust: soft beats hard
on every question, eval set and seed in both runs. But treat any specific
multiple as specific to this setup. The earlier results are archived in
`outputs/toy_experiment_dropout_bug_20260926/`.

### The unanswerable question: the toy model gets it right

The best possible behavior is to report the base rates (low 0.40, medium
0.30, high 0.20, critical 0.10) for every email and pick *low*. Both
models do essentially that. Their accuracy is identical on every eval set:
0.404 on held-out, 0.400 on the unanswerable family and 0.386 on the
ambiguous family. The differences between sets come only from which hard
labels happened to be sampled into each. Soft training's KL from the base
rates is about 0.0002 nats. Hard training's is about 0.002–0.003, small but
consistently larger on every seed.

This is the contrast with S3. A model built the way Jev is, trained on data
where priority genuinely doesn't depend on the text, learns to report the
base rates instead of guessing from urgency wording. So Jev's S3 behavior
isn't a limit of the architecture. The caveat from above applies in full,
though: the toy model was *shown*, in training, that priority is unrelated to
the wording. Jev wasn't, and a person might reasonably assume the opposite.

### The `urgency` head: a defect found and fixed

The first version of this experiment reported the `urgency` head as
underfit: held-out accuracy 0.48 against a ceiling of 0.56, and KL around
0.20 nats for both training modes. That was worse than a model that looked
only at the urgency phrase and ignored the exclamation marks entirely
(0.12). The head then used an *ordinal* design, which builds in
low < medium < high < critical. Four single-seed diagnostic runs (soft
training, same data) located the problem:

| Urgency head | Epochs | Held-out KL (nats) | Accuracy (ceiling 0.560) |
|---|---|---|---|
| Ordinal (original) | 3 | 0.204 | 0.483 |
| Ordinal | 8 | 0.131 | 0.483 |
| Ordinal, 5× head learning rate | 3 | 0.069 | 0.521 |
| Plain softmax | 3 | 0.0045 | 0.559 |

Fitting the ordinal form directly to the 16 exact urgency posteriors
reaches 0.011 nats, so it *could* represent the answers. It just trains
far too slowly in this setup. The head is now a plain softmax. With it,
the full three-seed experiment gives held-out urgency accuracy **0.558
(soft) / 0.560 (hard)**, at the ceiling, and KL **0.0025 / 0.0080**, the
same soft-beats-hard pattern as `category` (3×). The earlier results are
archived in `outputs/toy_experiment_ordinal_urgency_20260926/`. The
`urgency` question isn't used in any Jev test, so none of the Jev results
are affected.

### Checks that the toy results aren't artifacts

- **Shuffled labels.** Retraining with `category` labels randomly shuffled
  gives 0.253 held-out accuracy, against 0.25 for random guessing.
  With nothing real to learn, the pipeline learns nothing.
- **Three training seeds.** Every number is a mean over three
  independently trained models. The standard deviations are small next to
  the soft/hard gaps, so the gaps aren't a quirk of one run.
- **Accuracies just above the ceiling.** A few accuracies edge past the
  best possible (e.g. `is_business` 0.7095 against a ceiling of
  0.700). That's luck in which hard labels were sampled into one finite test
  set, within normal sampling noise (setup, Part 3, "Accuracy, and its
  ceiling"); the ceiling limits the long-run average, not one test set.
- **ECE agrees in direction.** The more common calibration metric also
  favors soft (held-out `category`: 0.0058 soft vs. 0.0130 hard), but it
  depends on bin choices, so KL is the one to rely on.

---

## Part C — Jev vs. a model trained on your own labels

Protocol in the setup's Part 2 ("Head-to-head"); the three ways of adjusting
Jev's probabilities are defined in its Part 3 ("Recalibrating Jev"). In
short: all 6,440 held-out test emails, a natural mix of easy and hard; 2,000
of them available as labeled data for fixing Jev; every option scored on
the other 4,440. The own-model option is trained on **hard labels only**,
the observed outcomes a business actually has. Values are mean ± std over
3 repeats or seeds.

### The category question

| Approach | Labeled emails used | Accuracy | Posterior KL (nats) | ECE | AURC |
|---|---|---|---|---|---|
| *Exact posterior (ceiling)* | — | *0.868* | *0* | *0.007* | *0.016* |
| Jev, as is | 0 | 0.788 | 0.31 | 0.102 | 0.046 |
| Jev + temperature scaling | 1,000 | 0.788 | 0.25 | 0.121 | 0.046 |
| Jev + logistic regression | 1,000 | 0.851 ± 0.003 | 0.11 | 0.061 ± 0.006 | 0.029 |
| Jev + isotonic regression | 1,000 | 0.860 ± 0.001 | 0.063 | 0.035 ± 0.005 | 0.021 |
| Own model | 50 | 0.43 ± 0.11 | 0.93 ± 0.21 | 0.09 ± 0.03 | 0.51 ± 0.23 |
| Own model | 200 | 0.74 ± 0.09 | 0.41 ± 0.13 | 0.14 ± 0.09 | 0.10 ± 0.07 |
| Own model | 1,000 | 0.858 ± 0.007 | 0.072 ± 0.016 | 0.034 ± 0.011 | 0.021 |
| Own model | 5,000 | 0.867 ± 0.001 | 0.021 ± 0.004 | 0.023 ± 0.009 | 0.017 |
| Own model | 23,560 | 0.870 ± 0.001 | 0.005 ± 0.001 | 0.013 ± 0.007 | 0.017 |
| *Own model, soft labels (reference)* | *23,560* | *0.871* | *0.0004* | *0.006* | *0.016* |

(Jev's KL here uses the rounding-aware floor; with the pre-registered
10⁻¹² floor it's 1.04. Jev's recalibration was also fitted with 50, 200 and
2,000 labels. Isotonic and logistic are already within noise of their
2,000-label results by 1,000, and temperature scaling never moves accuracy
or AURC, so the 1,000-label rows are representative. The full grid is in
`outputs/head_to_head/results.json`.)

**How the options compare, by number of labels:**

- **With no labels, Jev is the only option,** and it's decent but well off
  the ceiling: 0.788 accuracy against 0.868 possible, and it ranks emails
  noticeably worse (AURC 0.046 against 0.016). On this natural mix its KL
  is 0.31, lower than S2's 0.67 because most emails here are easy.
- **With 50–200 labels, fixing Jev beats building your own.** A model
  trained from scratch on 200 emails is worse than raw Jev, and wildly
  variable from seed to seed (accuracy 0.74 ± 0.09). Jev + logistic
  regression on just 50 labels already reaches 0.849.
- **At about 1,000 labels, it's a tie.** Your own model (0.858, KL 0.072)
  and Jev + isotonic regression (0.860, KL 0.063) are within noise of each
  other on every metric.
- **At 5,000 labels and up, your own model wins on everything.** It's close
  to the ceiling on accuracy and ranking, with 3–12× lower KL than the best
  version of Jev. Adding labels to Jev's recalibration stops helping by
  about 1,000.

**On ranking:** raw Jev ranks worse than an own model
trained on 1,000+ labels (AURC 0.046 vs. 0.021 and lower). Temperature
scaling can't fix that, since it never reorders anything (AURC stays
0.046). Isotonic and logistic regression *can*, because they're allowed to
change which label comes out on top. They bring Jev to 0.021 and 0.029.
They still can't beat a well-trained own model: they only see Jev's four
numbers, never the email itself.

**Why fixing Jev helps so much: it learns your conventions.** Most of raw
Jev's gap to the ceiling isn't misreading emails. It's disagreeing with this
world's conventions. Of the evaluation emails where Jev's top label differs
from the exact answer's, the largest groups are:

- emails with **no complaint sentence at all**, where the answer rests on
  the sender's domain and the base rates, and our world says spam or sales
  while Jev mostly says technical (about 490 emails), and
- **feature requests** (199), which Jev calls technical and our tables call
  sales (Jev gets only 46% of feature requests "right").

These are exactly the world-dependent disagreements from Part A, and
exactly what a few hundred local labels can teach a recalibrator. That's
the business lesson: Jev's out-of-the-box probabilities reflect Jev's idea
of what a support email usually is, and your labels tell it what *yours*
usually are.

### Equal label budgets: recalibrating Jev on the training emails

The table above caps Jev's recalibration at the 2,000-email calibration pool
while the own model gets up to 23,560 labels. To check that this isn't what
decides the comparison, Jev was also asked about all 23,560 training emails
(`run_head_to_head.py jev_train`), and each recalibrator was fitted on
exactly the training subsets the own model used (same sizes, same seeds),
then scored on the same 4,440 evaluation emails:

| Jev + …, fitted on training emails | 1,000 labels: accuracy / KL | 5,000 | 23,560 |
|---|---|---|---|
| temperature scaling | 0.788 / 0.33 | 0.788 / 0.33 | 0.788 / 0.33 |
| isotonic regression | 0.849 / 0.166 | 0.856 / 0.171 | 0.860 / 0.172 |
| logistic regression | 0.842 / 0.172 | 0.840 / 0.175 | 0.841 / 0.174 |
| *own model (same labels)* | *0.858 / 0.072* | *0.867 / 0.021* | *0.870 / 0.005* |

More labels don't help, and calibration gets *worse* than with the small
test-side pool (KL about 0.17 against 0.06), even worse than raw Jev for
temperature scaling. The cause is distribution shift from the compositional
split: the test emails use clue combinations that never appear in training,
so the mix differs (billing 0.23 of training labels vs. 0.33 of evaluation
labels; sales 0.27 vs. 0.19), and so does what Jev's answers mean. When Jev
picks "sales" it's right 64% of the time on training emails but 92% on
evaluation emails ("technical": 51% vs. 65%). A recalibrator learns
P(label | Jev's probabilities) for the mix it's fitted on, and that doesn't
transfer. The own model learns P(label | text), which does.

So the original comparison wasn't rigged against Jev. Calibrating on
emails from the same pool as the evaluation set was the *favorable*
setting for Jev. The practical lesson is to recalibrate on data that
matches deployment, and to recalibrate again when it drifts.

### The priority question (unanswerable from the text)

> **Deprecated** for the same reason as S3 (Part A): priority is unrelated
> to the text only by our hidden rule, which Jev had no way to know. Kept
> as a record; it does show that recalibration can learn a rule like that
> from a few hundred labels.

| Approach | Labeled emails used | Accuracy | Posterior KL (nats) | Mean top probability |
|---|---|---|---|---|
| *Exact posterior: the base rates (ceiling)* | — | *0.402* | *0* | *0.400* |
| Jev, as is | 0 | 0.281 | 1.29 | 0.721 |
| Jev + temperature scaling | 1,000 | 0.281 | 0.10 | 0.282 |
| Jev + logistic regression | 1,000 | 0.402 | 0.008 | 0.413 |
| Jev + isotonic regression | 1,000 | 0.401 | 0.005 | 0.413 |
| Own model | 200 | 0.35 ± 0.02 | 0.078 ± 0.023 | 0.443 |
| Own model | 1,000 | 0.377 ± 0.010 | 0.043 ± 0.014 | 0.439 |
| Own model | 23,560 | 0.402 | 0.004 | 0.411 |

(AURC is left out: every email has the same true answer, so there's no
correct ranking to measure.)

Raw Jev is confident (mean top probability 0.72) and worse than guessing (0.281 vs. 0.402), as in
S3. With 1,000 labels, isotonic or logistic regression learns that Jev's
answer carries no information here and reports the base rates. That's
exactly right, and it gets there with fewer labels than training from
scratch needs. Temperature scaling can only flatten Jev's answers toward
uniform: it stops the overconfidence (mean top probability 0.28) but can't learn
that *low* is more common than *critical*.

### What this does and doesn't show

The practical conclusion has a specific shape: **with a few thousand
labeled examples and an afternoon of work, your own model beats Jev on
every measure here. With a few hundred, the better move is to keep Jev and
recalibrate it on those labels, which is cheap and large.** Training each
own model took about 5 minutes on an M1 Max laptop.

The limits:

- **This world is easy to learn.** Short, templated emails with four labels
  and a handful of clue types. A real business's text is messier, and
  "about 1,000 labels to tie, 5,000 to win" would likely move up, perhaps a
  lot. The *pattern* (recalibrate when labels are scarce, train your own
  when they're plentiful) is the transferable part, not the exact numbers.
- **The own model isn't built from nothing.** It starts from a pretrained
  encoder (MiniLM, trained on general English), which is a big part of why
  1,000 labels go so far.
- **Jev's labels were defined only by our one-line descriptions.** Richer
  instructions, such as saying that feature requests count as sales, might
  close some of the raw gap for free. That wasn't tested.
- **One Jev version, one wording per question, one call per email.**

---

## Part D — What this means for Jev

Ranked by how much weight each can bear:

**Solid — holds regardless of our synthetic world:**

1. Jev's probabilities depend on how a question is phrased. The same
   priority question, asked three ways about the same email in one request,
   got answers 0.17 apart on average (S4).
2. Jev reports near-certainty on text that explicitly hedges: P(billing)
   ≥ 0.99 on 50 of 50 emails that say "it might be tied to a feature not
   working" (S2). More broadly, its answers pile up near 0 and 1.
3. Jev's `confidence` field is its top probability rescaled by a fixed
   formula (S5). It adds nothing, and it isn't a probability, so reading
   it as "chance of being right" under-reads Jev's claims.
4. Accuracy-only evaluations can't show whether probabilities are
   calibrated (toy model).

**Practical — solid within this world, numbers will differ elsewhere:**

5. If you need calibrated probabilities and have a few hundred labeled
   examples, recalibrate Jev on them (isotonic or logistic regression, not
   just temperature scaling). If you have a few thousand, a small model of
   your own, trained on ordinary observed outcomes, beats Jev outright
   (Part C). The crossover points depend on how hard your task is; the
   ordering doesn't.

**Suggestive — depends on how the question is read:**

6. Jev gives no signal that a question can't be answered from the text,
   and answers it confidently from surface cues (S3). Whether its answers
   there are *wrong* depends on our world's rule that priority is unrelated
   to wording.
7. A model built like Jev can learn to report base rates on such
   questions (toy model), so the S3 behavior isn't architecturally forced.

**Not supported — avoid in the post:**

- That Jev ignores or under-weights stated base rates. S6's pre-registered
  test says so, but measured against Jev's own starting belief, it
  responds about right (Part A, S6).
- That Jev is unreliable on easy emails. All of S1's misses are feature
  requests, which our answer key calls *sales*. Jev leaned *technical*,
  with moderate rather than confident probabilities (Part A, S1).
- "Mean KL 3.66 nats" without the rounding caveat. The fair figure is 0.67
  (Part A, S2).
- Anything about how Jev was trained.

**Limitations that apply to everything above.** 200 emails per test in
S1–S6 (6,440 in Part C); one
model version (`jev-1.13-20260917`), accessed through OpenRouter on a
single day; short, templated synthetic emails, not real support traffic;
questions and answer-option descriptions written by us, and different
wording could change Jev's answers; no repeat runs of the same email to
see how much Jev's answers vary from call to call.

---

## Part E — Open question: MC dropout

Everything above concerns one kind of uncertainty: the email itself being
ambiguous (*aleatoric*, in the setup's Part 4). The other kind, a model
being unsure because it hasn't seen enough examples like this one
(*epistemic*), isn't measured anywhere yet. Part 4 of the setup describes
how it could be, using MC dropout on the toy model, with an estimated cost
of 2–3 hours for a minimal version and 4–6 hours for a convincing one.
For the blog post, what it would add is a picture of what a *meaningful*
`confidence` field could look like: one that rises specifically on
unfamiliar inputs, in contrast to Jev's, which S5 shows is just its top
probability restated. Whether that's worth the time is the decision this
document is meant to inform.
