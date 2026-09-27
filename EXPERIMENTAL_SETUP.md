# Experimental setup

A from-first-principles walkthrough of how this project actually works, end
to end. `docs/explainers.md` is the math/concepts reference (what is a
proper scoring rule, what is attention); this document is the plumbing —
how the pieces in *this* repo connect, in the order data actually flows
through them.

## The idea in one paragraph

Normally, to check whether a model's probabilities are honest, you need
real-world outcomes to compare against — and those are noisy, expensive,
and never give you an *exact* answer to compare to. This project sidesteps
that by building a small synthetic world *backwards*: instead of collecting
text and then guessing at labels, it starts from the label and generates
text from it, using tables of numbers it wrote itself. Because the
direction of generation is known, Bayes' rule can be run in reverse,
*exactly*, with no estimation step — so for every piece of generated text,
the correct probability distribution over labels is a known number, not a
guess. That known number is then used twice: as the answer key for training
a small model from scratch, and as the answer key for grading the real
Jev's probabilities.

---

## Part 1 — The generator: why we already know all the answers

### The problem, restated plainly

To check whether a probability someone gives you is *correct*, you need to
already know the correct answer to compare it to. Normally, for something
like "is this email about billing," nobody knows that — you'd have to
collect a huge pile of real emails, have a person read and label every one,
and count. That's slow, it costs money, and the resulting "true"
probability is still just an estimate with its own error in it.

This project sidesteps that entirely by inventing its own tiny world, where
*we* get to decide the true probabilities ourselves, in advance, by writing
them down. We then use those numbers to manufacture emails by chance — and
because we already know every number that went into building each one, we
can calculate the exactly correct answer for it afterward, with arithmetic,
not estimate it by counting.

### Step 1: the idea in its smallest possible form

Say we only care about telling apart two kinds of email: **billing** and
**technical**. Say there's exactly one clue we're paying attention to:
whether the email contains a sentence like *"my invoice amount looks
wrong."* These aren't invented for this walkthrough — they're the real
`invoice_issue` realization from the real `complaint_phrase` table in
`jev/dgp/slots.py`, restricted here to just two of its four labels so the
example fits in your head. The table says how often that sentence shows up
in each kind of email — and *we* wrote this table, by hand, before any
email exists, so these are choices, not measurements:

- If the email is really about billing: this sentence shows up 35% of the
  time (something else shows up the other 65%).
- If the email is really about technical: this sentence never shows up —
  0% of the time.

**Generating one email.** Flip a coin to decide the email's true topic
(say 50/50 billing vs. technical, for this simplified example — the real
generator draws from four topics at 25% each). Say the
coin lands on billing. Now consult billing's row of the table above — 35%
chance of this sentence — to decide whether it actually gets written in.
Say it does. The only thing that leaves this process and becomes the
visible email is the sentence itself. The coin flip that produced it —
"billing" — is never written down anywhere in the email; it only ever
existed inside our generator, as a number on a piece of paper we're still
holding. (One implementation detail: the generator discards that top-down
draw and stores a hard label re-drawn from the exact posterior computed
below — `generate_item` in `generator.py`. The two procedures produce
identical joint distributions of topic and text, so nothing in this
walkthrough changes; it just means the stored label is always
consistent with the answer key, even for items whose clues were forced
rather than sampled.)

**Recovering the answer.** Here's the move this entire project rests on:
generating used the table in the forward direction (topic → chance of
sentence); recovering the answer runs the *exact same table*, backward
(sentence → chance of topic), for someone who only gets to read the
finished email. Concretely: for each candidate topic, ask "if this *had
been* the topic, how much weight does it get, given both the topic's own
starting odds and how well the topic explains the sentence we actually
see?" — starting odds times explanatory power, multiplied together:

```
billing:    0.5 (starting odds) × 0.35 (P(sentence | billing))    = 0.175
technical:  0.5 (starting odds) × 0.00 (P(sentence | technical))  = 0.000
```

Those two numbers, `0.175` and `0`, are relative weights, not yet
probabilities — turn them into probabilities the same way you'd turn any
set of weights into percentages: divide each by their sum (`0.175`):

```
P(billing | saw this sentence)   = 0.175 / 0.175 = 1.0
P(technical | saw this sentence) = 0.000 / 0.175 = 0.0
```

Certainty, in this case, because the technical row's number was exactly
zero — the sentence is simply impossible under "technical," so seeing it
eliminates that option outright. That's *why* the answer feels obvious
without doing any math — but the math was still there underneath, and
Step 2 is the same procedure with a case where the shortcut doesn't exist.

### Step 2: making it genuinely uncertain

Same slot, same two labels, but now look at a different realization from
that same real `complaint_phrase` table — `strong_lean`, a sentence
deliberately written to plausibly belong to either category:

- Billing emails: this sentence shows up 15% of the time.
- Technical emails: this sentence shows up 4% of the time.

These numbers have no relationship to Step 1's 35%/0% — they're a
different row of the same table, for a different sentence, chosen by
whoever wrote `slots.py` when deciding how that particular sentence should
behave. Nothing here was measured from real emails; nothing here is
derived from Step 1. Every percentage in this whole project is simply a
choice someone wrote into a table, for every slot and every realization,
independently.

Generate an email the same way as Step 1 (coin flip for the true topic,
then consult that topic's row to decide if the sentence appears), and
suppose the resulting email contains this sentence. Run the identical
recovery procedure as Step 1 — starting odds times explanatory power, for
each topic, then rescale:

```
billing:    0.5 × 0.15 = 0.075
technical:  0.5 × 0.04 = 0.020
```

Sum: `0.075 + 0.020 = 0.095`. Divide each weight by that sum:

```
P(billing | saw this sentence)   = 0.075 / 0.095 = 0.7895
P(technical | saw this sentence) = 0.020 / 0.095 = 0.2105
```

No shortcut this time — neither number was zero, so both topics stay in
play, just unevenly. This is the same mechanism as Step 1, run on a case
where it actually has work to do. The answer, `0.7895`, isn't a guess and
isn't measured from data: it falls straight out of arithmetic on two
numbers that are sitting in `slots.py` right now, unchanged by anything
this document does with them.

### Step 3: combining several clues at once — and where "slot" comes from

Real emails have more than one telling detail: what the complaint says,
what domain the sender's address is on, how urgent the writing sounds, and
so on. The generator handles several clues at once by giving *each* clue
its own table of percentages, structured exactly like the one-clue example
above, and combining them by multiplying all of their weights together —
then rescaling, exactly as before.

Multiplying is only valid if the clues don't affect each other once you
already know the true topic — i.e. if I tell you this is a billing email,
learning what domain it came from should tell you nothing extra about
whether some *other* clue, like tone, will show up. This project builds
its generator so that's true by design: every clue is decided by its own
separate coin flip, weighted according to the true topic, with no clue
influencing any other. That's a deliberate construction choice, not
something being tested — and it's exactly what makes plain multiplication
the correct way to combine them, rather than something more complicated.

Now the vocabulary this project actually uses — the same names as the
identifiers in `slots.py` and `generator.py`, so you can grep for any of
them. Each one names a step in the chain that builds an email, in the
order the generator runs it:

1. A **concept** is one question the email answers — one hidden variable.
   There are four: `category`, `urgency`, `is_business`,
   `hidden_priority`. Every email answers all four at once, each drawn
   independently of the others. The "topic" in Steps 1–2 is the
   `category` concept.
2. A **label** is one possible answer to a concept — `billing`,
   `technical`, `sales`, `spam` for `category`. Each concept draws exactly
   **one** label per email, from its own prior (its starting odds).
3. A **slot** is one clue that belongs to a concept — e.g.
   `complaint_phrase` ("what the complaint sentence says") or
   `sender_domain` ("what domain the email is from"). `category` has three
   slots, `urgency` two, `is_business` one, `hidden_priority` none. Each
   slot is one of the percentage tables from Steps 1–2: one row per label,
   giving the chance of each possible value under that label.
4. A **realization** is the value a slot actually took on for one email,
   drawn from the row of that slot's table for the concept's label. In
   Steps 1 and 2, `invoice_issue` and `strong_lean` were two of
   `complaint_phrase`'s twelve realizations. Every slot draws a realization
   on every email — some realizations, like `complaint_phrase`'s `none`,
   just render as nothing.
5. A **phrase** is the actual text that appears in the email. Each
   realization has a small bank of interchangeable wordings
   (`strong_lean` has three), and the renderer picks one uniformly at
   random, *without seeing the label*. This step adds variety to the text
   but no information: two emails with the same realizations get the same
   answer key no matter which wordings came up.

So for the worked example below: concept `category` → label (hidden) →
slot `complaint_phrase` → realization `strong_lean` → phrase *"I think I'm
being charged incorrectly, though it might be tied to a feature not
working as expected."* The label is drawn once per concept, **not** once
per slot: all three of `category`'s slots are drawn based on the same
single label. `jev/dgp/slots.py` is nothing but the slot tables and phrase
banks, written out by hand; the concepts and their priors are in
`generator.py`.

Putting several slots together, the full rule is the Step-2 recipe done
once per slot, then all multiplied in: start from the topic's overall
starting chance, then multiply in — once for *each* slot — the chance that
this topic would have produced the specific realization that came up in
that slot for this email. Do that for every candidate topic, then rescale
so they all add up to 100%. In the compact notation you'll see in the code
and in `docs/explainers.md`, that's:

```
P(label | realizations) ∝ prior(label) × Π_slots P(realization_slot | label)
```

(`Π` just means "multiply all of these together," one factor per slot; `∝`
means "proportional to" — i.e. this is the weight *before* the final
rescaling-to-100% step.) `dgp/generator.py`'s `posterior()` function is
this rule, and nothing else, run in code instead of by hand. A model
trained this way (weights multiplied from independent per-clue tables) is
classically called "naive Bayes" — naive specifically because it assumes
the clues don't influence each other, the same assumption from two
paragraphs up.

### A real, concrete example, fully worked

This is an actual item the generator produced (the 9th item from
`EmailGenerator(seed=42)`):

```
From: emma.berg@yahoo.com
Subject: Quick note

Thanks so much for your help.
I think I'm being charged incorrectly, though it might be tied to a
feature not working as expected! URGENT -- please treat as a priority.

Thanks,
Emma
```

Six slots, across three concepts, drew realizations to build this email.
Each piece of text traces back to exactly one of them:

| Concept | Slot | Realization | Phrase in the email |
|---|---|---|---|
| `category` | `tone` | `polite` | "Thanks so much for your help." |
| `category` | `complaint_phrase` | `strong_lean` | "I think I'm being charged incorrectly, … as expected" |
| `category` | `sender_domain` | `generic_webmail` | "@yahoo.com" |
| `urgency` | `exclaim_count` | `1` | the "!" ending the complaint sentence |
| `urgency` | `intensity_marker` | `extreme` | "URGENT -- please treat as a priority." |
| `is_business` | `signature_style` | `first_name_only` | "Thanks, Emma" (on two lines) |

(`hidden_priority` has no slots, so it contributes no text. The name,
"Emma Berg," and the subject line are random decoration drawn from a
separate random stream that never sees any label.)

To answer the `category` question, only `category`'s own three slots
matter. The urgency and business slots were drawn based on those concepts'
own labels, which are independent of the category label, so they carry no
information about the category. (The other
concepts' posteriors are computed the same way from their own slots.) The
three relevant realizations, and each label's chance of producing that exact
realization (read straight out of the tables in `slots.py`):

```
complaint_phrase = "strong_lean"       billing 0.15   technical 0.04   sales 0.01   spam 0.01
sender_domain    = "generic_webmail"   billing 0.30   technical 0.30   sales 0.50   spam 0.75
tone             = "polite"            billing 0.30   technical 0.30   sales 0.30   spam 0.30
```

(The `tone` slot's realization has the *same* chance, 0.30, under every
label — a clue that doesn't discriminate at all, on purpose. It'll cancel
out of the final ratio below; a slot can be part of the story and still
contribute nothing.)

With a uniform starting chance (25% each), multiply each label's three
numbers together, exactly per the rule above:

```
billing:   0.25 × 0.15 × 0.30 × 0.30 = 0.0033750
technical: 0.25 × 0.04 × 0.30 × 0.30 = 0.0009000
sales:     0.25 × 0.01 × 0.50 × 0.30 = 0.0003750
spam:      0.25 × 0.01 × 0.75 × 0.30 = 0.0005625
```

Sum: `0.0033750 + 0.0009000 + 0.0003750 + 0.0005625 = 0.0052125`. Divide
each by that sum (these are the exact values — worth using enough decimal
places to check by hand, since rounding an intermediate step here would
throw off the last digit of the final answers below):

```
P(billing)    = 0.0033750 / 0.0052125 = 0.6475
P(technical)  = 0.0009000 / 0.0052125 = 0.1727
P(sales)      = 0.0003750 / 0.0052125 = 0.0719
P(spam)       = 0.0005625 / 0.0052125 = 0.1079
```

That vector — a full distribution, not a single "correct" label — is the
ground truth this project trains against and grades Jev against. Nobody
labeled this email. Every one of the four numbers above is a direct,
checkable consequence of which realizations happened to come up, which is
a direct consequence of the tables in `slots.py` — arithmetic, start to
finish.

### Three families: different combinations picked from the same fixed tables

The tables never change between an "easy" item and a "hard" one. What
changes is *which combination of realizations* the item gets. Because
each concept has only a few slots, each with a handful of realizations,
every possible combination can be listed and its exact posterior
computed (144 combinations for `category`). `jev/dgp/scenarios.py` builds
each family by picking combinations from that list and forcing the
generator to use them (the `overrides` argument to `generate_item`),
instead of letting it sample freely:

| Family | Which combinations are used | Resulting posterior |
|---|---|---|
| Resolvable | those whose top label has posterior ≥ 0.97 — typically because a realization has zero chance under every other label (like `invoice_issue` in Step 1) | near one-hot |
| Ambiguous | the combination whose posterior is closest to a chosen target (like `strong_lean` in Step 2) | an intermediate value from the achievable grid (not arbitrary, since only 144 combinations exist) |
| Unanswerable | none needed — the `hidden_priority` concept has **no slots at all** | exactly the prior, for every input |

`hidden_priority` is the sharpest of the three: it's decided by its own
coin flip that never influences, and is never influenced by, any slot — so
*nothing* in the rendered text carries any information about it, by
construction. Its posterior given any text is always exactly
`{low: 0.4, medium: 0.3, high: 0.2, critical: 0.1}` — the starting chance,
unconditionally, no matter what the email says. This is what suites S3/S4
grade Jev against.

### The numbers, so this is checkable rather than asserted

(from `jev.dgp.generator`, current code, computed exactly — not sampled)

| Concept | Labels | Slots | Bayes error rate | Distinct evidence combos |
|---|---|---|---|---|
| `category` | billing/technical/sales/spam | 3 | 0.221 | 144 |
| `urgency` | low/medium/high/critical (ordinal) | 2 | 0.440 | 16 |
| `is_business` | business/personal | 1 | 0.300 | 3 |
| `hidden_priority` | low/medium/high/critical | **0** | 0.600 | 1 |

"Bayes error rate" is the error rate of the *best possible* classifier on
this exact problem, even one that knows every table — the irreducible
error left over after using the evidence as well as it can possibly be
used. It's a hard ceiling here, not a statistical estimate of one, because
we built the world and can compute it directly rather than approximate it.
Lower-bound on distinct literal email texts the generator can produce:
**208,051,200** — comfortably more than any training run here uses, which
is what makes memorizing the dataset implausible rather than merely
unlikely.

### Why this isn't just "a labeled dataset"

A normal dataset gives you `(text, label)` pairs where the label was
assigned by a person, a heuristic, or another model — every one of those is
a point estimate with unknown error. This generator gives you
`(text, exact P(label|text))` pairs where the second element is *provably*
correct given the tables, because the tables are the entire generative
process, not an approximation of one. Two things guard against this being
too-good-to-be-true by accident (both are automated tests, not just
claims):

- **Paraphrase can't leak the label.** The function that turns a
  realization into an actual sentence only ever receives the realization
  itself, never the label — structurally, not by convention, so it has
  nothing to leak. `tests/test_generator.py` checks the function's
  signature directly, to catch anyone accidentally adding a label
  parameter later; `tests/test_scenarios.py` backs it up empirically (a
  classifier trained on 400 items that all share the exact same
  realizations gets 52.5% accuracy predicting the label from raw text,
  against a 52.0% majority-class baseline — indistinguishable from chance,
  exactly as it should be since every one of those items has identical
  evidence).
- **Nothing can beat the Bayes rate.** Since it's computed exactly, "a
  model scored below the Bayes error rate" is not a good result, it's a
  bug report. Checked in tests, both on the generator's own oracle
  predictor and, implicitly, on every real result this project reports.

---

## Part 2 — How the models are trained and tested against the table

One clarification first: the model trained in `toy/` is **not an LLM**. It
doesn't generate text and has no notion of a next token. It's a small
pretrained *encoder* (`all-MiniLM-L6-v2`, ~22M params — the same category
of architecture as BERT, not GPT) with a handful of classification-style
heads bolted on top — the same broad shape TypeSafe claims for Jev itself
("System One": read the whole input, answer typed questions, no
generation). See `docs/explainers.md`'s encoder-vs-decoder section for why
that distinction is the whole reason this kind of model can have a
well-formed probability at all.

### The forward pass (this is identical whether training or evaluating)

```
  text
   |
   v
tokenizer (WordPiece)
   |
   v
encoder (MiniLM, one pass)  -->  token vectors
   |
   +---------------+---------------+---------------+
   |               |               |               |
 category query  urgency query  is_business query  hidden_priority query
   |               |               |               |
 cross-attn      cross-attn      cross-attn       cross-attn      <- one small
 pooling         pooling         pooling          pooling            matmul each,
   |               |               |               |                 not another
   v               v               v               v                 encoder pass
 choice head    ordinal head    noul head        3 heads (noul/
   |               |               |             choice/score)
   v               v               v               v
predicted       predicted       predicted        predicted
distribution    distribution    scalar           distributions
```

The probability *table* (the emission tables from Part 1) never enters this
diagram. The model never sees `slots.py`. All it ever sees is rendered
text — exactly the same information Jev gets. The table only shows up
**outside** this diagram, in one place: the loss function.

### Where the table and the model actually meet

For every training item, `dgp/generator.py` already computed the exact
posterior (Part 1). `toy/data.py` attaches that full vector to the item as
a training target, right alongside the tokenized text. `toy/train.py`'s
loss function is where the two things meet:

```
soft_xent(predicted_distribution, true_posterior_vector) =
    -sum_label true_posterior[label] * log(predicted_distribution[label])
```

This is ordinary cross-entropy — except the target is a **distribution**
(`[0.65, 0.17, 0.07, 0.11]` for billing/technical/sales/spam, the worked
example above), not a single class index. The model is
literally being shown the exact table-derived answer, every step, and
graded on how close its own distribution comes to matching it — a proper
scoring rule (`docs/explainers.md`) applied to a target that happens to be
exactly correct rather than merely observed.

### The other half of the ablation: what "hard" training sees instead

`toy/data.py` also stores a **hard** label for every item — one single
class, sampled from that same true posterior (so over many items, the hard
labels' frequencies still match the table, but any *one* item's hard label
throws away the rest of the distribution). `hard_xent` trains against that
single sampled index instead:

```
hard_xent(predicted_distribution, sampled_label_index) =
    -log(predicted_distribution[sampled_label_index])
```

Same architecture, same data, same optimizer, same seeds where it matters —
the *only* difference between the "soft" and "hard" experiment runs is
which of these two loss functions is called. That difference is the entire
toy-model experiment in this repo: does seeing the full table-derived
distribution during training produce a better-calibrated model than seeing
only one sampled draw from it? (Answer, from the multi-seed run: yes,
sharply, on posterior-KL — see `README.md`.)

### The same table, reused a second time, for evaluation

`toy/train.py`'s `evaluate()` and `probe/suites.py` both do the same thing
with the table: take a batch of items, get the generator's exact posterior
for each, get the model's (or Jev's) predicted distribution for each, and
compare — via ECE, the Brier decomposition, or posterior KL
(`eval/metrics.py`, `docs/explainers.md`). The training-time use and the
evaluation-time use are the same data, doing two different jobs: one is a
loss to descend, the other is an answer key to grade against.

### The toy experiment's protocol

`scripts/run_toy_experiment.py`:

- **Data.** One pool of 30,000 generated items (`data_seed=100`),
  generated once and reused by every training run. Split 23,560 train /
  6,440 test. The split is *compositional* on `category`
  (`scenarios.compositional_split`): each combination of `category`
  realizations goes entirely to one side, so every combination in the test
  set is one the model never trained on. It has to generalize to new
  combinations rather than memorize the ones it saw.
- **Training.** Each mode (soft, hard) is trained with 3 seeds
  (100/101/102) for 3 epochs, batch size 32. Learning rates are 2e-5 for
  the encoder and 2e-4 for the heads. Nothing differs between the modes
  except the loss function.
- **Evaluation sets.** Every trained model is scored on four sets: the
  held-out test split, plus 1,000 items each from the three families in
  Part 1. Those are *resolvable* (top posterior ≥ 0.97), *unanswerable*
  (ordinary items, where `hidden_priority` is automatically unanswerable)
  and *ambiguous* (a single forced combination with P(sales or spam)
  ≈ 0.6).
- **Metrics**, per concept head: accuracy against the sampled hard label,
  ECE, AURC, and mean posterior KL from the exact posterior.
- **Negative control.** One extra hard-mode run, 2 epochs, with the
  `category` labels randomly shuffled. It must score at chance.

### The live probe: six suites against the real Jev

`jev/probe/suites.py` sends generated emails to the real Jev API (via
`probe/client.py`) and grades the answers against the exact posterior.
Jev gets exactly what a user would give it: the rendered email text, plus
a typed question — a type (`choice`, `score` or `noul`), a one-line
instruction, and a short description of each answer option. For example,
the `category` question is *"Classify this support email into exactly one
category"*, with options such as billing = *"About invoices, payments,
refunds, or subscription costs."* Jev never sees `slots.py`,
realizations, or priors, and it was never trained on this synthetic world.
It is being tested cold.

Each suite's prediction and the result that would falsify it were written
into `HYPOTHESES` in `suites.py` *before* the live run. S2 and S3 were
designated the primary endpoints; S1, S4, S5 and S6 are exploratory. Each
suite uses 200 items.

| Suite | Tests | Items (built with Part 1's family builders) | Question(s) asked | Compared against | Pre-registered prediction | Falsified if |
|---|---|---|---|---|---|---|
| **S1** Resolvable | Is Jev right and confident when the answer is clear? | `category` combinations whose top posterior ≥ 0.97 | `category` (choice) | sampled hard label | accuracy and mean confidence both > 0.9 | either falls below 0.85 |
| **S2** Known-ambiguous (primary) | Does Jev report intermediate probabilities as intermediate? | a *sweep*: for target P(billing) = 0.3, 0.5, 0.7, 0.9, the closest achievable combination (actually 0.345, 0.533, 0.714, 1.0), 50 items each | `category` (choice) | full exact posterior vector (posterior KL) | mean KL < 0.15 nats | mean KL > 0.3 nats, or errors all in one direction |
| **S3** Unanswerable (primary) | Does Jev admit it can't know? | ordinary items; asks about `hidden_priority`, which has no evidence in the text | "What priority level should this ticket be assigned?" (choice) | hidden_priority's sampled label; best achievable accuracy = 0.40 (always guess `low`) | accuracy ≤ 0.40 *but* mean confidence above 0.40 — i.e. overconfident | mean confidence within 0.05 of 0.40 |
| **S4** Type consistency | Do different question types about the same thing agree? | same as S3 | `hidden_priority` asked three ways *in one request*: noul "should this be expedited (high or critical)?", choice (sum P(high) + P(critical)), score (same sum) | exact P(expedite) = 0.2 + 0.1 = 0.30 | the three disagree by > 0.1 | all three agree with each other and 0.30 within 0.05 |
| **S5** Confidence field | Does Jev's `confidence` carry information beyond max(probabilities)? | 100 resolvable (top posterior ≥ 0.9) + 100 ambiguous (P(billing) ≈ 0.6), so correctness varies | `category` (choice) | AURC of each signal; correlation between them | no meaningful AURC advantage; highly correlated | `confidence` clearly better AURC, or weakly correlated |
| **S6** Prior sensitivity | Does Jev use a stated base rate correctly? | one ambiguous combination, P(spam) ≈ 0.5 under the true prior | `category` asked twice per email, with the instruction prefixed *"historically, only about 10% … are spam"* vs *"about 60% … are spam"* | Bayes-correct shift: same evidence re-run through the Part 1 rule with each stated prior swapped in | Jev shifts by *less* than the Bayes-correct amount | shift matches Bayes-correct within 0.1 |

Two notes on reading the table. In S1 and S3, "accuracy" means agreement
with the *sampled* hard label, so even a perfect reader is capped by the
Bayes error rate. That is why S3's yardstick is 0.40, not 1.0. And S2 is
the only suite that grades Jev's full probability vector against the full
exact posterior. The others grade a chosen label, one derived probability,
or a shift.

---

## Part 3 — Results

Two experiments used the same answer key. The toy model (Part 2) asks
whether training on the exact posterior produces a better-calibrated model
than training on one sampled label. The live probe asks how close the real
Jev's probabilities come to that same exact posterior. Every number below
is read directly from `outputs/` — `toy_experiment/summary.json` and
`probe_results.json` — not from memory or from the README's prose.

### Toy model: soft vs. hard training

Protocol as in Part 2 ("The toy experiment's protocol"). Values are
mean ± std across the three training seeds. Where a std is omitted, it
rounds to zero at the precision shown.

**The headline, on the `category` head:**

| Eval set | Accuracy (soft) | Accuracy (hard) | Mean KL from truth (soft) | Mean KL (hard) | Hard ÷ soft |
|---|---|---|---|---|---|
| Held-out | 0.866 ± 0.001 | 0.869 ± 0.002 | 0.00016 | 0.00331 ± 0.0003 | 20× |
| Ambiguous | 0.498 ± 0.000 | 0.498 ± 0.000 | 0.00039 ± 0.0001 | 0.00936 ± 0.0003 | 24× |
| Resolvable | 1.000 | 1.000 | 0.00010 | 0.00097 ± 0.0002 | 10× |
| Unanswerable family | 0.761 ± 0.001 | 0.759 ± 0.001 | 0.00022 | 0.00536 ± 0.0002 | 24× |

"Mean KL from truth" is the posterior KL divergence from `eval/metrics.py`,
in nats. It measures how far the model's whole predicted distribution is
from the exact posterior, item by item. Zero means a perfect match, and
unlike ECE it needs no binning.

**What this shows.** The two training modes are indistinguishable on
accuracy, and on AURC (held-out: 0.0174 vs 0.0176), which measures how well
a model ranks its confident predictions above its unsure ones. They
differ by 10–24× on how close their stated probabilities are to the true
ones. This is the reliability/resolution split from the Brier
decomposition (`docs/explainers.md`), seen empirically. Both models *rank*
items equally well; only the soft-trained one reports the right *numbers*.
The same pattern holds on the other two heads that fit well:
`is_business` (hard ÷ soft ≈ 27–43×) and `hidden_priority` (≈ 115–160×),
though the absolute KLs there are tiny for both modes (under 0.003).

**Why accuracy is *exactly* equal on the ambiguous set.** That set uses a
single forced combination of realizations, so every item has the same
evidence and the same true posterior. Any model that puts its highest
probability on the right label will pick the same answer for all 1,000
items, and its accuracy is then just the fraction of sampled hard labels
that happen to match that answer: 0.498, for both modes, on every seed.
Accuracy cannot see the difference between a model saying 0.51 and one
saying 0.99 here; posterior KL can (24×). This is the clearest single
illustration in the project of why accuracy alone is the wrong yardstick
for calibration.

**`hidden_priority` (unanswerable by construction): both modes got it
right.** The best possible strategy is to report the prior and guess
`low`. Both modes do exactly that: accuracy is identical between them on
every eval set (0.404 held-out, 0.400 unanswerable family, 0.386
ambiguous — the variation is only which hard labels happened to be
sampled into each set). Soft training's KL from the prior is essentially
zero (0.00002). Hard training's is small but consistently larger
(≈ 0.0024) across every seed.

**Accuracies slightly above the "ceiling" are not a bug.** A couple of
accuracies edge past Bayes-optimal: `is_business` held-out 0.7095 against a
ceiling of 0.700, and `hidden_priority` 0.4037 against 0.400. Part 1's
Bayes error rate is an *expected* error rate. On a finite test set, the
sampled hard labels can land slightly luckier than expected. For
`is_business` on 6,440 items that's about 1.7 standard errors, well within
normal sampling noise. A model beating the ceiling *on average across many
fresh samples* would be a bug; one test set doing so is not.

**Honest limitation — `urgency` underfit.** Held-out accuracy 0.48 against
a Bayes-optimal 0.56 (Bayes error 0.440 from Part 1), and a mean KL of
≈ 0.20 for *both* modes, two orders of magnitude worse than `category`.
Soft vs. hard differ only slightly (0.2012 vs 0.2076). The likely cause is
the ordinal head, which describes all four urgency levels with a single
score, and probably can't represent this posterior shape well. More
epochs might also help. Treat any urgency-specific conclusion as
unsupported until this is fixed.

**Negative control passed.** Retraining on the same data with `category`
labels randomly shuffled gives 0.236 held-out accuracy, against 0.25 for
random guessing. The pipeline learns nothing when there's nothing to learn,
as it must. (This was a single run: it's a sanity check, not a headline
claim.)

**A caveat on ECE.** ECE, the more familiar calibration metric, also
favors soft (held-out `category`: 0.0048 vs 0.0103). But ECE depends on
how predictions are grouped into bins. The reliability-diagram PNGs in
`outputs/toy_experiment/` use 10 bins with equal numbers of items; the
table above uses 15 bins of equal width. So the two report different ECE
values for the same model. Posterior KL is the binning-free comparison and
the one to trust.

### Live probe: the real Jev, graded against the exact posterior

Suites as described in Part 2 ("The live probe: six suites against the
real Jev"). Model `typesafe/jev-1.13-20260917`, reached through OpenRouter
on 2026-09-26. The probe was run twice. The first run exposed a gap in
the `complaint_phrase` table: nothing in it could produce a P(billing)
between 0.37 and 1.0. So the `mixed_lean` and `strong_lean` realizations
were added (the same `strong_lean` used in Part 1's Step 2), and the probe
was re-run. The numbers below are from the second run. The first is kept
at `outputs/probe_results_run1_20260925.json`.

| Suite | Question put to Jev | Exact truth | What Jev said | Verdict |
|---|---|---|---|---|
| **S2** (primary) | `category`, on 4 combinations whose P(billing) is 0.345 / 0.533 / 0.714 / 1.0 | those values | see table below; mean KL **3.66 nats**, median 3.20 | **Prediction falsified** (needed KL < 0.15) |
| **S3** (primary) | `hidden_priority` as a choice question | prior; best achievable accuracy 0.40 | accuracy **0.265** (95% CI 0.209–0.330), mean confidence **0.672** | **Confirmed**: overconfident on an unanswerable question |
| S1 | `category`, resolvable items (top posterior ≥ 0.97) | near one-hot | accuracy 0.91 (CI 0.862–0.942), mean confidence 0.943, ECE 0.061 | Confirmed: good on easy items |
| S4 | `hidden_priority` "expedite?" asked 3 ways (noul / choice / score) | P = 0.30 for all three | 0.288 / 0.366 / 0.363 on average; per item, noul differs from choice by 0.167 on average | Confirmed: framings disagree |
| S5 | does `confidence` add anything beyond max(probabilities)? | — | correlation 0.9998; AURC 0.1339 vs 0.1341 | Confirmed: `confidence` ≈ max probability |
| S6 | same email, stated spam base rate changed from 10% to 60% | Bayes-correct shift in P(spam): +0.568 | shifted in the right direction on 200/200 items, but by only +0.389 on average | Confirmed: underweights stated priors |

**S2 in detail.** This is the result that needs an exact, adjustable
posterior to see at all. For each combination, 50 items, comparing
Jev's average reported P(billing) against the truth:

| True P(billing) | Jev's mean reported P(billing) | Direction |
|---|---|---|
| 0.345 | 0.011 (18 of 50 exactly 0) | far too low |
| 0.533 | 0.307 | too low |
| 0.714 | **0.997** | far too high |
| 1.000 | 1.000 | correct (trivially) |

Jev doesn't report the intermediate value. It pushes it toward whichever
extreme is nearer, with the switch from too-low to too-high somewhere
between true P = 0.53 and 0.71 (4 levels × 50 items locate a range, not
an exact switch point). This "extremizing" is the pattern the first run
*couldn't* show: with no combination between 0.37 and 1.0 available, only
the too-low half was visible. Adding `strong_lean` / `mixed_lean` didn't
just add data points; it revealed errors in the opposite direction that
the narrower table couldn't reach.

**S3 in detail.** On `hidden_priority`, the text contains no information
about the answer, by construction (Part 1). The best any reader can do is
report the prior and guess `low`, which is right 40% of the time. Jev did
worse than that (26.5%, with the entire confidence interval below 0.40)
while reporting 67% confidence. That's an overconfidence gap of +0.27 on a
question it cannot, even in principle, answer from the text. This
replicates the independently published result (44.7% accuracy, 0.74
confidence) on a separately built, provably unanswerable item with a
different prior.

### Reading the two together

The toy model, trained with soft targets on this world, reports the prior
for `hidden_priority` almost exactly; Jev reports 67% confidence. So that
failure isn't inevitable for a one-pass encoder with typed heads. With
clean training, a model this shape handles an unanswerable question
correctly. **This is not a like-for-like comparison, though.** The toy
model was trained on this exact synthetic world, and Jev never saw it.
The fair conclusion is "Jev's behavior here isn't forced by its
architecture," not "Jev was trained with hard labels." The toy results
support soft-target training as *one* mechanism that produces good
calibration. They don't identify what Jev's training actually did.

Similarly, the toy model's KL numbers (≈ 0.0002–0.009) and Jev's S2 KL
(3.66) are on the same scale but not comparable as a score: one model is
in-distribution and the other is being tested cold. What *is* comparable
is the shape. Both of the toy models, trained in-distribution, report
intermediate probabilities as intermediate. Jev pushes them toward the
extremes.

---

## Part 4 — Uncertainty: what's built, and what's only designed

Two different kinds of "the model isn't sure," worth keeping strictly
separate (`docs/explainers.md` has the formal definitions):

- **Aleatoric** — the world itself is ambiguous. More data about this
  exact input wouldn't resolve it, because the generator built genuine
  overlap into the tables on purpose.
- **Epistemic** — the *model* hasn't learned this region well. More
  training data would fix it.

### Aleatoric: fully built, and it's most of this project

The exact posterior from Part 1 *is* the aleatoric signal, by construction.
Every `probabilities` number this project produces or grades — the toy
model's softmax output, Jev's `choice`/`score`/`noul` answers — is being
compared against this. This part is not a design sketch; it's the training
loss and every metric in `eval/metrics.py`.

### Epistemic via MC dropout: designed in the plan and explainer, not implemented in code

This is worth being precise about, because it was described in
`docs/explainers.md` and the original plan as part of the approach, and
it's easy to read that as "done." It wasn't built. Checked directly against
the code, not from memory:

- `jev/toy/model.py` adds **no dropout layers** of its own to any head
  (the cross-attention pooling, the ordinal head, the linear heads are all
  plain, deterministic layers).
- The underlying MiniLM encoder *does* have dropout internally
  (`hidden_dropout_prob=0.1`, `attention_probs_dropout_prob=0.1` —
  standard BERT-style config, inherited from the pretrained checkpoint,
  not something this project added).
- But dropout only injects randomness when the module is in `.train()`
  mode. Every evaluation path in this repo (`evaluate()`, `predict_all()`)
  calls `model.eval()` first, which **turns that dropout off**. So today,
  a forward pass at evaluation time is fully deterministic — two calls
  with the same input give bit-identical output. There is currently no
  mechanism anywhere in this codebase that produces multiple different
  stochastic predictions for the same input, which is the one thing an
  epistemic-uncertainty signal needs to exist at all.

**Why it would be worth building** (the use case): the generator's exact
posterior can only ever capture aleatoric uncertainty — by construction, it
is a fact about the world, not about what the model has learned, so it
cannot serve as ground truth for epistemic uncertainty the way it does for
calibration. To actually demonstrate a model that separates the two, you
need a positive control: deliberately hide a coherent region of data from
training, then confirm a real uncertainty signal (MC-dropout spread) is
*elevated* specifically there. That would give a template for what a
`confidence` field doing its job should look like — a sharper basis for
interpreting Jev's own `confidence` field, which suite S5 already found is
empirically almost indistinguishable from `max(probability)` (correlation
0.9996–0.9998) without being able to say *what it's failing to be*.

**What building it would take** (not done, estimated, two tiers):

- *Minimal (~2–3 hours):* run inference with the model kept in `.train()`
  mode (so the encoder's existing 0.1 dropout stays active), N stochastic
  forward passes per item, measure the spread. Reuses the existing
  `compositional_split` held-out set rather than a deliberately-chosen
  region. Weaker signal (only the encoder's inherited dropout contributes;
  the held-out set is novel slot combinations, not a coherent "unlearned
  region"), but cheap.
- *Fuller (~4–6 hours):* add explicit dropout to the heads for a stronger
  signal, deliberately exclude one describable region of latent space from
  training (e.g. "never see the `strong_lean`/`mixed_lean` realizations"),
  and compare MC-dropout spread there against a size-matched *random but
  covered* control — the control is what turns "a number" into "a claim,"
  since without it there's no way to tell elevated spread apart from
  ordinary held-out noise.

### What this project did instead: a cruder, related, but genuinely different thing

The multi-seed toy runs (3 independently-trained models per mode, different
weight initialization and data-shuffling order, same architecture and data)
are an **ensemble**, not MC dropout. The distinction matters: MC dropout
asks "how much does *one trained model's* answer wobble under noise";
training-seed ensembling asks "how much do *several independently-trained
models* agree with each other." Both are legitimate proxies for epistemic
uncertainty in the literature, but they're not interchangeable, and this
project only ran the second one — and even there, only to get a mean±std
on accuracy/calibration *metrics* for robustness (`scripts/
run_toy_experiment.py`), not to compute a per-item disagreement score as an
uncertainty signal the way an epistemic-uncertainty experiment properly
would.

**Bottom line:** if the post says anything about epistemic uncertainty or
MC dropout, it should say it's the designed-but-not-built half of the
uncertainty story, not report numbers for it — there aren't any.

---

## Where to look in the code

| What | File |
|---|---|
| Emission tables (the actual numbers) | `jev/dgp/slots.py` |
| Exact posterior arithmetic, sampling, rendering | `jev/dgp/generator.py` |
| Item families, splits, negative controls | `jev/dgp/scenarios.py` |
| ECE, Brier decomposition, posterior KL | `jev/eval/metrics.py` |
| The toy model's architecture | `jev/toy/model.py` |
| Soft vs. hard loss, training loop | `jev/toy/train.py` |
| Item → tokenized tensors + both targets | `jev/toy/data.py` |
| The real soft-vs-hard experiment | `scripts/run_toy_experiment.py` |
| Jev API client (live/replay/fixtures) | `jev/probe/client.py` |
| The six probe suites, S1–S6 | `jev/probe/suites.py` |
