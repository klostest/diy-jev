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

## Part 2 — How the probability table and the model come together

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

---

## Part 3 — Uncertainty: what's built, and what's only designed

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
