# Experimental setup

A from-first-principles walkthrough of how this project actually works, end
to end. `docs/explainers.md` is the math/concepts reference (what is a
proper scoring rule, what is attention); this document is the plumbing —
how the pieces in *this* repo connect, in the order data actually flows
through them. It covers the whole design — including exactly how every
result is measured (Part 3) — so that `RESULTS_AND_DISCUSSION.md` can be
read without introducing anything new.

> **On `$` in this file:** following `docs/explainers.md`, `$...$` in
> Parts 2 and 3 are real LaTeX math delimiters, a deliberate exception to the
> project's convention of escaping dollar signs. A literal currency amount
> would still be escaped (\$).

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
- **Nothing can beat the Bayes rate.** Since it's computed exactly, a
  model whose *expected* error rate is below the Bayes error rate is not a
  good result, it's a bug report. (On one finite test set, a model can
  land slightly below it by luck in which hard labels were sampled; the
  ceiling applies to the long-run average.) Checked in tests on the
  generator's own oracle predictor.

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
 choice head    choice head     noul head        3 heads (noul/
   |               |               |             choice/score)
   v               v               v               v
predicted       predicted       predicted        predicted
distribution    distribution    scalar           distributions
```

**Step by step** (code in `jev/toy/model.py`):

1. **Encoder, once.** The email is tokenized (at most 64 tokens) and passed
   once through the encoder, a 6-layer BERT-style network with
   384-dimensional vectors, pretrained and then fine-tuned here. As in
   BERT, the output is one context-enriched vector per token,
   $H = [h_1, \dots, h_N]$ with each $h_t \in \mathbb{R}^{384}$. Nothing
   after this step looks at the text again.
2. **One summary vector per question, by attention pooling.** To classify,
   the model needs a single vector, not $N$ of them. Common choices are
   BERT's `[CLS]` vector, or the average of all token vectors (how this
   encoder was originally trained to make sentence embeddings). This model
   instead gives each question its own **learned query vector**: four
   trainable 384-dimensional parameters, $q_{\text{category}}$,
   $q_{\text{urgency}}$, $q_{\text{is\_business}}$ and
   $q_{\text{hidden\_priority}}$, which don't come from any text. For
   question $c$, one standard attention step uses $q_c$ as the query and
   the token vectors as keys and values. Written first as if there were a
   single attention head:

   $$
   \alpha_t = \text{softmax}_t\!\left(\frac{(q_c W_Q)(h_t W_K)^{\top}}{\sqrt{d}}\right),
   \qquad
   z_c = \sum_{t} \alpha_t \, h_t W_V
   $$

   The question's vector scores every token for relevance, the scores
   become weights summing to 1, and $z_c$ is the resulting weighted blend of
   tokens. Padding tokens are masked out.

   **With 4 heads, and the output projection $W_O$.** The layer actually
   runs 4 of these side by side. Head $h$ has its own slices
   $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}$ of the projections, mapping the
   384-dimensional vectors down to $d = 96$ dimensions, and computes its own
   weights $\alpha^{(h)}_t$ and its own blend $z_c^{(h)}$ (96 numbers). The
   four blends are then placed side by side and mixed by one more learned
   matrix, the **output projection** $W_O$ (384 × 384):

   $$
   \alpha^{(h)}_t = \text{softmax}_t\!\left(\frac{(q_c W_Q^{(h)})(h_t W_K^{(h)})^{\top}}{\sqrt{96}}\right),
   \qquad
   z_c^{(h)} = \sum_{t} \alpha^{(h)}_t \, h_t W_V^{(h)},
   \qquad
   z_c = \big[\, z_c^{(1)};\, z_c^{(2)};\, z_c^{(3)};\, z_c^{(4)} \,\big]\, W_O
   $$

   Because each head has its own weights, the heads can attend to
   *different* tokens in the same email. After concatenation each head's
   findings sit in their own block of 96 numbers; $W_O$ lets every output
   dimension draw on all four heads at once (for example, combining "found
   an intensity phrase" with "found exclamation marks") and returns the
   result to 384 dimensions. (With a single head, $W_O$ would be optional,
   since the output layer after it could absorb it.) In PyTorch's
   `nn.MultiheadAttention` it's called `out_proj`.

   All the projections, $W_Q, W_K, W_V$ and $W_O$, are shared by the four
   questions; only the query vectors differ. So each question learns to
   *look at different tokens*, and the weights are recomputed for every
   email, so it finds those tokens wherever they appear.
   `scripts/show_attention.py` prints the weights for a few emails: one
   urgency head puts nearly all its weight on the intensity phrase
   ("priority", "urgent" or "rush", whichever the email has) and another
   on the "!". The category heads spread over the complaint sentence, the
   business heads go to the signature, and the `hidden_priority` heads
   spread thin over boilerplate, since there's nothing there to find. The
   one set of token vectors is read four ways.
3. **Output heads.** Each $z_c$ goes through a small linear layer: a
   softmax over labels for `category` and `urgency`, a sigmoid for the
   yes/no `is_business`, and three views (yes/no, choice, score) for
   `hidden_priority`. Everything, encoder included, is trained end to end
   with the loss below.

**How this relates to cross-attention in translation.** It's the same
attention operation; only the source of the queries changes:

| | Translation (encoder–decoder) | This model |
|---|---|---|
| Keys and values | encoder's token vectors (the source sentence) | encoder's token vectors (the email) |
| Queries | decoder's vectors for the target words so far, one per target position | one learned vector per question, fixed after training |
| How often | at every decoder layer, at every step of generation | once: a single attention layer |
| Output | next-word probabilities, one step at a time | one vector per question, then its answer |

It's a decoder cut down to a single step, with no self-attention and no
generation, whose "target side" is just one learned slot per question.
Nothing is generated, and all questions are answered in parallel from one
encoder pass. The same pattern appears elsewhere as "attention pooling" or
"learned queries" (for example, the Perceiver and DETR architectures).

**Why build it this way.** It copies the two properties TypeSafe advertises
for Jev's "System One" design (`docs/explainers.md`, "This project's head
design"): asking another question costs one small attention step over
vectors that already exist, not another encoder pass; and each question's
summary is computed separately, so questions can't contaminate each other
the way several questions listed in one chat prompt could.

**An important difference from Jev.** Here the four questions are fixed at
training time: each question's "meaning" is just a learned vector. Jev
accepts *new* questions at request time, written as text (the
`instructions` and `criteria` in "How we talk to Jev" below), so it must
turn question text into queries somehow, presumably by encoding it too.
TypeSafe doesn't publish how. The toy model therefore copies the *shape* of
Jev's design, not its workings, and it can't answer a question it wasn't
trained on. It couldn't answer S6's instruction with a stated base rate,
for example. That's one reason the toy results show what's *possible*
for this kind of design rather than how Jev actually works.

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
only one sampled draw from it? (The answer is in
`RESULTS_AND_DISCUSSION.md`, measured with the metrics defined in Part 3.)

### The same table, reused a second time, for evaluation

`toy/train.py`'s `evaluate()` and `probe/suites.py` both do the same thing
with the table: take a batch of items, get the generator's exact posterior
for each, get the model's (or Jev's) predicted distribution for each, and
compare them using the metrics defined in Part 3 (code in
`eval/metrics.py`, theory in `docs/explainers.md`). The training-time use and the
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
  except the loss function. (The `urgency` head is a plain softmax over the
  four levels. An earlier version used an ordinal head that builds in the
  low < medium < high < critical ordering; it trained too slowly to fit in
  3 epochs, and `RESULTS_AND_DISCUSSION.md` reports that diagnosis.)
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

### How we talk to Jev

Jev isn't a chatbot. You don't converse with it, and it never writes free
text. Each call sends it one piece of text, called the **state**, plus one
or more named **questions** about that text. It returns a structured answer
to each question, with probabilities. That's the entire interface.

**A real call, verbatim** (from `outputs/jev_cache.jsonl`, an S1 item).
The request:

```json
{
  "state": "From: sofia.costa@gmail.com\nSubject: Quick note\n\nDear Support Team,\nCould you add a dark mode option? Whenever you get a chance.\n\nBest regards,\nSofia Costa\nOperations Manager, Riverstone Logistics",
  "questions": {
    "category": {
      "type": "choice",
      "instructions": "Classify this support email into exactly one category.",
      "criteria": {
        "billing":   "About invoices, payments, refunds, or subscription costs.",
        "technical": "About bugs, crashes, error codes, or app malfunctions.",
        "sales":     "About pricing, plan features, or discounts.",
        "spam":      "Unsolicited marketing, or an unsubscribe request."
      }
    }
  }
}
```

The response:

```json
{
  "model": "typesafe/jev-1.13-20260917",
  "answers": {
    "category": {
      "type": "choice",
      "choice": "technical",
      "probabilities": {"technical": 0.66, "sales": 0.34, "billing": 0, "spam": 0},
      "confidence": 0.54
    }
  },
  "usage": {"input_tokens": 424, "output_tokens": 45, "cost": 1.7808e-05}
}
```

**Anatomy of a question.** Every question has three parts:

- **`type`** — one of Jev's three answer shapes:
  - `choice` (pick one of several options; returns `choice`, a probability
    per option, and `confidence`),
  - `noul` (yes/no; returns a single probability of "yes" as `noul`),
  - `score` (a position on an ordered scale; returns an expected `score`, a
    `legend` mapping positions 0, 1, 2, … to the scale's names, a probability
    per position, and `confidence`).
- **`instructions`** — the question in plain English.
- **`criteria`** — the answer options. For `choice` and `noul`, each option
  carries a one-line description that we wrote. For `score`, it's just the
  ordered list of level names.

**The criteria are Jev's only definition of each label.** Jev never sees
`slots.py`, realizations, priors or this document, and it was never trained
on this synthetic world. It's tested cold, the way a customer would use it.
So when Jev decides whether "Could you add a dark mode option?" is *sales*
or *technical*, all it has to go on is the four descriptions above plus
its own general sense of how support email works. Our tables say feature
requests are always sales, and "plan features" in the sales description is
meant to cover them. Jev's answer above leans technical. Where our
definitions and Jev's own sense of the world differ, it can disagree with
the answer key for reasons that have nothing to do with calibration.
`RESULTS_AND_DISCUSSION.md` flags each result where this matters.

**Every question used, word for word** (defined at the top of
`jev/probe/suites.py`):

| Name | Type | Instructions | Criteria | Used in |
|---|---|---|---|---|
| category | `choice` | "Classify this support email into exactly one category." | billing: "About invoices, payments, refunds, or subscription costs." · technical: "About bugs, crashes, error codes, or app malfunctions." · sales: "About pricing, plan features, or discounts." · spam: "Unsolicited marketing, or an unsubscribe request." | S1, S2, S5, S6 |
| priority (choice) | `choice` | "What priority level should this ticket be assigned?" | low: "Low priority" · medium: "Medium priority" · high: "High priority" · critical: "Critical priority" | S3, S4 |
| priority (yes/no) | `noul` | "Should this ticket be expedited, i.e. treated as high or critical priority?" | true: "Should be treated as high or critical priority." · false: "Standard priority is fine." | S4 |
| priority (score) | `score` | "Rate the priority level this ticket should be assigned." | low, medium, high, critical | S4 |

Two suites alter or combine these:

- **S4** sends all three priority questions in a *single* request, about
  the same email, so all three answers come from one reading of the text.
- **S6** sends the category question twice per email, with a sentence added
  to the front of the instructions: *"Context: historically, only about 10%
  of tickets in this queue turn out to be spam."* or *"Context:
  historically, about 60% of tickets in this queue turn out to be spam."*

**Plumbing.** TypeSafe's own signup was paused when this was run, so calls
went through OpenRouter, a gateway that forwards requests to many model
providers (`POST https://openrouter.ai/api/v1/systemone`, with the model
named in the payload). Every response reported model
`typesafe/jev-1.13-20260917` and provider `TypeSafe`. `probe/client.py`
appends every live request and response to `outputs/jev_cache.jsonl`, so
any result can be re-derived offline without paying again. Each email was
sent once per question set: there are no repeat calls on the same email,
so call-to-call variation in Jev's answers isn't measured. Across both
probe runs, 2,803 calls cost \$0.05 in total.

### The live probe: six suites against the real Jev

`jev/probe/suites.py` builds emails with the generator, sends them to Jev
using the questions above, and grades the answers against the exact
posterior.

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

> **Deprecated: S3 and the `hidden_priority` concept.** Priority is drawn
> independently of the text, but it shares its labels with urgency, and
> the question gives Jev no priority policy, so treating urgent wording as
> high priority is a reasonable reading, not a calibration failure. S3's
> results therefore test our hidden rule more than Jev. The code, data and
> results are kept as a record. S4 still stands: it uses the priority
> questions only to check whether Jev's three answer formats agree.

Two notes on reading the table. In S1 and S3, "accuracy" means agreement
with the *sampled* hard label, so even a perfect reader is capped by the
Bayes error rate. That is why S3's yardstick is 0.40, not 1.0. And S2 is
the only suite that grades Jev's full probability vector against the full
exact posterior. The others grade a chosen label, one derived probability,
or a shift. Every metric named in this table is defined, with its equation,
in Part 3, along with what the pass/fail thresholds mean.

### Head-to-head: Jev vs. a model trained on your own labels

The suites test Jev on purpose-built email sets. A practitioner's question
is different: *I have some labeled historical emails. Should I use Jev, fix
Jev's probabilities using my labels, or train my own model, and how many
labels does each option need?* `scripts/run_head_to_head.py` answers that
by putting every option on the **same** emails.

- **Emails.** All 6,440 emails in the toy experiment's held-out test split:
  a natural mix of difficulty, made of clue combinations no toy model
  trained on. They are shuffled once and split into a **calibration pool**
  of 2,000 emails, the labeled data available for fixing Jev, and an
  **evaluation set** of 4,440, where every option below is scored. Nothing
  is ever scored on emails it learned from.
- **Jev.** One request per email asking two questions together: `category`
  and priority (the `choice` version), worded exactly as in "How we talk to
  Jev." That's 6,440 live calls, all cached, all from
  `typesafe/jev-1.13-20260917`.
- **Jev + your labels.** Three ways of adjusting Jev's probabilities (Part
  3, "Recalibrating Jev"), each fitted on 50, 200, 1,000 or 2,000 labeled
  emails from the calibration pool. Each size is repeated with 3 different
  random subsets, except 2,000, which is the whole pool.
- **Your own model.** The toy model, trained on *hard labels only*, because
  a business has observed outcomes, not true probabilities. It's trained on
  50, 200, 1,000, 5,000 or all 23,560 labeled training emails, 3 seeds each
  (each seed also draws a different subset). Of each set, 20% is held back
  and the model keeps the weights from the epoch with the lowest loss on
  those held-back hard labels. That's standard practice, and it uses no
  information a practitioner wouldn't have. Smaller sets get more epochs
  (up to 40), so each run takes roughly the same number of training steps.
  The "labels used" count includes the held-back 20%.
- **Reference lines.** A soft-trained toy model on all training emails (the
  best a model can do when shown the true probabilities, which no business
  has), and the **exact posterior** itself (the ceiling: the best possible
  accuracy, zero KL, and the best possible ranking).
- **Metrics** (Part 3): accuracy, posterior KL, ECE and AURC on the
  4,440 evaluation emails, as mean ± std across the repeats or seeds. Jev's
  raw KL uses the rounding-aware floor, with the pre-registered floor
  reported alongside. On the priority question AURC isn't meaningful: the
  true answer is the same for every email, so there's no correct ranking.

---

## Part 3 — How results are measured

Every number in `RESULTS_AND_DISCUSSION.md` is one of the quantities
defined below. For each one: what it means in words, the exact equation
(matching the code in `jev/eval/metrics.py` or `jev/probe/suites.py`), a
small worked example, and where it's used. `docs/explainers.md` has the
deeper theory behind several of these (why a scoring rule is "proper,"
where the Brier decomposition comes from); this part only defines them.

### Notation

For item $i$ (out of $N$ items in a set):

- $p_{\text{true},i}$ — the exact posterior from Part 1: the correct
  probability for every label, given the realizations this item's text was
  built from. $p_{\text{true},i}(y)$ is its value for one label $y$.
- $p_{\text{model},i}$ — the distribution the model (the toy model, or
  Jev) reports for the same item.
- $\hat y_i$ — the label the model picks (its highest-probability label, or
  Jev's `choice`).
- $\hat p_i$ — the model's **confidence** in that pick: the highest
  probability it reports (for a yes/no answer, $\max(q, 1-q)$), for the
  toy model and Jev alike. Jev also returns a separate `confidence` field,
  but it isn't a probability: TypeSafe's docs describe it as a statistic
  computed from the probabilities, and it's the top probability rescaled
  so that a uniform guess reads 0 (`RESULTS_AND_DISCUSSION.md`, S5). So it can't be compared with an accuracy. The pre-registered S1 and
  S3 computations in `probe/suites.py` did use that field; the results
  report those numbers and the corrected ones side by side.
- $y_i$ — the item's hard label, sampled from $p_{\text{true},i}$ (Part 1).
- $o_i = \mathbb{1}[\hat y_i = y_i]$ — whether the pick matched the hard
  label: 1 if so, 0 if not.

### Accuracy, and its ceiling

The fraction of items where the model's pick matches the hard label:

$$
\text{accuracy} = \frac{1}{N} \sum_{i=1}^{N} o_i
$$

Because the hard label is itself a random draw, even a perfect reader
can't score 100% on an item that's genuinely ambiguous. If the truth is
70% billing / 30% technical, the best possible move is always to pick
billing — and that's wrong 30% of the time. So every accuracy here has a
**ceiling** of $1 - \text{Bayes error rate}$ (the table in Part 1): 0.779
for `category`, 0.560 for `urgency`, 0.700 for `is_business`, and 0.400 for
`hidden_priority`, where the best anyone can do is always guess `low`.

Accuracy only looks at the pick. It can't tell a model that says "51%
billing" from one that says "99% billing," as long as both pick billing.
That blind spot is the reason for every other metric below.

**How precise is an accuracy on 200 items?** Each suite reports a 95%
Wilson confidence interval: with $\hat a$ the measured accuracy and
$z = 1.96$,

$$
\frac{\hat a + \frac{z^2}{2N} \;\pm\; z\sqrt{\frac{\hat a(1-\hat a)}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}
$$

Worked example: 53 correct out of 200 ($\hat a = 0.265$) gives the interval
0.209–0.330. Read it as "the true long-run accuracy is very likely in this
range; a different 200 items could plausibly have landed anywhere in it."
(`metrics.wilson_ci`.)

### Posterior KL: how far the reported probabilities are from the truth

This is the project's main metric, and the one only a known answer key
makes possible. It compares the model's *whole* reported distribution
against the exact one, item by item:

$$
\text{KL}\bigl(p_{\text{true},i} \,\big\|\, p_{\text{model},i}\bigr) = \sum_{y} p_{\text{true},i}(y)\, \ln \frac{p_{\text{true},i}(y)}{p_{\text{model},i}(y)}
$$

then averaged over items (the results report both the mean and the
median). In words: for each label, how much less probability the model
gave it than it deserved, measured on a log scale and weighted by how
likely that label really is. Some properties worth having in your head:

- It's 0 exactly when the two distributions match, and positive
  otherwise. Lower is better. It's never negative.
- It punishes *confidently* wrong numbers much more than mildly wrong
  ones: the log term blows up as the model's probability for a label that
  really can happen approaches 0.
- It's not symmetric: it's measured from the truth's point of view, so
  only labels that can really happen contribute.

**Units: nats.** KL is measured in *nats* when it uses the natural log
($\ln$), as here, and in *bits* when it uses $\log_2$. It's the same
quantity in different units: 1 nat ≈ 1.44 bits. The number itself
has a concrete meaning — the average extra "surprise" you'd experience
if you believed the model's numbers and then watched the truth play out —
but it's easier to get a feel for by example. For a yes/no question whose
true probability is 0.714 (the `strong_lean` item from Part 1, collapsed
to "billing or not"):

| Model reports | KL (nats) | Roughly |
|---|---|---|
| 0.70 | 0.0005 | essentially right |
| 0.60 | 0.028 | a bit timid |
| 0.50 | 0.095 | "no idea" |
| 0.90 | 0.135 | overconfident |
| 0.95 | 0.295 | quite overconfident |
| 0.99 | 0.726 | near-certain, wrongly |
| 0.997 | 1.065 | effectively certain |
| 0.011 | 2.625 | confidently the opposite |

For comparison, a true probability of 0.345 reported as 0.011 costs 0.919
nats. The four-label KL used in the results is the same formula summed
over four labels instead of two.

**What a reported 0 costs.** If the model says exactly 0 for a label that
really has some probability, the log term is infinite. The code avoids
this by treating any reported probability below $10^{-12}$ as $10^{-12}$
(`metrics.kl_divergence`). That's a very harsh floor: each such label costs
$p_{\text{true}}(y)\,\bigl(27.6 + \ln p_{\text{true}}(y)\bigr)$ nats — roughly $25 \times p_{\text{true}}(y)$ — so a label with true
probability 0.067 that the model calls 0 adds 1.7 nats on its own. This
matters for Jev, which reports probabilities rounded to two decimals, so
its "0" really means "below 0.005." Where this changes a number, the
results report both the pre-registered computation (floor $10^{-12}$) and
a **rounding-aware** version: each of Jev's probabilities is raised to at
least 0.005, the vector is rescaled to sum to 1, and then KL is computed.
The toy model's softmax outputs are never exactly 0, so the issue doesn't
arise there.

(`metrics.mean_posterior_divergence`; theory in `docs/explainers.md`,
"KL divergence and posterior divergence.")

### Confidence-based metrics: ECE and the overconfidence gap

These only use the model's confidence $\hat p_i$ and whether it was right,
$o_i$. They're the standard calibration metrics when no exact answer key
exists, so they're included for comparability with other work.

**Expected calibration error (ECE).** Group items into $M$ bins by
confidence; in each bin $m$ (with $n_m$ items), compare the average
confidence $\bar p_m$ with the fraction actually correct $\bar o_m$:

$$
\text{ECE} = \sum_{m=1}^{M} \frac{n_m}{N} \bigl|\, \bar p_m - \bar o_m \,\bigr|
$$

Worked example: 100 items all at confidence 0.9, of which 80 are correct,
gives ECE = $|0.9 - 0.8| = 0.10$ — "10 points overconfident." The code uses
$M = 15$ bins of equal width on $[0, 1]$ by default. ECE's value depends on
the binning choice (the reliability-diagram PNGs use 10 bins holding equal
numbers of items and report different values for the same model), and it
only checks the top pick's confidence against sampled labels. Posterior KL
has neither weakness, so where the two disagree, trust KL.
(`metrics.expected_calibration_error`; `docs/explainers.md`, "Expected
Calibration Error (ECE) and its finite-sample bias.")

**Overconfidence gap** (used in S3). On a question where the best possible
accuracy is known, compare the model's average confidence with it:

$$
\text{overconfidence gap} = \frac{1}{N}\sum_{i=1}^{N} \hat p_i \;-\; \bigl(1 - \text{Bayes error rate}\bigr)
$$

An honest model on `hidden_priority` would report confidence ≈ 0.40 (the
prior's largest value), since that is how often its best guess can be
right. Gap 0 is honest; positive is overconfident.

### Ranking: the risk–coverage curve and AURC

A different question from calibration: are the model's *more* confident
answers actually the *more* likely to be right? Sort items from most to
least confident. For each $k = 1, \dots, N$, keep only the top $k$ and
measure the error rate among them:

$$
\text{coverage}_k = \frac{k}{N}, \qquad \text{risk}_k = \frac{1}{k} \sum_{j=1}^{k} \bigl(1 - o_{(j)}\bigr)
$$

where $o_{(j)}$ is the correctness of the $j$-th most confident item. **AURC**
is the area under the curve of risk against coverage (trapezoid rule). It's
low when errors are concentrated among the least confident items. Lower is
better.

AURC only depends on the *order* of the confidences, not their values:
squaring every confidence leaves AURC unchanged but wrecks calibration.
That's exactly why it's useful here. It separates "ranks items well" from
"reports the right numbers." (`metrics.risk_coverage_curve`,
`metrics.area_under_risk_coverage`; `docs/explainers.md`, "Selective
prediction: risk–coverage and AURC.")

### The two halves of "good probabilities": reliability and resolution

A standard way to state the distinction the last two sections draw is the
Brier score and its decomposition. The Brier score is the average squared
gap between confidence and outcome, and, grouped into the same bins as ECE,
it splits into three parts:

$$
\text{Brier} = \frac{1}{N}\sum_{i=1}^{N} (\hat p_i - o_i)^2 \;\approx\;
\underbrace{\frac{1}{N}\sum_m n_m(\bar p_m - \bar o_m)^2}_{\text{reliability}}
\;-\;
\underbrace{\frac{1}{N}\sum_m n_m(\bar o_m - \bar o)^2}_{\text{resolution}}
\;+\;
\underbrace{\bar o(1 - \bar o)}_{\text{uncertainty}}
$$

**Reliability** is the calibration error (lower is better). **Resolution**
is how well the confidences separate right answers from wrong ones (higher
is better); it's the quantity AURC tracks. **Uncertainty** is how hard the
set is, fixed regardless of the model. A model can be strong on one half and
weak on the other. The toy experiment's results use this vocabulary, with
posterior KL standing in for reliability and AURC for resolution.
(`metrics.brier_decomposition`; `docs/explainers.md`, "The Murphy / Brier
decomposition.")

### Quantities specific to one suite

**S2 per-level comparison.** Besides mean KL over the whole sweep, S2 is
reported per target level: the average of Jev's reported
$p_{\text{model}}(\text{billing})$ over the 50 items at that level, next
to the exact $p_{\text{true}}(\text{billing})$ for that level's combination.

**S4 — P(expedite) three ways.** "Expedite" means high or critical
priority. Its true probability is the same for every item, since
`hidden_priority` has no evidence in the text:
$0.2 + 0.1 = 0.30$ from the prior. Jev is asked three differently typed
questions about it, and each answer is converted to the same quantity:

$$
p^{\text{noul}}_i = \text{Jev's yes/no probability}, \qquad
p^{\text{choice}}_i = p^{\text{choice}}_i(\text{high}) + p^{\text{choice}}_i(\text{critical}), \qquad
p^{\text{score}}_i = p^{\text{score}}_i(\text{high}) + p^{\text{score}}_i(\text{critical})
$$

Reported: the mean of each over items, and the **pairwise disagreement**,
e.g. $\frac{1}{N}\sum_i \bigl|p^{\text{noul}}_i - p^{\text{choice}}_i\bigr|$,
which measures how far apart the framings are *on the same email*.

**S5 — does `confidence` add anything?** The Pearson correlation between
Jev's `confidence` and its highest probability, across the 200 items, and
the AURC computed once ranking by each. Correlation near 1 and equal AURCs
mean the field is a restatement of the probability, not a separate signal.

**S6 — reacting to a stated base rate.** The same email is sent twice, with
the instruction prefixed by "only about 10% of tickets … are spam" or
"about 60% … are spam." The correct response follows from Part 1's rule
with the stated rate as the prior $\pi$:

$$
P_{\pi}(\text{spam} \mid \text{email}) = \frac{\pi(\text{spam}) \prod_{s} P(r_s \mid \text{spam})}{\sum_{y} \pi(y) \prod_{s} P(r_s \mid y)}
$$

$$
\text{Bayes-correct shift} = P_{\pi_{60\%}}(\text{spam} \mid \text{email}) - P_{\pi_{10\%}}(\text{spam} \mid \text{email}), \qquad
\text{observed shift} = p^{60\%}_{\text{model}}(\text{spam}) - p^{10\%}_{\text{model}}(\text{spam})
$$

The instruction states only the spam rate, so the analysis has to split the
rest among billing, technical and sales. The code uses
$\{0.30, 0.30, 0.30\}$ and $\{0.15, 0.15, 0.10\}$. For the email S6 uses,
this choice doesn't matter: its clues (`generic_greeting`,
`no_domain_shown`, `curt`) are equally likely under all three non-spam
labels, so any split gives the same answer, $0.818 - 0.250 = +0.568$.
Reported: the mean observed and Bayes-correct shifts, and the fraction of
emails where Jev moved in the right direction.

### Recalibrating Jev: three ways to adjust its probabilities with your labels

Each method learns a correction from $n$ labeled emails (Jev's reported
distribution $p_i$ and the observed label $y_i$), then applies it to new
emails. Jev's probabilities are first floored at 0.005 and rescaled to sum
to 1, as in the rounding-aware KL, so that logarithms are finite.

**Temperature scaling** (one parameter). Soften or sharpen every answer by
the same amount:

$$
q(y) = \frac{\exp\bigl(\ln p(y) / T\bigr)}{\sum_{y'} \exp\bigl(\ln p(y') / T\bigr)},
\qquad
T = \arg\min_T \; -\frac{1}{n}\sum_{i=1}^{n} \ln q_i(y_i)
$$

$T > 1$ pulls answers toward uniform, fixing overconfidence. It never
changes which label is on top, or the order of emails by confidence, so it
can't change accuracy or AURC. (`metrics.fit_temperature_multiclass`.)

**Isotonic regression, per label.** For each label $k$, learn a
non-decreasing function $g_k$ mapping Jev's probability for $k$ to how often
$k$ was actually the answer:

$$
g_k = \arg\min_{g \text{ non-decreasing}} \sum_{i=1}^{n} \bigl(g(p_i(k)) - \mathbb{1}[y_i = k]\bigr)^2,
\qquad
q(k) = \frac{\max\bigl(g_k(p(k)),\, 10^{-3}\bigr)}{\sum_{k'} \max\bigl(g_{k'}(p(k')),\, 10^{-3}\bigr)}
$$

Each $g_k$ preserves order on its own, but because the four are learned
separately and then rescaled together, the top label *can* change. For
example, if "technical at 0.66" historically turned out to be technical
only 40% of the time, it gets pulled down. (scikit-learn
`IsotonicRegression`.)

**Logistic regression on Jev's log-probabilities** (a small model stacked
on Jev). Treat Jev's four log-probabilities as input features:

$$
q(y) = \frac{\exp\bigl(w_y^{\top} \ln p + b_y\bigr)}{\sum_{y'} \exp\bigl(w_{y'}^{\top} \ln p + b_{y'}\bigr)}
$$

with weights $w_y$ and offsets $b_y$ fit by L2-regularized log loss
(scikit-learn `LogisticRegression`, $C = 1$). This is the most flexible of
the three: it can learn cross-label patterns such as "when Jev splits
between technical and sales, the answer is usually sales," and it can
re-weight the base rates entirely. It's still limited to what Jev's four
numbers contain; it never sees the email.

### Toy-experiment summaries

Each toy-model number is computed once per training seed, then reported as
the mean ± sample standard deviation across the three seeds. A small
standard deviation means the result isn't a fluke of one training run.
Soft-vs-hard comparisons of KL are reported as a ratio, hard ÷ soft ("how
many times further from the truth").

### How a suite passes or fails: pre-registration

Before the first live call to Jev, each suite's **prediction** and a
**falsification criterion** ("we were wrong if…") were written into the
`HYPOTHESES` table in `jev/probe/suites.py`. That's pre-registration. Its
purpose is to rule out deciding after seeing the data what would have
counted as interesting. A result is reported against what was written
down, even when that turns out to be awkward.

Two designations, also fixed in advance:

- **Primary** (S2, S3): the claims the project stands on.
- **Exploratory** (S1, S4, S5, S6): worth reporting, but run alongside
  several others, so any single one could come out striking by chance. Read
  them as leads, not conclusions.

The thresholds are judgment calls. Nothing in the repo derives them; what
makes them useful is only that they were fixed beforehand. Translated into
plain terms:

| Suite | Prediction | Falsified if | In plain terms |
|---|---|---|---|
| S1 | accuracy and mean confidence both > 0.90 | either < 0.85 | Between 0.85 and 0.90 is "neither." |
| S2 | mean posterior KL < 0.15 nats | mean KL > 0.3, or every level's error in the same direction | 0.15 nats ≈ reporting 0.91 (or 0.44) when the truth is 0.714; 0.3 ≈ reporting 0.95 (or 0.33). The prediction's wording ("within 0.15 of 0.7") describes the 0.7 level, but the KL threshold is applied to the whole four-level sweep. |
| S3 | accuracy ≤ 0.40 *and* mean confidence clearly above 0.40 | mean confidence within 0.05 of 0.40 | Predicts Jev is overconfident on an unanswerable question. |
| S4 | the three framings disagree by > 0.1 | all three within 0.05 of each other and of 0.30 | |
| S5 | `confidence` has no meaningful AURC advantage, and is highly correlated with max probability | clearly lower AURC, or weak correlation | "Meaningfully," "clearly" and "weak" were not given numbers — a gap in the pre-registration. |
| S6 | observed shift < Bayes-correct shift | observed within 0.1 of Bayes-correct | Predicts Jev under-reacts to a stated base rate. |

A result between the prediction and the falsification line counts as
inconclusive, not as a pass.

For the full record, the S2 suite was run twice. After the first run, a
check of which P(billing) values the tables could produce at all showed a
gap between 0.37 and 1.0, so the sweep couldn't test the 0.5 and 0.7 levels
it was designed for. `mixed_lean` and `strong_lean` were added to fill it,
and the whole probe was re-run. The thresholds weren't changed. The first
run is archived in `outputs/probe_results_run1_20260925.json`.

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

### Epistemic via MC dropout: designed in the explainer, not implemented in code

This is worth being precise about, because it was described in
`docs/explainers.md` and the original project plan (not included in this repo) as part of the approach, and
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
interpreting Jev's own `confidence` field, which suite S5 found is a fixed
rescaling of `max(probability)` (for `choice` answers, exactly
(top − 1/K) / (1 − 1/K)), so it carries no information of its own about
when to trust an answer.

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

That completes the setup. The results, measured with exactly the
definitions in Part 3, are in
[`RESULTS_AND_DISCUSSION.md`](RESULTS_AND_DISCUSSION.md).

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
| Jev vs. recalibrated Jev vs. own model, on the same emails | `scripts/run_head_to_head.py` |
| Jev API client (live/replay/fixtures) | `jev/probe/client.py` |
| The six probe suites, S1–S6 | `jev/probe/suites.py` |
