# Explainers

A companion reference for the concepts this project leans on. Written for
someone comfortable with classical ML (cross-entropy, ROC/AUC, regularization,
the bias-variance decomposition, Bayes' rule) and picking up transformers —
so those basics are assumed, and the calibration-specific machinery and
transformer internals get the space instead.

Each entry is intuition, then the equation, then why it matters in *this*
repo, with a pointer to the code that computes it.

> **On `$` in this file:** the global convention in this project is to escape
> dollar signs so they don't trigger math mode. This file is the deliberate
> exception — `$...$` below are real LaTeX math delimiters. A literal
> currency amount (e.g. \$0.06) still gets escaped.

---

## Part 1 — Scoring rules and calibration

### Proper scoring rules

A scoring rule grades a probabilistic forecast against the outcome that
actually happened. It's *proper* if the forecaster's own best strategy,
in expectation, is to report their true belief — no other number scores
better on average. Formally, a rule $S(p, y)$ is proper if for every true
distribution $q$,
$$
q = \arg\max_p \; \mathbb{E}_{y\sim q}[S(p, y)]
$$
This is the foundation the whole project rests on: it's *why* training a
model to minimize cross-entropy or Brier score against a known-correct soft
target is the right thing to do, rather than a heuristic. Log score and
Brier score (below) are both proper; accuracy is not a scoring rule at all
(it grades a decision, not a distribution). See [[dgp/generator.py]]'s exact
posteriors — they're targets a proper scoring rule can actually be graded
against, not just estimated.

### Log score (NLL) and Brier score

Log score: $S(p, y) = \log p(y)$ (maximize) — equivalently, negative
log-likelihood, $-\log p(y)$ (minimize). This is ordinary cross-entropy loss.
Brier score, for outcome $o\in\{0,1\}$ and forecast $p$:
$$
\text{Brier} = (p - o)^2
$$
generalizing to $K$ classes as $\sum_k (p_k - \mathbb{1}[y=k])^2$. Both are
proper. Log score punishes confident wrong answers far more harshly (it's
unbounded as $p\to 0$); Brier score is bounded in $[0,2]$ and decomposes
cleanly into interpretable parts (next section), which is why calibration
work usually reports Brier alongside NLL. Code: `eval/metrics.py`'s
`brier_decomposition`; the soft-target training loss in `toy/train.py`'s
`soft_xent`/`soft_bce` is the log-score side of this pair.

### Calibration, stated formally

A forecaster is calibrated if, among all the times it says probability $p$,
the outcome happens a $p$ fraction of the time:
$$
\mathbb{E}[\,Y \mid \hat p = p\,] = p \quad \text{for (almost) every } p
$$
This is a statement about *long-run frequency given the forecast*, not about
any single prediction — a calibrated model can still be wrong on any
individual item, in fact it must be, exactly $1-p$ of the time when it says
$p$. This is the property TypeSafe's marketing claims for Jev and the
property none of their own published evals actually measure.

### Expected Calibration Error (ECE) and its finite-sample bias

ECE approximates the calibration gap by bucketing predictions into $M$ bins
and averaging the gap between confidence and accuracy, weighted by bin size:
$$
\text{ECE} = \sum_{m=1}^{M} \frac{n_m}{N}\, \bigl|\, \bar p_m - \bar o_m \,\bigr|
$$
where $\bar p_m$ is the mean confidence and $\bar o_m$ the mean outcome
(accuracy) in bin $m$. The catch: even a perfectly calibrated model has
finite-sample noise in $\bar o_m$ (it's an average of Bernoulli draws), and
because ECE takes an absolute value, that noise is a one-directional
(upward) bias — ECE is never negative, so sampling noise can only inflate
it. A raw ECE of 0.03 is meaningless without knowing what a perfectly
calibrated model would score at that same sample size and confidence
histogram. `eval/metrics.py`'s `ece_noise_floor` computes that number
directly: simulate `outcome ~ Bernoulli(confidence)` at the model's own
confidences, many times, and report the resulting ECE's distribution — the
floor any real ECE should be judged against, not zero. This is exactly the
comparison the independent OOD-calibration test made (ECE 0.107 vs. a
noise floor of 0.024 — 4.4× the floor, not "close to zero").

### The Murphy / Brier decomposition

Brier score splits exactly into three interpretable terms (Murphy, 1973):
$$
\text{Brier} = \underbrace{\text{reliability}}_{\text{calibration gap}} \; - \; \underbrace{\text{resolution}}_{\text{discrimination}} \; + \; \underbrace{\text{uncertainty}}_{\text{irreducible floor}}
$$
with, for bins $m$ of size $n_m$, overall base rate $\bar o$:
$$
\text{reliability} = \frac{1}{N}\sum_m n_m(\bar p_m - \bar o_m)^2, \quad
\text{resolution} = \frac{1}{N}\sum_m n_m(\bar o_m - \bar o)^2, \quad
\text{uncertainty} = \bar o (1-\bar o)
$$
This is the single most important piece of machinery in the whole project,
and the one thing worth sitting with: **reliability** is "does 0.9 mean
90%?" (calibration in the strict sense above); **resolution** is "does the
forecast separate outcomes at all?" (a forecaster who always says the base
rate is perfectly calibrated and has zero resolution — useless but never
wrong on average). A forecaster needs *both* to be useful. Rereading the Jev
evidence through this lens: the business-email benchmark result — Jev's
most-confident 85.5% of predictions had zero errors, at lower accuracy
overall than Gemini — is almost pure **resolution** (confidence sorts
correct from incorrect very well). The unanswerable-question result — 0.74
mean confidence at 44.7% accuracy — is a **reliability** failure (confidence
doesn't track truth at all there). Same underlying system, two different
properties, opposite verdicts. One subtlety: this identity is exact only
for the *binned* Brier score (using each bin's representative value, not the
raw per-item probability) — `brier_decomposition` returns both `brier_raw`
(what you'd actually quote) and `brier_binned` (what the identity
reconstructs exactly), with the gap between them (`grouping_loss`) shrinking
to zero as bins get finer. Verified by `test_brier_binned_identity_holds_exactly`.

### Reliability diagrams and temperature scaling

A reliability diagram plots $\bar o_m$ against $\bar p_m$ per bin — the
scatter of bars around the diagonal *is* reliability, visually. Temperature
scaling is the standard one-parameter fix for a systematic miscalibration:
rescale the logits before the softmax,
$$
\hat p = \mathrm{softmax}(z / T)
$$
$T>1$ flattens the distribution toward uniform (fixes overconfidence);
$T<1$ sharpens it (fixes underconfidence); $T=1$ is a no-op. Fitting $T$ by
minimizing NLL on held-out data is a cheap, robust post-hoc calibration
check. It only needs logits, and Jev returns probabilities, not logits — but
because softmax is shift-invariant, $\log p$ *is* a valid logit vector whose
softmax reproduces $p$ exactly (`logits_from_probs` in `eval/metrics.py`),
so temperature scaling applies to any probability-returning model, Jev
included, with no information lost in the recovery. The detector test in
`test_metrics.py` (verification step 3 in the project plan) constructs a
predictor with a *known* sharpening factor and confirms the fit recovers it
almost exactly, with the correct sign.

### Discrimination vs. calibration, and why AUROC misses half the story

AUROC (and ranking metrics generally) only cares about the *order* of scores,
not their absolute value: it's invariant to any monotone rescaling. A model
whose scores are $\{0.99, 0.01\}$ for correct/incorrect has the same AUROC as
one whose scores are $\{0.51, 0.49\}$ for the same items — identical
resolution, wildly different reliability. This is precisely why "Jev's
confidence ranks outcomes well" (resolution, AUROC/AURC-flavored) and "Jev's
probabilities are calibrated" (reliability, ECE/Brier-reliability) are
different claims that need different metrics, and why a vendor citing only
accuracy-vs-cost curves (as TypeSafe's own evals do) says nothing about
calibration at all.

### Selective prediction: risk–coverage and AURC

If you can abstain below a confidence threshold, the question becomes: at
what error rate can you auto-handle what fraction of cases? Sort items by
confidence descending; at coverage $c$ (the top $c$ fraction), *selective
risk* is the error rate among just those items:
$$
\text{risk}(c) = \frac{1}{\lfloor cN \rfloor}\sum_{i=1}^{\lfloor cN \rfloor} \mathbb{1}[\hat y_{(i)} \ne y_{(i)}]
$$
with items ordered by decreasing confidence. AURC integrates this curve —
lower is better, and it rewards good *ranking* (resolution) regardless of
whether the absolute confidence values are calibrated. This is the metric
that actually answers "can I auto-route the confident slice and escalate the
rest?", which is the real production question behind both Jev's confidence-
gated pitch and the email-routing benchmark's headline finding. Code:
`risk_coverage_curve` / `area_under_risk_coverage`.

---

## Part 2 — Probability and the generator

### Bayes' rule in log-odds form, and its link to logistic regression

For two labels with likelihoods $P(x\mid y)$:
$$
\underbrace{\log\frac{P(y{=}A\mid x)}{P(y{=}B\mid x)}}_{\text{posterior log-odds}}
= \underbrace{\log\frac{P(y{=}A)}{P(y{=}B)}}_{\text{prior log-odds}} +
\underbrace{\log\frac{P(x\mid y{=}A)}{P(x\mid y{=}B)}}_{\text{log-likelihood ratio (the "evidence")}}
$$
This is exactly logistic regression's functional form — a linear model in
log-odds space *is* a Bayes-optimal classifier whenever the log-likelihood
ratio happens to be a linear function of the features, which the
naive-Bayes structure below guarantees by construction. `dgp/generator.py`'s
emission tables are just $P(\text{realization}\mid \text{label})$; the
posterior arithmetic in `posterior()` is this identity, generalized to
$K$ labels and multiple slots.

### Naive Bayes conditional independence, and why the posterior stays closed-form

"Naive Bayes" means: given the label, the evidence channels (slots) are
drawn independently. That's an assumption baked into most real classifiers
by choice; here it's a fact about the generator, true by construction, not
approximated:
$$
P(r_1,\dots,r_k \mid y) = \prod_{i=1}^k P(r_i \mid y)
$$
so the posterior is exactly
$$
P(y \mid r_1,\dots,r_k) \; \propto \; \text{prior}(y) \prod_i P(r_i\mid y)
$$
with no estimation step anywhere — this single equation is the reason the
whole project can train against a *true* target instead of a label scraped
from the world. See `[[jev-project]]` and the plan's "where the training
targets come from" section for the worked numeric example.

### Bayes error rate

The irreducible error rate of the *optimal* classifier for a given problem —
even a perfect model can't beat it, because the label genuinely isn't
determined by the input beyond this point:
$$
\text{Bayes error} = \mathbb{E}_x\Bigl[\,1 - \max_y P(y\mid x)\,\Bigr]
$$
Ordinarily this is unknowable (it requires the true posterior). Here it's
exactly computable by enumerating the small, finite realization space
(`bayes_error_rate` in `generator.py`) — which turns it from a theoretical
curiosity into a hard invariant: *no model, ours or Jev's, can legitimately
score better than this number on our generated data.* If one does, something
is leaking (see the negative controls in `dgp/scenarios.py`).

### KL divergence and posterior divergence

Kullback-Leibler divergence measures how much information is lost using
distribution $q$ to approximate the true $p$:
$$
D_{\mathrm{KL}}(p \,\|\, q) = \sum_y p(y) \log\frac{p(y)}{q(y)} \;\ge 0, \quad = 0 \iff p=q
$$
It's not symmetric, and that asymmetry is meaningful here: $D_{\mathrm{KL}}(p_{\text{true}}\,\|\,p_{\text{model}})$
penalizes the model heavily for assigning near-zero probability to
something the truth says is possible (confidently missing a real outcome)
more than the reverse. This is the metric no published Jev test can compute,
because it requires *owning* $p_{\text{true}}$, not just observing samples
from it — this project's whole reason for building a generator with an
exact posterior rather than using any existing labeled dataset. Code:
`kl_divergence` / `mean_posterior_divergence`.

### Aleatoric vs. epistemic uncertainty

Two reasons a model might be uncertain, and they call for different fixes.
**Aleatoric** ("of the dice") is irreducible — the world itself is
ambiguous given what you can observe; more data about *this exact input*
wouldn't help, because the ambiguity is in the data-generating process, not
in what the model has learned. **Epistemic** ("of knowledge") is
reducible — the model just hasn't seen enough like this yet; more training
data in that region would sharpen it. `dgp/generator.py`'s posteriors are
pure aleatoric uncertainty by construction (the `probabilities` a model
should match). Epistemic uncertainty is created deliberately in `toy/`
by holding out regions of latent space, and measured as *disagreement*
across seeds/dropout masks on those held-out regions — a different
computation from reading a probability, because a single forward pass
can't tell you which kind of uncertainty it's reporting. This is exactly
the ambiguity in Jev's `confidence` field that suite S5 is built to probe:
TypeSafe's docs say confidence is "distinct from probability" without
specifying which kind of uncertainty it estimates, or whether it estimates
either coherently.

### Soft targets vs. label smoothing (and knowledge distillation)

These look alike — both replace a one-hot target with something softer —
and are conceptually opposite. **Label smoothing** replaces the one-hot
target with an arbitrary fixed mixture, e.g. $0.9$ on the true class and
$0.1/(K{-}1)$ spread over the rest, *regardless of the input* — a
regularizer that discourages overconfidence on average, with no claim that
the smoothed value is correct for any particular item. The **soft targets**
this project trains on are the exact, item-specific true posterior — not a
regularization trick, the actual answer. This is also precisely
**knowledge distillation**'s target (Hinton et al., 2015): a student
trained to match a teacher's full output distribution rather than its
hard argmax. The difference is where the soft target comes from: distillation's
teacher is itself a fallible model, ours is exact by construction. `toy/train.py`'s
`soft_xent` is literally the distillation loss; the novelty here is having
a "teacher" that's provably correct rather than merely bigger.

### Simulation-based calibration and amortized inference

The founder's phrase "statistically well-understood synthetic data" points
at real, established methods this project's core trick is a small instance
of. **Simulation-based calibration** (Talts et al., 2018) checks a Bayesian
inference procedure by simulating parameters from a prior, simulating data
from those parameters, running inference, and checking that the resulting
posterior ranks are uniform — i.e., checking the *machine's* outputs against
a generative process you wrote yourself, exactly the move
`test_category_posterior_matches_monte_carlo` makes against our own
generator. **Amortized inference / neural posterior estimation** trains a
network once, on simulated (parameter, data) pairs with known parameters, to
predict the posterior directly for new data — "amortized" because the cost
of doing inference is paid once during training rather than every time you
query. That is exactly the toy model's soft-target training: it's an
amortized approximator to the exact posterior the generator can compute
directly but slowly (by enumeration), learning to produce it in one forward
pass instead.

### Ordinal regression: the cumulative-link (proportional-odds) model

`urgency` and `hidden_priority`'s score view have ordered levels (low <
medium < high < critical) — a plain softmax head ignores that order
entirely and would happily be equally "confident" mistaking low for
critical as low for medium. The cumulative-link / proportional-odds model
(McCullagh, 1980) instead posits a single latent utility $\eta$ per item and
a set of increasing cutpoints $c_1 < c_2 < \dots < c_{K-1}$:
$$
P(Y \le k \mid \eta) = \sigma(c_k - \eta)
$$
so level probabilities are differences of consecutive cumulative
probabilities, guaranteed non-negative and ordered by construction. `toy/model.py`'s
`OrdinalHead` parameterizes the cutpoints as a base value plus a cumulative
sum of `softplus`-transformed increments, which structurally enforces
$c_1<c_2<\dots$ rather than merely hoping training finds it that way.

### MC dropout and deep ensembles

Two cheap ways to turn "one model" into "a distribution over models," which
is what's needed to measure epistemic uncertainty (a single forward pass
gives aleatoric uncertainty at best). **MC dropout** (Gal & Ghahramani, 2016)
keeps dropout active at inference time and runs the same input through
multiple times; the spread across those stochastic forward passes
approximates posterior uncertainty over weights. **Deep ensembles**
(Lakshminarayanan et al., 2017) train several full models from different
random seeds/data orders and look at their disagreement — cruder but often
better-calibrated in practice. This project uses the ensemble/seed-spread
version: train the toy model across several seeds, and read disagreement in
regions of held-out latent space as the epistemic signal, kept structurally
separate from the aleatoric posterior the generator already hands us exactly.

### Bootstrap CIs and the noise-floor simulation

The bootstrap estimates a statistic's sampling distribution by resampling
the observed data *with replacement* many times and recomputing the
statistic on each resample — no formula needed for statistics (like ECE)
that don't have a clean closed-form variance. `bootstrap_ci` in
`eval/metrics.py` is exactly this. The ECE noise floor (above) is a
close cousin: instead of resampling *observed outcomes*, it *simulates*
fresh outcomes from a hypothetically perfect model at the same confidences,
many times — same logic (build up a distribution by repeated resampling/
simulation), different question (what would a perfect model's own noise
look like here, rather than how uncertain is my estimate of this model).

---

## Part 3 — Transformers, for the toy model

### Tokenization: why subwords

Splitting text into whole words breaks on anything unseen at training time
(typos, names, new terms); splitting into characters loses too much
structure and makes sequences long. Subword tokenization (WordPiece, BPE)
is the practical middle ground: a fixed vocabulary of common whole words
plus word-pieces, built so any string can be represented as *some*
sequence of vocabulary pieces, however unfamiliar. `all-MiniLM-L6-v2` uses
WordPiece (BERT's original tokenizer); the toy model never sees this
directly — `transformers.AutoTokenizer` handles it in `toy/data.py`.

### Embeddings and positional information

A token ID is just an arbitrary integer; an embedding layer is a learned
lookup table mapping each ID to a dense vector, so tokens with similar
meaning end up with similar vectors after training. Attention (below) has
no built-in notion of order — it would treat "dog bites man" and "man bites
dog" identically without help — so a positional signal (learned or
sinusoidal position embeddings, or, in newer designs, RoPE — see
ModernBERT below) is added or otherwise mixed in so the model can tell
*where* a token is, not just *what* it is.

### Self-attention, from the equation up

For a sequence of token vectors, attention lets every position gather
information from every other position, weighted by relevance. Each token
produces a query $Q$, key $K$, and value $V$ (three learned linear
projections of the same input); attention output is
$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$
$QK^\top$ scores how much each query matches each key (dot-product
similarity); dividing by $\sqrt{d_k}$ (the key dimension) keeps those dot
products from growing with dimension and pushing softmax into a
saturated, nearly-one-hot regime where gradients vanish — pure numerical
housekeeping, not a modeling choice. The softmax turns scores into a
weighted average over the values $V$. **Multi-head** attention runs several
of these in parallel with smaller dimensions each, letting different heads
specialize in different kinds of relationships (e.g. one head tracking
syntactic adjacency, another tracking topical similarity), then concatenates
and projects back. `toy/model.py`'s cross-attention pooling is a direct,
minimal use of this same equation: the *query* is a learned, per-concept
vector (not derived from the input at all), and the *keys/values* are the
encoder's token states — so each concept's query "asks the text" its own
question independently, in one extra small matrix multiply per question
rather than another full pass through the network.

### Residual stream, layer norm, and the MLP block

Each transformer layer doesn't replace its input, it *adds* a computed
update to it: $x \leftarrow x + f(x)$. That additive "residual stream" is
why very deep networks are trainable at all — gradients have a direct path
backward through the `+` regardless of how complicated $f$ gets. Layer
normalization rescales activations to stable mean/variance before (or
after, depending on architecture) each sub-block, which is what keeps
training stable at depth. Every attention block is paired with a small
per-token MLP (usually expand-then-contract, e.g. $d\to 4d\to d$) that lets
the model do per-position nonlinear feature mixing that attention itself
(a weighted average) cannot.

### Encoder vs. decoder; bidirectional vs. causal masking

The *only* structural difference between a BERT-style encoder and a
GPT-style decoder is the attention mask. A decoder uses **causal masking** —
position $i$ can only attend to positions $\le i$ — because it has to work
at generation time, one token at a time, with no future tokens to look at
yet. An encoder has no such constraint: every position attends to every
other position, forward and backward, because the whole input is available
at once and there's nothing being generated left to right. This is exactly
why a "System One" model — and this project's toy model — is built on an
**encoder**, not a decoder: the task is "read the whole input, answer typed
questions about it," never "produce the next token of an open-ended
continuation." `all-MiniLM-L6-v2` and ModernBERT are both encoders.

### Pretraining objective: masked language modeling vs. next-token prediction

A decoder LLM is pretrained to predict the next token given everything
before it — autoregressive, causal. An encoder like BERT/ModernBERT is
pretrained with **masked language modeling**: some tokens are hidden, and
the model predicts them using context from *both* directions
simultaneously. This isn't incidental to the calibration story: MLM
training never produces a properly normalized probability over "the
answer" the way next-token prediction produces a distribution over the
vocabulary — an encoder has no native output distribution at all until you
attach a task-specific head, which is exactly what `toy/model.py` does
(the choice/score/noul heads on top of pooled encoder states).

### The specific encoders: MiniLM and ModernBERT

`all-MiniLM-L6-v2` (this project's default) is a **distilled** model:
trained to mimic a much larger sentence-embedding model's outputs rather
than from scratch, which is how it gets to ~22M parameters and still
produces useful sentence-level representations — a smaller, cheaper student
matching a bigger teacher's behavior (same core idea as the soft-target
training above, one level up the stack, applied to the encoder itself
rather than to our task heads). **ModernBERT** (the heavier alternative
noted in the plan) modernizes the original BERT recipe: rotary position
embeddings (RoPE) instead of learned absolute positions (better length
generalization), GeGLU activations in the MLP block, "unpadding" (packing
variable-length sequences without wasting compute on padding tokens), and a
much longer supported context. Either works as the encoder underneath
`toy/model.py`; MiniLM trains faster and is the right default for
iterating on an M1 Max, ModernBERT-base is the "if you want more headroom"
option the plan names.

### This project's head design: learned queries, cross-attention, per-question isolation

Rather than pooling the encoder's output once (e.g. averaging all token
vectors) and running every task head off that single shared vector,
`toy/model.py` gives each concept (category/urgency/is_business/
hidden_priority) its **own learned query vector**, and pools the encoder's
token states independently for each one via the attention equation above
(query = that concept's vector, keys/values = the token states). This
reproduces, in miniature, the two things TypeSafe's Jev docs advertise as
System One properties: adding another question costs one more small
attention operation over states that are already computed, not another
pass through the encoder; and each question's pooled representation is
computed independently of every other question's, so one question's
answer can't be contaminated by another's the way a single shared context
(as in a chat-style prompt with several questions listed together) could
leak between them.

### Why decoder LLMs tend to be poorly calibrated for decisions

A next-token softmax is a distribution over *vocabulary pieces*, not over
"the answer" to a decision — turning "yes/no" or "which of 4 categories"
into a well-formed probability requires either reading off token
probabilities for specific strings (fragile: "Yes" vs "yes" vs "Yes." are
different tokens) or asking the model to *state* a confidence in words,
which is a generated claim, not a computed one, and RLHF/preference tuning
optimizes for outputs humans rate highly — a target with no necessary
relationship to being numerically well-calibrated, and evidence (both
published and independent) suggests it actively sharpens overconfidence.
This is the structural argument for why a purpose-built decision head over
an encoder — this project's toy model, and plausibly Jev itself, though
TypeSafe doesn't disclose the architecture — has a real shot at being
better calibrated than reading a probability out of a chat model, even
before any training-data argument.

### Fine-tuning mechanics: learning rates, layer-wise treatment, warmup

Fine-tuning a pretrained encoder with freshly-initialized task heads on top
creates a mismatch: the encoder's weights already encode useful structure
and should move gently; the heads start from random noise and need to move
fast to become useful at all. `toy/train.py`'s `build_optimizer` uses two
learning-rate groups for exactly this reason (a lower rate for `encoder.*`
parameters, a higher rate for everything else). **Warmup** — linearly
ramping the learning rate up from zero over the first slice of training
before decaying it — avoids destructively large updates while the Adam
optimizer's moment estimates are still unreliable in the first few steps;
`get_linear_schedule_with_warmup` implements the standard linear-warmup,
linear-decay schedule used here.

### MPS/Metal specifics on the M1 Max

`torch.backends.mps` is Apple's Metal-backed accelerator backend — the
right device for this project's training on an M1 Max (`pick_device()` in
`toy/train.py` selects it automatically when available). One caveat worth
keeping: MPS's fp16 kernels have historically been less numerically
reliable than CUDA's; this project trains in the default fp32 throughout
rather than opting into autocast/fp16, trading some speed for correctness
it doesn't have to think about again. `PYTORCH_ENABLE_MPS_FALLBACK=1` is
worth setting in the environment if a future change introduces an op MPS
doesn't yet implement — it silently falls back to CPU for just that op
instead of crashing the run.
