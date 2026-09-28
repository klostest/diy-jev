In recent weeks, AI safety has increasingly entered the public conversation, as frontier labs disclose hacking incidents, and the existential threats posed by AI are being considered by many who may never have taken them seriously. Internet searches for "AI safety" have skyrocketed. The frontier labs have publicly supported a slowing down of AI development, while the observed pace of model releases accelerates.

Jev arrives at this moment and likely elicits a certain level of comfort. It is powerful like the frontier chatbots in that it can understand natural language from the user, but the responses are numbers in user-defined formats, as opposed to free text. You can try and tame an LLM system with evals, but Jev is tame by construction. It's simpler. It's controllable. You tell it how to behave and you can't imagine it taking over the world.

Instead, Jev supplies answers to yes/no question with a predicted probability (which it calls a "noul"), and also gives what is called a confidence score for that prediction. Multiclass ("choice") and ordinal ("score") classification are also offered; for each Jev returns both a probability and confidence score [***check this] per class.

Jev was trained entirely on synthetic data and its architecture is closed source. This made me wonder how Jev could possibly hope to supply a calibrated probability to a user's question, as Typesafe AI, Jev's developers, claim. I explore the calibration of Jev's probabilities, the confidence score, and Jev's consistency here. I also train from scratch a system with similar functionality to Jev, but only for a defined set of tasks where labeled training data exists, and compare the results to Jev's outputs.

# Part 0: The Data: Emails, constructed so that class probabilities are precisely known.

This project rests on a foundation of synthetic email messages, that a hypothetical support team at a business might receive. The emails were created in a way that certain labels are precisely known for each one: whether it fits in the billing, technical, sales, or spam category, for example. The email texts were generated with phrases, tone of voice, and sender domains, for example, that mapped to these labels. In some cases, label overlap was injected to make it impossible to perfectly recover all labels from the training set: there is a known Bayes optimal ceiling to the performance of any machine learning model in learning to predict this data, which we know because we controlled the data generating process.

[paste in the full example of an email from experimental setup]

[Distinguish between "recoverable" and "ambiguous", quantitatively with a statement from Claude]

This creates a dataset that we understand the statistical properties of. Jev has never seen this dataset before, so it's not expected that Jev would recover these properties. But they give us some material to start some open exploration of Jev with. Later in part 2 we'll use the labels to train a model, and compare it to Jev's outputs, calibrated to those same labels.

# Part 1 some probes of Jev

## S1: Obvious answers

This probe asks some questions to Jev that are recoverable from the labeled training data. While Jev has not seen these labels, recoverable emails were designed with obvious wording, so a capable natural language system may easily be able to classify them, with some descriptive labels.

We accessed Jev version XXX from OpenRouter for this exploration. When making a request to Jev for a multiclass classification, you supply the context as a string ("state"), let Jev know you're submitting a "choice" request, and supply some information about how to choose between each choice. Each choice also gets a free text description. So there are many opportunities for the user to inject context. Here is a simple prompt:

[ paste in from markdown How we talk to Jev]

The response includes a probability for each class, and an overall confidence score.

[paste in response]

For 200 recoverable emails, Jev correctly classified all but 18. These 18 were entirely in the sales class and involved feature requests. My theory as to what happened here is that Jev has built-in context associating feature requests with building software, as opposed to inbound support messages a team might receive alongside other customer-facing questions like billing questions. This counter-acted the presence of the word "feature" in the description of the sales category, shown in Jev's inputs above. There were actually 26 emails in the sales category mentioning feature requests and Jev correctly classified 8 of them, indicating the question choice description was partially successful. There are likely many ways to rephrase the inputs to Jev to "fix" this.

Jev's reported confidence in the 26 feature request emails is notable - only these emails received confidence scores less than 1, where were spread across 0.35-0.8 with a median of 0.6. So on the obvious questions, when Jev was struggling, it represented that through lower confidence.

[insert s1_confidence_by_type.png, with annotations removed, and the bottom graph moved to the top]

## S2: Calibrated to the unknown?

When I first heard that Jev was trained on synthetic data and produced calibrated probabilities to new questions, it sounded like an oxymoron. Later I learned on Jev's website that they define calibration as having the confidence score that goes along with with the predicted probability, as opposed to the standard definition of calibration in statistics, which measures how accurate predicted probabilities are in absolute terms, not just whether they correctly rank outcomes of various likelihoods. So when Jev claims "calibrated probabilities," it is in the sense of S1, where we saw that less certain decisions get reported with lower confidence. Nonetheless, I was curious to see "how well calibrated" Jev's probabilities are, given this synthetic dataset where we know them, and have attempted to create the associated texts in ways that make them recoverable semantically to various degrees.

[show picture, need to ask Claude to remove text annotations]

Jev correctly ranks the probabilities into four levels, although the separation between the top two levels is minute. Jev appears to tend toward extremes, and only have middling probabilities when it is uncertain, in this multiple choice question.

## S5: The confidence number

[Simple realization, I think discovered just by exploration, admit Claude found it. Not a long section. Maybe let Claude write this section in that spirit. Claude, what does this imply about the graph from S1? Was the certainty mechanically tied to the largest probability there too? I would think so, as you reported this is a consistent phenomenon]

## S4: Internal consistency

Jev allows three different question formats; do they agree when asked what is essentially the same question?

On each of 200 emails, Jev got three questions about priority in a single request, so all three answers come from one reading of the same email:
- yes/no: "Should this ticket be expedited, i.e. treated as high or critical priority?"
- choice: "What priority level should this ticket be assigned?", with options low, medium, high and critical.
- score: "Rate the priority level this ticket should be assigned", on the scale low → critical.

Each converts to the same number, P(high or critical). For yes/no it's the "yes" probability; for choice and score it's the probability on high plus critical.

┌─────────────────────────────────────────────┬────────┬────────┬───────┐
│               Urgency phrase                │ yes/no │ choice │ score │
├─────────────────────────────────────────────┼────────┼────────┼───────┤
│ none (70 emails)                            │ 0.24   │ 0.26   │ 0.29  │
├─────────────────────────────────────────────┼────────┼────────┼───────┤
│ "no particular rush" (60)                   │ 0.14   │ 0.11   │ 0.12  │
├─────────────────────────────────────────────┼────────┼────────┼───────┤
│ "this is fairly urgent" (52)                │ 0.46   │ 0.66   │ 0.63  │
├─────────────────────────────────────────────┼────────┼────────┼───────┤
│ "URGENT -- please treat as a priority" (18) │ 0.45   │ 0.76   │ 0.70  │
└─────────────────────────────────────────────┴────────┴────────┴───────┘

So Jev has some internal inconsistency. [elaborate briefly]

This is actually a more complicated question to consider asking Jev than it may seem at first, and this may have made Jev relatively less certain in this case. Here Jev is asked about prioritizing incoming messages. But no rules have been given to Jev for prioritization that the real-world team might use, like for example "billing requests go to the top of the stack." In this case Jev appears to interpret relative expressions of urgency or requests for prioritization in mostly reasonable ways, failing to prioritize messages on average when asked in yes/no form [Claude please sense-check that statement]. Adjusting the context provided to Jev, e.g. providing the prioritization rules the real-life team would use and grading Jev against those on test cases should underpin actual usage.

We've now probed Jev against some very well-understood data, synthetic but generated to be semantically believable, that Jev had never seen, and found that Jev generally behaves intuitively. While we actually know the true recoverable class probabilities for the classification problems we are posing to Jev because we designed and controlled the data generating process, in the real world a practicing data scientist would more likely only have access to one label per sample: the observed outcome, whatever the underlying class probabilities are.

# Part 2

Of course Jev's probabilities are not calibrated to unseen data, in the absolute accuracy sense of calibration. But what if you calibrate them? This assumes you have some data with known labels, which also begs the question, why not try to create your own version of a system like Jev from scratch? Here I compare these approaches.


## Concise model architecture

Sentences are first encoded by a small BERT-like model [Claude please fill in specifics]. These embeddings are used to train a classification model. [Claude please add a minimal explanation that includes the following equation which we should render in markdown]

$$
\alpha_t = \text{softmax}t!\left(\frac{(q_c W_Q),(h_t W_K)^{\top}}{\sqrt{d}}\right),
\qquad
z_c = \Big(\sum{t} \alpha_t , h_t W_V\Big) W_O
$$

[Claude please provide model training details, hard labels only, training time if you have it. You can say it was on a Mac M1 Max. The idea should be that this toy model did not require many resources to train.]

## Results from this

[Claude, please report conclusions from comparison to locally trained model and calibrated Jev, in particular that calibrated Jev is close to Bayes optimal, and with training on enough examples we can close the gap. Preamble this with a blurb on the calibration procedure and anything notable about that]


## Conclusion

Jev has never trained on your data, although it may have indirectly trained on things you've seen, like Wikipedia or whatever the encoder it presumably uses was trained on. Jev aims to be a generalist in terms of inputs, but a specialist in terms of outputs. Jev can be impressive in the face of questions it has never seen labels for, but can also struggle to output consistent answers for different ways of asking the same question. While the confidence number supplied from Jev appears to admit when Jev is uncertain, it also appears to be tautologically linked with the largest probability, at least in the case of the choice-type question examined here.

If one can afford to design and train a Jev-inspired system for one's specific data, which may be very cheap to do, one may, perhaps unsurprisingly, outperform Jev. However, there may still be a place for Jev and Jev-like systems, especially in situations where limited training data is available, or where out-of-the-box solutions are favored. Many have expressed enthusiasm for it and it is likely being tested in many production systems right now. The response to Jev may indicate pent-up demand for more certainty coming out of systems that process natural language, pointing the way to further developments in this direction.
