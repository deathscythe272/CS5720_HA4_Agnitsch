# CS5720 Neural Networks and Deep Learning — Home Assignment 4

**Student:** Jeff Agnitsch
**Student ID:** 7005146960
**Course:** CS5720, Fall 2026, University of Central Missouri

## Summary

Sequence-to-sequence models, attention, and Transformers. Three parts:

- **Part I** — five short-answer questions, answered below.
- **Part II** — two Python programs, an NLTK preprocessing pipeline and a NumPy
  scaled dot-product attention function, with their short-answer questions. Code is in
  [`src/`](src/) and, with executed outputs, in
  [CS5720_HA4_Agnitsch.ipynb](CS5720_HA4_Agnitsch.ipynb).
- **Part III** — the step-by-step self-attention calculation for the third token of the
  lecture example, worked by hand below and checked in
  [`src/part3_self_attention_check.py`](src/part3_self_attention_check.py).

Everything runs on CPU in a few seconds with Python 3.13, NumPy, and NLTK.

## How to run

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\q1_nlp_preprocessing.py
python src\q2_scaled_dot_product_attention.py
cd src; python part3_self_attention_check.py; cd ..
```

Or open the notebook and run all cells:

```powershell
jupyter notebook CS5720_HA4_Agnitsch.ipynb
```

The first run of Q1 downloads two small NLTK resources (`punkt_tab`, `stopwords`) into
your user NLTK data folder.

---

## Part I — Short answer

### Question 1. Sequence-to-sequence models and attention

**(a) Encoder and decoder.** In a Seq2Seq translation model the **encoder** is an RNN
that reads the source sentence one token at a time and compresses what it has read into
its hidden state. The **decoder** is a second RNN that starts from the encoder's final
state and generates the target sentence one token at a time, feeding each predicted word
back in as the next input until it emits `<EOS>`. The encoder turns variable-length text
into a representation; the decoder turns that representation back into variable-length
text in the other language.

**(b) Why the fixed-length context vector fails on long sentences.** In the original
Sutskever et al. model the encoder's single final hidden state is the only thing the
decoder ever sees. That one fixed-size vector has to summarize every word, in order, no
matter how long the input is. A 5-word and a 40-word sentence get the same capacity, so
as sentences grow the vector cannot hold everything, and information from early tokens is
overwritten by later ones. Cho et al. showed exactly this: BLEU scores drop steadily as
source and reference sentence length increase. The hypothesis is that the fixed-length
vector lacks the capacity to capture all the relevant information.

**(c) How attention fixes it.** Attention stops asking the encoder to summarize. Instead
the encoder passes **all** of its per-token hidden states to the decoder, and at every
decoding step the decoder computes a score between its current state and each encoder
state, softmaxes the scores into weights, and forms a context vector as the weighted sum
of the encoder states. Through attention the decoder can access every encoder hidden
state, weighted by how relevant each source position is to the word being produced right
now, rather than one compressed summary. The context vector therefore changes at every
output step, and performance no longer degrades with length.

### Question 2. Attention mechanisms

**(a) Hard vs soft attention.** **Hard attention** picks one discrete input position at
each decoding step and uses only that encoder state. **Soft attention** uses a weighted
combination of all encoder states, with weights from a softmax. Hard attention is a
discrete choice, so there is no gradient through the selection and it has to be trained
with sampling or reinforcement learning tricks. Soft attention is a differentiable
weighted average, so every part of the model, including the attention weights, can be
trained end to end with ordinary backpropagation. It is also richer: each target word can
draw on several source words at once.

**(b) Dot-product scores.** With `s = [1, 0]`:

| encoder state | score e_i = sᵀh_i |
|---|---|
| h₁ = [1, 0] | 1·1 + 0·0 = **1** |
| h₂ = [0, 1] | 1·0 + 0·1 = **0** |
| h₃ = [1, 1] | 1·1 + 0·1 = **1** |

Score vector e = [1, 0, 1].

**(c) Softmax weights and context vector.**

Z = e¹ + e⁰ + e¹ = 2.7183 + 1.0000 + 2.7183 = 6.4366

| state | score | exp(score) | weight α_i = exp / Z |
|---|---|---|---|
| h₁ | 1 | 2.7183 | 0.4223 |
| h₂ | 0 | 1.0000 | 0.1554 |
| h₃ | 1 | 2.7183 | 0.4223 |

Sum of weights = 1.0000. To three decimals: **α = [0.422, 0.155, 0.422]**.

Context vector c = Σ α_i h_i:

- 0.4223 · [1, 0] = [0.4223, 0.0000]
- 0.1554 · [0, 1] = [0.0000, 0.1554]
- 0.4223 · [1, 1] = [0.4223, 0.4223]

First dimension = 0.4223 + 0.4223 = 0.8446. Second dimension = 0.1554 + 0.4223 = 0.5777.

**c = [0.845, 0.578]**. h₁ and h₃ tie for the largest weight because both have a dot
product of 1 with the decoder state; h₂ is orthogonal to it and gets the least.

### Question 3. Self-attention in Transformers

**(a) What self-attention is and how it differs.** Self-attention produces a new
representation of every token in a sequence that reflects its relationship to all the
other tokens in the **same** sequence. The query, key, and value for each position all
come from the same input. In the encoder-decoder attention of a traditional Seq2Seq model
the two sides are different: the query comes from the decoder's hidden state and the keys
and values come from the encoder's hidden states, so it relates tokens from two different
sources, target to source. Self-attention relates a sequence to itself and needs no
recurrence, so all positions can be computed at once.

**(b) Query, Key, Value.** Each input embedding is multiplied by three learned matrices
W_Q, W_K, W_V to produce three vectors. The **query** is the current token asking "what
is relevant to me?"; it is compared against every key. The **key** is each token's
advertisement of what it contains; the query–key dot product gives the attention score,
which says how relevant that token is and by how much. The **value** is the actual
content that gets passed on: the output for the query token is the weighted sum of all
value vectors, weighted by the softmaxed scores. The lecture's analogy is a key–value
store: the query is the search, keys are matched against it, and values are the records
returned.

**(c) Long-range relationships.** Because every token attends directly to every other
token in one step, distance in the sentence does not matter. An RNN has to carry
information about an early word through every intermediate hidden state, and it fades;
self-attention computes a direct query–key score between any two positions, so a word
ten tokens away is as reachable as the neighbor. Example from the lecture: in "Rashonda
accepted a job in deep learning because **she** loves the topic," the query for "she"
gets a high score against the key for "Rashonda," so the new representation of "she"
encodes who it refers to even though eight words separate them. Likewise "bank" in "I
arrived at the bank across the river" attends to "river" and is disambiguated.

### Question 4. Multi-head attention

**(a) What it is and why.** Multi-head attention runs several self-attention mechanisms
in parallel, each with its own W_Q, W_K, W_V matrices. A single head produces one set of
attention weights per token, so it can capture only one kind of relationship at a time.
Multiple heads let each token relate to the others in several ways at once, in different
representation subspaces: in the lecture example one head has "it" attend to "the
animal" (what "it" is) while another head has "it" attend to "tired" (how "it" feels).
Heads can specialize in syntax, coreference, position, and so on.

**(b) Dimension per head.** With embedding dimension d_model = 512 and h = 8 heads
split evenly:

d_head = d_model / h = 512 / 8 = **64**

Each head works with 64-dimensional queries, keys, and values. Eight heads × 64 = 512,
so the total width is unchanged.

**(c) Combining the heads.** Each head i produces an output matrix Z_i of width 64. The
eight outputs are **concatenated** along the feature dimension back to width 512, and the
result is multiplied by a learned output projection matrix W_O (512 × 512). That single
linear layer mixes information across heads and produces the final multi-head attention
output, which has the same dimension as the input so it can feed the residual connection
and the next layer.

### Question 5. Transformer architecture

**(a) Positional encodings.** Self-attention is a set operation: the weighted sums do
not depend on where tokens are, so without extra information "I ate an apple" and "apple
an ate I" would produce identical representations. Positional encodings are vectors added
to the input embeddings that tell the model each token's position in the sequence. They
provide order information, both absolute position and, through the sinusoidal design,
relative distance between positions. The lecture's requirements: unique for each
position, not cyclic, and bounded.

**(b) Masked self-attention.** In the decoder, the self-attention scores for every
position after the current one are set to negative infinity before the softmax, so their
attention weights become exactly 0 and a token can only attend to itself and earlier
tokens. This is necessary because the decoder is trained with teacher forcing: the whole
target sentence is fed in at once so all positions can be computed in parallel. Without
the mask, the representation for position t could look at the token at t+1, which is the
very thing it is supposed to predict. The model would learn to copy instead of to
predict, and at inference time, when future tokens do not exist, it would fail. The
output at time T may only depend on outputs up to T−1.

**(c) RNNs vs Transformers.** An RNN processes a sequence strictly in order: hidden
state t depends on hidden state t−1, so both training and inference are sequential and
cannot be parallelized across time steps. A Transformer has no recurrence; self-attention
computes every position from the whole input at once, so the encoder, and the decoder
during teacher-forced training, process all tokens in parallel on a GPU. **Advantage:**
much faster training on long sequences and direct, constant-length paths between any two
tokens, so long-range dependencies are learned more easily. **Limitation:** at
autoregressive generation time the decoder must still produce one token at a time,
because each new token depends on the tokens already generated and future tokens are
unknown. Inference is therefore sequential, and each step re-attends over the growing
output, so generation cost grows with sequence length.

---

## Part II — Programming

### Q1. NLP preprocessing pipeline

[`src/q1_nlp_preprocessing.py`](src/q1_nlp_preprocessing.py). The function `preprocess()`
tokenizes with NLTK's `word_tokenize`, drops tokens whose lowercase form is in NLTK's
English stop word list, and stems with the Porter stemmer. Output:

```
Sentence: NLP techniques are used in virtual assistants like Alexa and Siri.

1. Original Tokens:           ['NLP', 'techniques', 'are', 'used', 'in', 'virtual', 'assistants', 'like', 'Alexa', 'and', 'Siri', '.']
2. Tokens Without Stopwords:  ['NLP', 'techniques', 'used', 'virtual', 'assistants', 'like', 'Alexa', 'Siri', '.']
3. Stemmed Words:             ['nlp', 'techniqu', 'use', 'virtual', 'assist', 'like', 'alexa', 'siri', '.']

Stop words removed: ['are', 'in', 'and']
```

Notes on the output. The tokenizer splits the final period into its own token, so the
first list is "all words and punctuation" as the assignment describes. Three stop words
are removed: "are", "in", "and". "like" survives because it is not in NLTK's English
list. The Porter stemmer lowercases and strips suffixes by rule, which is why
"techniques" becomes "techniqu" and "assistants" becomes "assist", neither of which is a
dictionary word. The period is kept through all three stages; stripping punctuation would
be a fourth, separate step.

**Stemming vs lemmatization.** Stemming chops suffixes off with fixed rules and no
dictionary, so it is fast but can produce non-words. Lemmatization looks the word up in a
vocabulary, uses its part of speech, and returns the dictionary base form, so it is
slower but always yields a real word and is more accurate. For "running": the Porter
stemmer gives `run`, and the WordNet lemmatizer with a verb tag also gives `run`. The
difference shows on other words: stemming turns "flies" into `fli` and "happily" into
`happili`, while lemmatizing gives `fly` and `happily`. And lemmatization is
context-aware: "running" as a noun ("a running of the bulls") lemmatizes to `running`,
while as a verb it lemmatizes to `run`; a stemmer gives `run` either way.

**When removing stop words helps, and when it hurts.** It helps in bag-of-words style
tasks such as topic classification, document clustering, keyword extraction, and search
indexing: words like "the", "is", "in" appear in every document, carry no topical
signal, and removing them shrinks the vocabulary, speeds up training, and lets the
informative words dominate the features. It hurts when function words carry meaning.
Sentiment analysis loses negation ("not good" becomes "good"). Machine translation,
question answering, and language modeling need articles, prepositions, and pronouns to
produce grammatical output and resolve references ("who", "when", "not" are stop words
in many lists). Phrase matching and named entities break too: "The Who", "to be or not to
be", and "Take That" collapse to almost nothing. Modern Transformer models are trained on
raw text precisely so attention can learn which words matter.

### Q2. Scaled dot-product attention

[`src/q2_scaled_dot_product_attention.py`](src/q2_scaled_dot_product_attention.py).
`scaled_dot_product_attention(Q, K, V)` computes `QKᵀ`, divides by `√d_k`, applies a
numerically stable row-wise softmax, and multiplies by `V`. Output on the assignment's
test matrices:

```
Q K^T (raw scores):
[[2. 0.]
 [0. 2.]]

Scaled by sqrt(d_k) = sqrt(4) = 2.0000:
[[1. 0.]
 [0. 1.]]

Attention weights (softmax over each row):
[[0.7311 0.2689]
 [0.2689 0.7311]]
Row sums: [1. 1.]

Output = weights @ V:
[[2.0758 3.0758 4.0758 5.0758]
 [3.9242 4.9242 5.9242 6.9242]]
```

Each query matches its own key (score 2) and is orthogonal to the other (score 0), so
after scaling by 2 and softmaxing, each query puts weight e/(e+1) ≈ 0.731 on its own
value row and 1/(e+1) ≈ 0.269 on the other. Row 1 of the output is 0.731·[1,2,3,4] +
0.269·[5,6,7,8]; row 2 is the mirror image.

**Why divide by √d.** The dot product of two d-dimensional vectors is a sum of d terms,
so its magnitude grows with d: if the components have unit variance, the score has
variance d. Large scores push softmax into its saturated region, where one weight is
nearly 1 and the rest nearly 0, and the gradient of softmax there is close to zero, so
training stalls. Dividing by √d brings the scores back to unit variance regardless of
the key dimension, keeping softmax in the range where it produces soft, trainable
weights. The lecture puts it as "scaling down the size helps preserve gradients needed
for training."

**How self-attention helps the model understand relationships between words.** Every
token's query is scored against every other token's key, so each token gets a
distribution over the whole sentence saying which other words are relevant to it, and
its new representation is the value vectors of those words blended by that
distribution. The representation of a word therefore depends on its context: "bank"
next to "river" and "bank" next to "loan" come out different. Because every pair is
compared directly, the model picks up dependencies regardless of distance, such as
pronoun-to-antecedent links, subject-verb agreement across clauses, and which adjective
modifies which noun, and multiple heads can capture several such relationships at once.

---

## Part III — Self-attention calculation, third token

Inputs from the Chapter 9 lecture example (three tokens, embedding size 4, projected to
size 3). Multiplying each input by W_Q, W_K, W_V gives:

| token | input x | query q | key k | value v |
|---|---|---|---|---|
| 1 | [1, 0, 1, 0] | [1, 0, 2] | [0, 1, 1] | [1, 2, 3] |
| 2 | [0, 2, 0, 2] | [2, 2, 2] | [4, 4, 0] | [2, 8, 0] |
| 3 | [1, 1, 1, 1] | [2, 1, 3] | [2, 3, 1] | [2, 6, 3] |

Method, as in class: unscaled dot-product scores, softmax, round the weights to one
decimal place, then weighted sum of the values.

**Step 1 — scores for token 3.** Dot product of q₃ = [2, 1, 3] with each key:

| key | calculation | score |
|---|---|---|
| k₁ = [0, 1, 1] | 2·0 + 1·1 + 3·1 | **4** |
| k₂ = [4, 4, 0] | 2·4 + 1·4 + 3·0 | **12** |
| k₃ = [2, 3, 1] | 2·2 + 1·3 + 3·1 | **10** |

Score vector = [4, 12, 10].

**Step 2 — softmax.**

Z = e⁴ + e¹² + e¹⁰ = 54.60 + 162754.79 + 22026.47 = 184835.86

| key | exp(score) | weight | rounded to 1 dp |
|---|---|---|---|
| k₁ | 54.60 | 0.0003 | **0.0** |
| k₂ | 162754.79 | 0.8805 | **0.9** |
| k₃ | 22026.47 | 0.1192 | **0.1** |

Rounded attention weights = [0.0, 0.9, 0.1], sum 1.0. Token 3 is most related to token
2 and least related to token 1.

**Step 3 — weighted values.**

- 0.0 · v₁ = 0.0 · [1, 2, 3] = [0.0, 0.0, 0.0]
- 0.9 · v₂ = 0.9 · [2, 8, 0] = [1.8, 7.2, 0.0]
- 0.1 · v₃ = 0.1 · [2, 6, 3] = [0.2, 0.6, 0.3]

**Step 4 — sum.**

First dimension: 0.0 + 1.8 + 0.2 = 2.0. Second: 0.0 + 7.2 + 0.6 = 7.8. Third:
0.0 + 0.0 + 0.3 = 0.3.

**New representation of token 3 = [2.0, 7.8, 0.3]**, matching the lecture slide.

`src/part3_self_attention_check.py` reproduces this in NumPy and also runs the same Q,
K, V through the Part II scaled function. With scaling by √3 and no rounding the
weights for token 3 become [0.007, 0.755, 0.238] and the output [1.99, 7.48, 0.74]: the
same ranking, but softer, which is the point of the scaling. The script also notes one
quirk: for token 1 the slide treats softmax([2, 4, 4]) as [0.0, 0.5, 0.5], but the
true first weight is 0.063, which rounds to 0.1; the slide flags this approximation
itself. Token 3, the one asked for here, has no such issue.

---

## Repository layout

```
CS5720_HA4_Agnitsch.ipynb            notebook with Part II code and Part III check, outputs included
src/q1_nlp_preprocessing.py          Part II Q1
src/q2_scaled_dot_product_attention.py   Part II Q2
src/part3_self_attention_check.py    Part III numeric check, imports from q2
requirements.txt
```

## AI use

I used Claude (Anthropic) as a helper on this assignment, not as a substitute for doing
it. It helped me review the lecture slides against the assignment, plan the work,
structure the code, wordsmith the comments and README, and check the arithmetic. I ran
everything myself on my own machine, and every number in this repo comes from those
runs or from the hand calculations shown above.
