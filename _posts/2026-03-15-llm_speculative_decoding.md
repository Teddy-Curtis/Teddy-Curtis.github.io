---
layout: post
title: Speculative Decoding
date: 2026-03-16 10:45:00
description: Accelerating LLM Inference using Speculative Decoding
tags: LLM Inference PyTorch HuggingFace 
categories: LLM
disqus_comments: true
toc:
  beginning: true
tabs: true
---

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/posts/2026-03-15-llm_speculative_decoding/speculative_vs_baseline_illustrative.gif" class="img-fluid rounded z-depth-1 w-100 d-block mx-auto" zoomable=true caption="Figure 1: Example of using speculative decoding to increase inference speed." %}
    </div>
</div>

Large language models generate text one token at a time. Even when a prompt is short, each new token requires another forward pass through the model. For small models this is manageable, but for larger models inference quickly becomes expensive. In practice, this means that generation latency is often dominated not by the prompt, but by the long tail of repeated autoregressive decoding steps.

In this post, I implement speculative decoding from scratch in PyTorch and Hugging Face, using a small **draft model** and a larger **target model**. The core idea is simple: let the cheap model propose a short block of future tokens, then ask the expensive model to verify that block in one go. If the target agrees with most of the proposals, we can reduce the number of expensive target-model decoding steps and improve throughput.

The full code for this post lives in [this repository](https://github.com/Teddy-Curtis/LLM_Speculative_Decode), with the main scripts in `scripts/`. This can also be run with the following [Google Colab notebook](https://colab.research.google.com/drive/1qiNuh9T63f10LSPZqQ3qsOnt8bX2Whp6?usp=sharing). I will first explain the decoding problem and the speculative decoding algorithm at the theory level, and only after that move on to the implementation and the benchmark results.

## Why normal decoding is slow

With standard autoregressive decoding, we start from a prompt $$x_{1:n}$$ and generate the next token $$x_{n+1}$$ from

$$
p(x_{n+1} \mid x_{1:n}).
$$

After sampling or greedily choosing $$x_{n+1}$$, we append it to the sequence and repeat:

$$
p(x_{n+2} \mid x_{1:n+1}), \quad
p(x_{n+3} \mid x_{1:n+2}), \quad \ldots
$$

Even with a KV cache, this still means one target-model decode step per output token. If we want to generate $$T$$ new tokens, we still need $$T$$ separate target-model decoding steps. For a large model, that sequential dependence is the bottleneck.

So the question speculative decoding tries to answer is:

> Can a cheap model propose several future tokens, and can the expensive model verify those proposals faster than generating all of them itself?

## Speculative decoding

Speculative decoding introduces a second model:

- a small **draft model** $$p$$
- a larger **target model** $$q$$

The draft model is cheap, so we let it propose several candidate tokens:

$$
\hat{x}_{n+1}, \hat{x}_{n+2}, \ldots, \hat{x}_{n+\gamma}
$$

where $$\gamma$$ is the draft block size.

The target model then verifies that whole block in one forward pass on top of the cached accepted prefix. If many of those draft tokens are accepted, then one expensive target-model verification pass can replace several expensive target-model decode steps.

At a high level, the loop is:

1. Prime both draft and target caches on the same prompt.
2. Ask the draft model to propose $$\gamma$$ tokens.
3. Run the target model on the proposed block using its cached prefix.
4. Accept or reject each proposed token.
5. If all proposed tokens are accepted, sample one extra bonus token from the target.

This is the core reason speculative decoding can be faster: the target verifies a block rather than generating each token sequentially by itself.

### A concrete example: prefix `[A, B]`, draft proposal `[C, D, E]`

It is worth spelling this out carefully.

Suppose the currently accepted prefix is:

$$
[A, B].
$$

The draft model then proposes a block of three candidate tokens:

$$
[C, D, E].
$$

At this point, the target model already has a KV cache for the prefix `[A, B]`. That means it also already has the next-token distribution after `[A, B]`, which I will write as

$$
q(x \mid A, B),
$$

where $x$ is the next token.

This distribution is used to verify the **first** proposed draft token $$C$$.

Now the target model processes the proposed block `[C, D, E]` in one forward pass on top of the cached prefix. The important point is that a causal transformer returns logits at **every** position in the block, not just for the final next token. Because the model is causal, each position only attends to earlier tokens, so those logits are still valid autoregressive next-token distributions.

So the target forward pass gives us:

- the cached-prefix logits for

$$
q(x \mid A, B),
$$

- the logits at the position of $$C$$, which represent

$$
q(x \mid A, B, C),
$$

- the logits at the position of $$D$$, which represent

$$
q(x \mid A, B, C, D),
$$

- the logits at the position of $$E$$, which represent

$$
q(x \mid A, B, C, D, E).
$$

This means the target gets all of the following from one cached block pass:

- to verify $x=C$, use

$$
q(C \mid A, B)
$$

- to verify $$D$$, use

$$
q(D \mid A, B, C)
$$

- to verify $$E$$, use

$$
q(E \mid A, B, C, D)
$$

- and if all three are accepted, the final logits after $$E$$ can be used to sample one extra bonus token after

$$
[A, B, C, D, E].
$$

So the target is **not** just computing the probability of the next token after the full sequence and ignoring the intermediate steps. Instead, the intermediate logits produced while processing the block are exactly the quantities needed to verify each proposed token one by one.

### Why this does not leak future information

At first glance, it can seem like if the target processes `[C, D, E]` all at once, then the logits for $$C$$ might somehow already depend on $$D$$ and $$E$$. But that is not how a decoder-only causal transformer works.

The causal attention mask forces each position to attend only to earlier positions. So:

- the distribution used to verify $$C$$ cannot see $$D$$ or $$E$$
- the distribution used to verify $$D$$ can see $$C$$, but not $$E$$
- the distribution used to verify $$E$$ can see $$C$$ and $$D$$

In table form:

| Quantity | What it conditions on |
| --- | --- |
| verify `C` | `[A, B]` |
| verify `D` | `[A, B, C]` |
| verify `E` | `[A, B, C, D]` |
| bonus next token | `[A, B, C, D, E]` |

That is why one target-model block pass can safely replace several sequential target decoding steps.

### Does verification actually save target-model compute?

This is an important subtlety. When the target verifies a proposed block like `[C, D, E]`, it still has to do most of the real transformer computation for those positions. In particular, it still computes the hidden states and attention for the tokens in that block. So speculative decoding does **not** mean that the target somehow skips all tensor multiplications for `C`, `D`, and `E`.

The difference is in how that work is packaged.

With normal autoregressive decoding, the target has to:

1. run one step to get `C`
2. stop
3. choose `C`
4. run another step to get `D`
5. stop
6. choose `D`
7. run another step to get `E`

In speculative decoding, the draft has already proposed `[C, D, E]`, so the target can process that block in one cached verification pass instead of repeatedly returning control after every token.

So the right intuition is:

- the target still does most of the same large matrix maths for those tokens
- speculative decoding does not remove all target compute
- the speedup comes from doing verification in larger chunks and cutting down sequential overhead
- if enough draft tokens are accepted, one expensive target verification pass can replace several expensive target decode steps
- better-shaped block computation is often more efficient on GPU than many tiny sequential decode calls
- if the acceptance rate is poor, then the extra draft work and correction logic can erase the benefit
- this is why speculative decoding only helps in the right regime: cheap draft model, good agreement, and low implementation overhead

## Acceptance and correction rule

Suppose the draft proposes token $$x$$ at some step. Let

$$
p(x) = p_{\text{draft}}(x \mid \text{prefix}),
$$

and

$$
q(x) = q_{\text{target}}(x \mid \text{prefix}).
$$

The speculative acceptance probability is

$$
\alpha(x) = \min\left(1, \frac{q(x)}{p(x)}\right).
$$

So if the target likes the proposed token at least as much as the draft does, it is accepted with probability 1. If the target assigns lower probability, it is accepted only proportionally.

This formula is doing something very specific. The token $$x$$ has already been sampled from the draft distribution $$p$$. So the question is not:

- "would the target independently sample the same token?" 

<!-- Because when normally using an LLM model you might get $ q_{\text{target}}(x \mid \text{prefix}) $, from which you take the tokens with the top-30 probabilities, then you randomly sample from them using the probability distribution. In this case even if the draft and target had identical distributions, you might end up with a different token because of the randomness. This is why we compare the probability distributions and not just the selected token.   -->

Instead, the question is:

- "given that the draft sampled $$x$$, does the target assign enough probability mass to $$x$$ for us to keep it?"

This matters especially when decoding is sampled rather than greedy. In a normal language-model decode, we might start from

$$
q_{\text{target}}(x \mid \text{prefix}),
$$

restrict to the top-$$k$$ candidates, and then sample randomly from that truncated distribution. Once randomness enters the process, two models can have very similar or even identical probability distributions and still end up selecting different tokens on a particular run. So speculative decoding cannot simply ask:

- "did the draft and target choose the same token?"

That would reject too many perfectly reasonable draft proposals just because of sampling noise.

Instead, speculative decoding compares the probability distributions themselves, evaluated at the specific token that the draft actually sampled. That is why the acceptance rule is based on $$q(x)$$ and $$p(x)$$, not on whether the target would have independently sampled the exact same token on that run.

That is why the acceptance rule depends on the ratio

$$
\frac{q(x)}{p(x)}.
$$

There are two main cases.

### Case 1: the target likes the token at least as much as the draft

If

$$
q(x) \ge p(x),
$$

then

$$
\frac{q(x)}{p(x)} \ge 1,
$$

so

$$
\alpha(x) = 1.
$$

In that case, the target sees no reason to reject the draft proposal. The draft did not overstate how plausible that token was.

### Case 2: the draft likes the token more than the target does

If

$$
q(x) < p(x),
$$

then

$$
\alpha(x) = \frac{q(x)}{p(x)} < 1.
$$

Now the draft has effectively "oversampled" that token relative to the target. The acceptance rule compensates for that by only accepting the token with probability $$q(x)/p(x)$$.

This is the key idea: the draft can propose tokens freely, but if it proposes a token too often compared with the target distribution, the target will reject it proportionally often enough to correct the mismatch.

So the right way to think about

$$
\alpha(x) = \min\left(1, \frac{q(x)}{p(x)}\right)
$$

is:

- the token $$x$$ has already been chosen by the draft
- the target is not resampling from scratch
- the target is checking whether that specific token has enough probability under $$q$$
- the ratio $$q(x)/p(x)$$ measures whether the draft proposed that token too aggressively or not

This also explains why sampled decoding usually reduces speculative speedups in practice. Once temperature and top-$$k$$ sampling are enabled, the draft is more likely to sample tokens that are plausible under $$p$$ but not especially favored by $$q$$. The acceptance rule still keeps the overall algorithm aligned with the target distribution, but the acceptance rate often falls, which reduces the speed benefit.

If a token is rejected, we do not simply resample from $$q$$. Instead, we sample from the positive part of

$$
q - p,
$$

which in code is implemented as:

```python
remainder = torch.clamp(target_probs - draft_probs, min=0.0)
remainder = remainder / remainder.sum(dim=-1, keepdim=True)
```

This correction step is what keeps speculative decoding aligned with the target model distribution.

For the blog demo and the cleanest benchmark comparisons, I also added a greedy mode. In greedy mode, the draft still proposes tokens, but the final committed sequence is forced to follow the target model's greedy path exactly. This is useful for side-by-side visualizations because both baseline and speculative decoding can display the same text while revealing it at different speeds.

## Why block verification works

At the theory level, the whole algorithm rests on one fact about decoder-only transformers:

> if we pass a block of tokens through a causal transformer, the model returns logits at every position in that block, and each position only depends on the earlier tokens because of the causal mask.

That means a single target-model forward pass over a draft block does not just give the probability of the token after the entire block. It also gives the intermediate next-token distributions needed to verify the proposed tokens one by one.

This is the precise reason speculative decoding can save work. The target model is still doing real computation over the draft block, but it is doing that work in one cached block pass rather than in multiple separate sequential decode steps controlled by Python.

So the conceptual comparison is:

- baseline decoding: one expensive target step per committed token
- speculative decoding: one cheap draft block proposal plus one expensive target verification pass for several candidate tokens

If the draft and target agree often enough, the target ends up doing fewer expensive sequential decode steps.

## Implementation details

Once the theory is clear, the code becomes much easier to read. The implementation in this repository is split across:

- `scripts/common.py`
- `scripts/baseline_decode.py`
- `scripts/speculative_decode.py`
- `scripts/benchmark.py`
- `scripts/benchmark_sweep.py`

### Baseline autoregressive decoding

The baseline implementation follows the usual cached autoregressive loop:

```python
logits, past_key_values = prime_model_cache(model, generated)

for _ in range(max_new_tokens):
    _, next_token = select_next_token(logits, temperature, top_k, greedy)
    generated = torch.cat([generated, next_token], dim=1)
    logits, past_key_values = advance_model_cache(model, next_token, past_key_values)
```

So the baseline is conceptually simple: keep a KV cache for the current accepted prefix, choose the next token, append it, and advance the cache by one token.

This baseline matters because it is the reference point for everything that follows. If speculative decoding is going to be useful, it has to beat this much simpler loop in practice.

### KV cache handling

Both the draft and target models keep their own KV caches. These caches are **not** shared between models. They are only aligned in the sense that both represent the same accepted token prefix.

This is an important point. The draft model cache comes from running the draft model on the accepted prefix, and the target cache comes from running the target model on that same accepted prefix. Since the two models have different parameters and internal activations, their cache tensors are different objects.

The helper functions in `scripts/common.py` are:

- `prime_model_cache(...)`: run the full prompt once and initialize the cache
- `advance_model_cache(...)`: append one new token to an existing cache
- `trim_past_key_values(...)`: crop a cache back to a shorter accepted prefix after a rejection
    - I.e. if the draft model proposes [C, D, E] but the target model only agrees with C but picks a different D and E, then we need to remove D and E from the draft cache. 

### Speculative decoding in code

The core speculative decoding implementation then directly mirrors the theory above. The most important code path is the target verification step:

In code, the relevant part is:

```python
verification_logits = torch.cat(
    [target_next_logits.unsqueeze(1), target_outputs.logits[:, :-1, :]],
    dim=1,
)
```

Here:

- `target_next_logits` verifies the first draft token
- `target_outputs.logits[:, :-1, :]` verifies the later draft tokens
- `target_outputs.logits[:, -1, :]` is the bonus-next-token distribution after the full block

For the `[A, B]` and `[C, D, E]` example above, this corresponds to:

$$
\left[
q(\cdot \mid A, B),\;
q(\cdot \mid A, B, C),\;
q(\cdot \mid A, B, C, D)
\right]
$$

for verification, and then

$$
q(\cdot \mid A, B, C, D, E)
$$

for the bonus token after the full accepted block.

### EOS-aware stopping

For the final demos, I also added EOS-aware stopping to both the baseline and speculative decoders. Without this, the scripts would always run to `max_new_tokens`, even when the model had already produced a sensible complete answer.

## Benchmark results

The most important lesson from this project was that speculative decoding does **not** automatically produce a speedup. In many early experiments it was slower than the baseline. This depended heavily on the model pair, the block size, and whether decoding was sampled or greedy.

For example:

- `distilgpt2 -> gpt2` remained slower than baseline
- `distilgpt2 -> gpt2-medium` also remained slower
- even `Qwen/Qwen2.5-0.5B -> Qwen/Qwen2.5-7B` was slower in the current manual implementation

The strongest win came from a long greedy decode with:

- draft model: `distilgpt2`
- target model: `gpt2-xl`
- `max_new_tokens = 300`
- draft block size $$= 6$$

That benchmark gave:

- baseline throughput: `26.2539 tokens/sec`
- speculative throughput: `43.1552 tokens/sec`
- average acceptance rate: `0.7146`
- speedup vs baseline: `1.6438x`

This was not the first configuration I tried, which is exactly why I think the final result is interesting. The implementation only became faster than the baseline in a favourable regime:

- large target model
- longer generations
- greedy decoding
- tuned draft block size

Sweeping the draft block size in the same regime gave the following pattern:

- block size 2: `1.0431x`
- block size 3: `1.2464x`
- block size 4: `1.3966x`
- block size 5: `1.5350x`
- block size 6: `1.6874x`
- block size 7: `1.5720x`
- block size 8: `1.5560x`

So there is a clear sweet spot. Too small a block underuses speculative verification, while too large a block causes enough disagreement that the acceptance rate drops and the advantage fades.

## How this can be improved

This repository is intentionally a from-scratch PyTorch and Hugging Face implementation. That makes it useful for understanding the algorithm, but it also introduces a lot of overhead:

- Python-side control flow
- repeated small tensor operations
- model calls made through general-purpose transformer modules rather than specialized inference kernels

That overhead is one of the main reasons several of the model pairs I tried did **not** beat the baseline, even when the draft model was much smaller than the target. In other words, the algorithm can be sound while the implementation is still too expensive.

There are at least three obvious directions for improvement.

### 1. Use a faster inference stack

The first improvement is systems-oriented rather than algorithmic. Production inference stacks often combine speculative decoding with:

- continuous batching
- quantization
- more specialized KV-cache handling

This matters because speculative decoding only helps if the savings from fewer large-model decode steps outweigh the extra bookkeeping and verification work.

### 2. Train a better speculator

In this project I used the "small separate draft model" approach. That is the simplest way to understand speculative decoding, but it is not the only approach.

The PyTorch blog post *A Hitchhiker’s Guide to Speculative Decoding* describes another method: instead of using a separate small model, they attach multiple speculative heads directly to the base model and train those heads to predict several future tokens. In their setup, the model predicts the usual next token plus several lookahead tokens in the same forward pass, and they report substantially better speedups in a production inference environment than I achieved here. They also note that a naive implementation would replicate KV-cache across heads, so they modify the attention kernel and masking to make this efficient in practice. See the PyTorch post for details:

- https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/

This is a much more production-oriented direction than the simple two-model setup in this repository.

### 3. Train or choose a more aligned draft-target pair

Even when using two separate models, speculative decoding works best when the draft agrees with the target often enough. In practice, that means:

- same-family model pairs
- draft and target trained in similar ways
- a block size that matches the actual agreement rate

The benchmarks in this project made that clear. Some pairs produced poor acceptance rates and no speedup, while others only became competitive in longer greedy-decoding runs.

### 4. Improve the benchmark methodology

There is also room to improve the experimental setup itself. In a future version I would like to:

- benchmark more prompt sets
- separate quality-oriented demos from speed-oriented benchmarks
- compare greedy and sampled decoding more systematically
- report metrics like time-to-first-token and inter-token latency in addition to tokens/sec

That would make the final results more robust and easier to compare with production-style systems work.

## Final note

My main takeaway from this project is that speculative decoding is conceptually simple, but practically quite sensitive to the setup. A from-scratch implementation does not automatically win against a well-cached baseline. The details matter:

- model pair
- cache handling
- decode mode
- block size
- implementation overhead

That said, once tuned into the right regime, the method really can work. In the best configuration above, the speculative decoder delivered a substantial speedup while still following the target model's output.

This makes speculative decoding a good example of the gap between algorithmic ideas and systems reality. On paper, it sounds like an obvious improvement. In practice, it only becomes convincing once the implementation, the model pair, and the benchmark conditions are all aligned.
