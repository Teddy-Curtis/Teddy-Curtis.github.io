---
layout: post
title: The Reparameterization Trick and REINFORCE
date: 2026-06-02 10:45:00
description: What is the reparameterization trick / REINFORCE and why do we need them?
tags: PyTorch ML AI RL Diffusion VAE
categories: AI
disqus_comments: true
toc:
  beginning: true
---

# Introduction

The reparameterization trick is one of those ideas that appears in many different places in machine learning: variational autoencoders, diffusion models, stochastic policies in reinforcement learning, stochastic computation graphs, and more. Similarly, in reinforcement learning, we end up with policy-gradient updates such as REINFORCE.
 
At first glance, these applications can look unrelated. One looks like a trick with Gaussian noise, the other looks like a trick with log-probabilities. But both are really responses to the same underlying problem: a stochastic optimisation problem in which the learned parameters $\theta$ live inside a probability distribution that we sample from. For example

$$
J(\theta)
=
\mathbb{E}_{x \sim P_\theta}[f(x)].
$$

<!-- In Gaussian latent-variable models and diffusion models, for example, we end up rewriting a sample like $x \sim \mathcal{N}(\mu_\theta,\sigma_\theta^2)$ as $x = \mu_\theta + \sigma_\theta \epsilon$ with $\epsilon \sim \mathcal{N}(0,1)$. -->

So the real question is not just "how does the reparameterization trick work?", but rather:

> Why do we need tricks like REINFORCE and reparameterization in the first place?

The goal of this post is to answer that question from the ground up using a simple biased-die example. The key point is:

> If the parameters $\theta$ appear inside the probability distribution from which we sample, then a Monte Carlo estimate of the objective contains a stochastic sampling step. A raw sampled value is just a number. Changing $\theta$ changes the probability of seeing that number, but not the value of the number itself. Therefore ordinary backpropagation does not know how to pass gradients through the sampling step.

Don't worry if you don't fully understand what this means yet, we will go over this slowly in this post.

-----

# Fixed Underlying Data Distribution

In standard supervised learning, we often see objectives of the form

$$
L(\theta)
=
\mathbb{E}_{(x,y) \sim \mathcal{D}}
\left[
\left(y - \hat{y}_\theta(x)\right)^2
\right].
$$

Here, the data distribution $\mathcal{D}$ does not depend on $\theta$. The model parameters only appear inside the deterministic prediction function $\hat{y}_\theta(x)$.

When we train with minibatches, we approximate the expectation by samples from the dataset:

$$
\hat{L}(\theta)
=
\frac{1}{n}
\sum_{i=1}^n
\left(y_i - \hat{y}_\theta(x_i)\right)^2.
$$

This is completely compatible with backpropagation: the sampled data points $(x_i,y_i)$ are treated as fixed constants, and the dependence on $\theta$ is still explicit through the network output $\hat{y}_\theta(x_i)$. The computation graph looks like

$$
\theta \longrightarrow \hat{y}_\theta(x_i) \longrightarrow \left(y_i - \hat{y}_\theta(x_i)\right)^2.
$$

So automatic differentiation can compute

$$
\nabla_\theta \left(y_i - \hat{y}_\theta(x_i)\right)^2
$$

using the ordinary chain rule.

There is randomness in minibatch training, but the randomness is only in which training examples we choose. Once a minibatch has been chosen, the path from $\theta$ to the loss is deterministic and differentiable.

-----

# Parameter-Dependent Underlying Data Distribution

Now consider a different-looking objective:

$$
J(\theta)
=
\mathbb{E}_{x \sim P_\theta}[f(x)].
$$

This looks superficially similar, but is fundamentally different. The parameter $\theta$ no longer appears inside a deterministic function. Instead, it controls the distribution of the random variable $x$ itself, with $x \sim P_{\theta}$. Before clarifying why this leads to issues with automatic differentiation, we will first show some examples of ML problems that have an objective in this form. 

## Example 1: policy optimisation in reinforcement learning

In reinforcement learning, we often want to maximise expected return:

$$
J(\theta)
=
\mathbb{E}_{\tau \sim p_\theta(\tau)}[R(\tau)].
$$

Here $\tau = (s_0,a_0,s_1,a_1,\ldots)$ is a trajectory, and its distribution depends on the policy parameters $\theta$ because actions are sampled from the policy:

$$
a_t \sim \pi_\theta(\cdot \mid s_t).
$$

So this is exactly of the form $$ \mathbb{E}_{x\sim P_\theta}[f(x)] $$, with the random object now being an entire trajectory. OpenAI Spinning Up derives the policy-gradient update using the log-derivative trick, but that derivation by itself does not really explain why the stochastic sampling step caused a problem in the first place: [OpenAI Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html).

## Example 2: Gaussian sampling in variational autoencoders, latent-variable models, and diffusion models

The objective for training Variational Autoencoders (VAE) is the evidence lower bound (ELBO) which contains the term

$$
\mathbb{E}_{z \sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)],
$$

where the encoder network is $q_\phi(z\mid x)$ and the decoder network is $p_\theta(x\mid z)]$. So, some learned parameters, $\phi$, sit inside the distribution being sampled from. The same basic structure also appears in other latent-variable models and in diffusion-style Gaussian sampling, such as in Lilian Weng's great diffusion post [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). In this case, suppose a model outputs $\mu_\theta$ and $\sigma_\theta$, and our objective contains a term like

$$
J(\theta)
=
\mathbb{E}_{x \sim \mathcal{N}(\mu_\theta,\sigma_\theta^2)}[f(x)].
$$

Again, $\theta$ appears inside the distribution we are sampling from. In both cases, the next step is usually the reparameterization trick, converting a draw from the latent variable distribution as  $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0,I)$, along with the rather obscure reasoning (at least it was to me!) that this reparameterization allows "gradients to flow through the sampled latent variable." 

The reinforcement-learning example and the Gaussian example look very different, but mathematically they share the same structure:

$$
\text{parameters inside the distribution} \quad \Longrightarrow \quad \text{sampling becomes part of the optimisation problem.}
$$

That is the central difference from ordinary supervised learning.

-----

# A biased die example

To clearly explain why having learnable parameters inside the underlying distribution is such a problem,  we will consider a simple dice problem. Consider a six-sided die whose probabilities depend on a scalar parameter $\theta$. Larger $\theta$ makes rolling a 6 more likely.

Let

$$
s(\theta)=\sigma(\theta)=\frac{1}{1+e^{-\theta}}.
$$

Define the die probabilities as

$$
P_\theta(x=6)=s(\theta),
$$

and for the other faces,

$$
P_\theta(x=k)=\frac{1-s(\theta)}{5},
\qquad k \in \{1,2,3,4,5\}.
$$

So all non-6 faces share the remaining mass equally.

A few concrete values help make this more tangible:

- If $\theta=-2$, then $s(\theta)\approx 0.119$, so $P_\theta(x=6)\approx 0.119$ and each of $1,\dots,5$ has probability about $0.176$.
- If $\theta=0$, then $s(\theta)=0.5$, so $P_\theta(x=6)=0.5$ and each of $1,\dots,5$ has probability $0.1$.
- If $\theta=2$, then $s(\theta)\approx 0.881$, so $P_\theta(x=6)\approx 0.881$ and each of $1,\dots,5$ has probability about $0.024$.

Now choose the simplest possible payoff function:

$$
f(x)=x.
$$

So the objective is just the expected die roll:

$$
J(\theta)
=
\mathbb{E}_{x \sim P_\theta}[x].
$$

If we want to maximise $J(\theta)$, we want to make high rolls more likely. If we want to minimise it, we want to make low rolls more likely.

This toy example is small enough that we can solve it analytically. In many real stochastic optimisation problems we cannot, and then we are forced to rely on Monte Carlo estimates - this is where the real difficulty shows up.

-----

# The analytic expectation is easy to differentiate

Analytically, the expectation is just a finite sum:

$$
J(\theta)
=
\sum_{x=1}^6 xP_\theta(x).
$$

Expanding the terms gives

$$
J(\theta)
=
\sum_{x=1}^5 x\frac{1-s(\theta)}{5}
+
6s(\theta).
$$

Since $$ 1+2+3+4+5 = 15 $$, we get

$$
J(\theta)
=
\frac{15}{5}(1-s(\theta)) + 6s(\theta)
=
3(1-s(\theta)) + 6s(\theta)
=
3 + 3s(\theta).
$$

Now the derivative is straightforward:

$$
\frac{dJ}{d\theta}
=
3\frac{ds}{d\theta}.
$$

Since

$$
\frac{ds}{d\theta}
=
s(\theta)(1-s(\theta)),
$$

we have

$$
\frac{dJ}{d\theta}
=
3s(\theta)(1-s(\theta)).
$$

This is a perfectly ordinary derivative. There is no conceptual problem here. The probability distribution depends on $\theta$, but we have expanded the expectation explicitly, so that dependence is visible in the formula.

At $\theta=0$, we have $s(0)=0.5$, so

$$
J(0)=3+3(0.5)=4.5,
$$

and

$$
\left.\frac{dJ}{d\theta}\right\vert_{\theta=0}
=3(0.5)(0.5)=0.75.
$$

So if we were doing gradient ascent on the exact expression, we would increase $\theta$, making 6 more likely and increasing the expected roll.

The important point is:

> When the expectation is available analytically, differentiating with respect to $\theta$ is easy because the probabilities $P_\theta(x)$ explicitly appear in the expression.

-----

# The Monte Carlo estimate breaks naive Autograd

In many interesting problems, the analytic expression above is not available in closed form, or is too expensive to compute exactly. So instead we estimate the expectation by Monte Carlo sampling:

$$
x_1,\ldots,x_n \sim P_\theta.
$$

The Monte Carlo estimate is

$$
\hat{J}(\theta)
=
\frac{1}{n}\sum_{i=1}^n x_i.
$$

This is a perfectly good estimator of the objective $J(\theta)$. But if we now try to differentiate this sampled computation naively using Autograd, we run into a problem.

Suppose $n=1$, and our sampled roll is just

$$
x_1=4.
$$

Then the realised Monte Carlo estimate is

$$
\hat{J}(\theta)
=
\frac{4}{1}=4.
$$

As a realised computation, this is just the number $4$, there is no dependence on $\theta$ because $\theta$ changes the **probability** of seeing $x=4$, but it doesn't change the **value**: nudging $\theta$ slightly does not make that realised sample become, for example, $4.001$. As a result, there is no meaningful value for 
$$
\frac{dx}{d\theta},
$$
which therefore means there is nothing for autograd to differentiate. 

This is the crucial intuition:

> Changing $\theta$ changes the **probability** of seeing a particular roll, but it does not change the value of the roll after it has been sampled.

In a PyTorch-like sketch, the naive, incorrect computation looks like this:

```python
theta = torch.tensor(0.0, requires_grad=True)

s = torch.sigmoid(theta)
probs = torch.cat([
    ((1 - s) / 5) * torch.ones(5),
    s[None],
])

dist = torch.distributions.Categorical(probs=probs)

# A hard sample from the categorical distribution.
x = dist.sample() + 1

# Try to maximise the sampled die roll.
loss = -x.float()
loss.backward()
```

This does not give the gradient of the expected die roll. Depending on how you write it, Autograd will either complain that there is no differentiable path, or it will simply treat the sampled value as a constant. This is to say: **there is no pathwise gradient of $J$**

So the real issue is not that the true objective has no derivative. We already showed that

$$
\frac{dJ}{d\theta}=3s(\theta)(1-s(\theta)).
$$

The issue is this:

> The exact expectation is differentiable, but the naive Monte Carlo computation does not expose that derivative to the automatic differentiation system.

-----

# What we need instead: a gradient estimator

At this point it helps to be precise about the goal. The Monte Carlo average $\hat{J}(\theta)$ is an estimator of the objective $J(\theta)$. But for optimisation, what we really need is not just an estimate of the objective. We need a way to estimate its derivative:

$$
\nabla_\theta J(\theta).
$$

A **gradient estimator** is simply a random quantity built from samples whose expectation equals, or at least approximates, the true gradient we care about.

In symbols, we want some sample-based quantity $g(\theta,\omega)$ such that

$$
\mathbb{E}[g(\theta,\omega)] = \nabla_\theta J(\theta)
$$

in the unbiased case, or at least

$$
g(\theta,\omega) \approx \nabla_\theta J(\theta)
$$

in a useful approximate sense.

This is the key distinction:

- $\hat{J}$ estimates the value of the objective.
- A gradient estimator estimates the derivative of the objective.

The naive sampled die roll is a fine estimator of $J(\theta)$, but it is not a useful pathwise estimator of $\nabla_\theta J(\theta)$.

There are two main fixes:

1. The **score-function estimator**, also known as REINFORCE.
2. The **pathwise estimator**, usually implemented via the reparameterization trick.

They solve the same broad problem in different ways.

-----

# The score-function estimator / REINFORCE

Start again from

$$
J(\theta)
=
\mathbb{E}_{x\sim P_\theta}[f(x)].
$$

For a discrete random variable,

$$
J(\theta)
=
\sum_x P_\theta(x)f(x).
$$

Differentiate:

$$
\nabla_\theta J(\theta)
=
\sum_x \nabla_\theta P_\theta(x)f(x).
$$

Now use the identity

$$
\nabla_\theta P_\theta(x)
=
P_\theta(x)\nabla_\theta \log P_\theta(x).
$$

This gives

$$
\nabla_\theta J(\theta)
=
\sum_x P_\theta(x)f(x)\nabla_\theta \log P_\theta(x).
$$

Returning to expectation notation,

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{x\sim P_\theta}
\left[
f(x)\nabla_\theta \log P_\theta(x)
\right].
$$

This is the score-function estimator.

The important feature is that it does **not** try to compute $dx/d\theta$. Instead, it computes

$$
\nabla_\theta \log P_\theta(x),
$$

which asks:

> How would changing $\theta$ change the log-probability of the sample that I just saw?

That is a meaningful derivative, because $P_\theta(x)$ is a differentiable function of $\theta$, even though the sampled value $x$ itself is discrete.

## REINFORCE on the biased die

For our die,

$$
P_\theta(x=6)=s(\theta),
$$

and

$$
P_\theta(x=k)=\frac{1-s(\theta)}{5},
\qquad k\in\{1,2,3,4,5\}.
$$

The score term is

$$
\nabla_\theta \log P_\theta(x).
$$

If $x=6$, then

$$
\nabla_\theta \log P_\theta(6)
=
\nabla_\theta \log s(\theta)
=
1-s(\theta).
$$

If $x\in\{1,2,3,4,5\}$, then

$$
\nabla_\theta \log P_\theta(x)
=
\nabla_\theta \log \left(\frac{1-s(\theta)}{5}\right)
=
-s(\theta).
$$

Since $f(x)=x$, the one-sample REINFORCE gradient estimator is

$$
g_{\mathrm{RF}}
=
x\nabla_\theta \log P_\theta(x).
$$

So

$$
g_{\mathrm{RF}}
=
\begin{cases}
6(1-s(\theta)), & x=6,\\
-xs(\theta), & x\in\{1,2,3,4,5\}.
\end{cases}
$$

This estimator is noisy, but it is unbiased. Its expectation is

$$
\mathbb{E}[g_{\mathrm{RF}}]
=
6s(\theta)(1-s(\theta))
+
\sum_{x=1}^5
\frac{1-s(\theta)}{5}
\left(-xs(\theta)\right).
$$

Using $1+2+3+4+5=15$,

$$
\mathbb{E}[g_{\mathrm{RF}}]
=
6s(\theta)(1-s(\theta))
-
3s(\theta)(1-s(\theta))
=
3s(\theta)(1-s(\theta)),
$$

which matches the exact derivative:

$$
\frac{dJ}{d\theta}
=
3s(\theta)(1-s(\theta)).
$$

So REINFORCE gives us the right gradient in expectation, even though the sample itself was not differentiable.

In code, the important difference is that the loss uses the log-probability of the sampled outcome:

```python
theta = torch.tensor(0.0, requires_grad=True)

s = torch.sigmoid(theta)
probs = torch.cat([
    ((1 - s) / 5) * torch.ones(5),
    s[None],
])

dist = torch.distributions.Categorical(probs=probs)

# Sample a face index 0,...,5, then convert to die face 1,...,6.
idx = dist.sample()
x = idx + 1

# This is differentiable with respect to theta.
logp = dist.log_prob(idx)

# Gradient ascent on expected roll = gradient descent on negative objective.
# x is used as a sampled weight, not as a differentiable path.
loss = -(x.float().detach() * logp)
loss.backward()
```

Now Autograd does have a differentiable path:

$$
\theta \longrightarrow P_\theta(x) \longrightarrow \log P_\theta(x) \longrightarrow x \log P_\theta(x).
$$

The sampled value $x$ is used as a weight. The gradient flows through the log-probability, not through the sampled integer.

A useful nuance is that a single REINFORCE sample is generally **not** the exact true gradient, and in higher-dimensional problems it need not be perfectly aligned with the true gradient sample-by-sample either. What matters is that, on average, it points in the correct direction:

$$
\mathbb{E}[g_{\mathrm{RF}}]=\nabla_\theta J(\theta).
$$

That is the sense in which REINFORCE is a valid gradient estimator.

This is exactly the same pattern used in policy-gradient reinforcement learning. A stochastic policy samples actions, the environment returns rewards, and the update takes the form

$$
\nabla_\theta J(\theta)
\approx
R \nabla_\theta \log \pi_\theta(a \mid s)
$$

or, over full trajectories,

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau \sim p_\theta(\tau)}
\left[
R(\tau)\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)
\right].
$$

The return says whether the sampled behaviour was good. The score term says how to change $\theta$ to make that sampled behaviour more or less likely in the future.

-----

# The pathwise estimator / reparameterization trick

The second solution is the pathwise derivative estimator. This is the setting where the reparameterization trick is usually used.

Suppose that instead of a discrete die roll, we have a continuous random variable such as

$$
x \sim \mathcal{N}(\mu_\theta,\sigma_\theta^2).
$$

If we sample $x$ directly from this parameterized distribution, we again have the same issue: the computation contains a stochastic sampling operation. But for a Gaussian, we can rewrite the sample as

$$
\epsilon \sim \mathcal{N}(0,1),
$$

$$
x = \mu_\theta + \sigma_\theta \epsilon.
$$

Now the randomness comes from $\epsilon$, whose distribution does not depend on $\theta$. The sampled value $x$ is a deterministic differentiable function of $\theta$ and $\epsilon$.

So, for an objective

$$
J(\theta)
=
\mathbb{E}_{x\sim P_\theta}[f(x)],
$$

we rewrite it as

$$
J(\theta)
=
\mathbb{E}_{\epsilon}
\left[
f(\mu_\theta + \sigma_\theta\epsilon)
\right].
$$

Now a Monte Carlo estimator is

$$
\hat{J}(\theta)
=
\frac{1}{n}\sum_{i=1}^n
f(\mu_\theta + \sigma_\theta\epsilon_i),
\qquad
\epsilon_i \sim \mathcal{N}(0,1).
$$

This turns the sampled computation into an ordinary differentiable graph:

$$
\theta
\longrightarrow
\mu_\theta,\sigma_\theta
\longrightarrow
\mu_\theta+\sigma_\theta\epsilon_i
\longrightarrow
f(\mu_\theta+\sigma_\theta\epsilon_i).
$$

So Autograd can compute the derivative by the chain rule.

For example, if

$$
f(x)=x^2,
$$

then for one sample

$$
\hat{J}(\theta)
=
(\mu_\theta + \sigma_\theta\epsilon)^2.
$$

This has a perfectly ordinary derivative with respect to $\theta$, because $\epsilon$ is treated as fixed noise during the backward pass.

In code:

```python
theta = torch.tensor(0.0, requires_grad=True)

mu = theta
sigma = torch.exp(0.5 * theta)

eps = torch.randn(())
x = mu + sigma * eps

loss = x ** 2
loss.backward()
```

This is the essence of the reparameterization trick:

> Move $\theta$ out of the sampling distribution and into a deterministic differentiable transformation of fixed noise.

This gives us another gradient estimator, this time by restoring an ordinary pathwise derivative through the sampled computation.

-----

# Why ordinary reparameterization does not fix the hard die example

It is tempting to ask: can we reparameterize the die too? In a weak sense, yes. We can always generate a die roll from a uniform random variable $u \sim \mathrm{Uniform}(0,1)$. But the mapping from $u$ to the die face is a step function, not a smooth differentiable transformation.

For the hard die roll,

$$
x = g_\theta(u)
$$

jumps between integer values. Therefore the pathwise derivative is zero almost everywhere and undefined at the jump points.

That is not useful for gradient-based optimisation.

This is why discrete distributions typically use score-function estimators such as REINFORCE. There are continuous relaxations, such as the Gumbel-Softmax / Concrete distribution, which replace a hard categorical sample with a differentiable soft sample, but then the gradient corresponds to the relaxed problem rather than the exact discrete one.

So for the exact hard die:

$$
\text{Use REINFORCE, not the ordinary pathwise estimator.}
$$

For a Gaussian sample:

$$
\text{Use the pathwise estimator, because the sample can be written as a smooth function of fixed noise.}
$$

-----

# When to use each approach

The two approaches solve the same broad problem in different regimes.

| Setting | Typical approach | Reason |
|---|---|---|
| Discrete sample, e.g. categorical action or die roll | Score-function / REINFORCE | The sampled value is not smoothly differentiable with respect to $\theta$. |
| Continuous sample with differentiable reparameterization, e.g. Gaussian | Pathwise / reparameterization trick | The sample can be written as $x=g_\theta(\epsilon)$, so Autograd can differentiate through $g_\theta$. |
| Black-box environment or unknown dynamics | Often score-function | We only observe sampled trajectories and returns. |
| Differentiable model or simulator | Often pathwise | Gradients can flow through the sampled path. |
| Both stochastic and differentiable nodes | Combination | Stochastic computation graphs combine score-function and pathwise terms. |

The score-function estimator is more general. It can handle discrete random variables and discontinuous downstream functions. However, it often has high variance.

The pathwise estimator is less general, because it requires differentiability, but when applicable it is usually much lower variance and more directly compatible with neural-network training.

-----

# Connection to reinforcement learning

In reinforcement learning, the most common version of this problem is policy optimisation. A stochastic policy samples actions:

$$
a_t \sim \pi_\theta(\cdot \mid s_t).
$$

The objective is expected return:

$$
J(\theta)
=
\mathbb{E}_{\tau\sim \pi_\theta}[R(\tau)].
$$

The trajectory $\tau$ depends on $\theta$ because the actions are sampled from $\pi_\theta$. If the action is discrete, such as choosing a move in a board game, the sampled action is like the die roll above. Once the action is sampled, it is just an integer or category. We cannot ask Autograd to compute a useful derivative of the action itself with respect to $\theta$.

So policy-gradient methods use the score-function identity:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim \pi_\theta}
\left[
R(\tau)\nabla_\theta \log p_\theta(\tau)
\right].
$$

Under the usual assumption that the environment dynamics do not depend on $\theta$, this becomes

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim \pi_\theta}
\left[
R(\tau)
\sum_t
\nabla_\theta \log \pi_\theta(a_t\mid s_t)
\right].
$$

This is REINFORCE.

For continuous-control methods with differentiable stochastic policies, we can sometimes use the pathwise form instead. For example, a Gaussian policy can be written as

$$
a_t
=
\mu_\theta(s_t)
+
\sigma_\theta(s_t)\epsilon_t,
\qquad
\epsilon_t\sim \mathcal{N}(0,I).
$$

Then gradients can flow through the sampled action into $\mu_\theta$ and $\sigma_\theta$, assuming the rest of the objective is differentiable or has been approximated by a differentiable critic. This idea is central to stochastic value-gradient methods and to algorithms such as Soft Actor-Critic.

For more detail, see:

- [OpenAI Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [Learning Continuous Control Policies by Stochastic Value Gradients](https://arxiv.org/abs/1510.09142)
- [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)

-----

# Connection to variational autoencoders

Variational autoencoders are one of the clearest places to see why the reparameterization trick matters. A VAE introduces a latent variable $z$ and optimises a variational lower bound of the form

$$
\mathcal{L}(x;\theta,\phi)
=\mathbb{E}_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
- D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big).
$$

The reconstruction term (first term on the RHS) is an expectation over the approximate posterior $q_\phi(z\mid x)$. That makes it a stochastic optimisation problem of exactly the type we have been discussing: the encoder parameters $\phi$ live inside the distribution we are sampling from. If we naively sample

$$
z \sim q_\phi(z\mid x)
$$

and plug that sample into $\log p_\theta(x\mid z)$, then we are back in the same situation as before. The sampled latent code is just a realised value, and the sampling step does not expose an ordinary differentiable path from $\phi$ to that value.

The key move is to choose an approximate posterior that can be written in reparameterized form. For a diagonal Gaussian posterior, the encoder outputs $\mu_\phi(x)$ and $\sigma_\phi(x)$, and we sample using

$$
\epsilon \sim \mathcal{N}(0,I),
\qquad
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon.
$$

Now the randomness sits in $\epsilon$, whose distribution does not depend on $\phi$, while the latent variable $z$ is a deterministic differentiable function of the encoder outputs and the sampled noise. That turns a problematic stochastic node into an ordinary computation graph.

So in VAE training, the reparameterization trick is not just a cosmetic rewrite of a Gaussian. It is what makes low-variance stochastic gradient optimisation of the ELBO practical with standard backpropagation. This is exactly the role highlighted in Kingma and Welling's paper, which introduces a reparameterized lower-bound estimator and the Auto-Encoding Variational Bayes algorithm: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).

-----

# Connection to diffusion models

Diffusion models also repeatedly use Gaussian reparameterizations. The forward noising process is usually written as

$$
q(x_t\mid x_{t-1})
=
\mathcal{N}\left(\sqrt{1-\beta_t}x_{t-1},\beta_t I\right).
$$

Because this is Gaussian, it can be written as

$$
x_t
=
\sqrt{1-\beta_t}x_{t-1}
+
\sqrt{\beta_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

After recursively merging the Gaussian noise terms, we get the closed-form expression

$$
x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

This form is useful because we can sample a noisy image at any timestep directly. But it also reflects the same general principle: instead of treating the sample as a mysterious draw from a parameterized Gaussian, write it as a deterministic function of known quantities and fixed Gaussian noise.

In the reverse process, diffusion models often define a learned Gaussian transition such as

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}\left(\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)\right).
$$

If we sample from this Gaussian, the corresponding reparameterized form is

$$
x_{t-1}
=
\mu_\theta(x_t,t)
+
\Sigma_\theta(x_t,t)^{1/2}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

Now the sample is a differentiable function of $\theta$, provided the mean and variance networks are differentiable. This is why Gaussian distributions are especially convenient in modern generative modelling: they are easy to sample from, easy to evaluate, and easy to reparameterize. For a detailed diffusion-model derivation, see Lilian Weng's post: [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).

-----

# Stochastic computation graphs and the broader picture

The broader framework here is stochastic optimisation on stochastic computation graphs.

A normal neural-network computation graph has deterministic nodes:

$$
\theta \longrightarrow h_1 \longrightarrow h_2 \longrightarrow L.
$$

Backpropagation works because each node is a deterministic differentiable function of its parents.

A stochastic computation graph adds stochastic nodes. For example:

$$
\theta \longrightarrow x \sim P_\theta \longrightarrow f(x).
$$

Now the graph contains a random sampling step. To compute gradients of expected costs in such graphs, we need sample-based ways of estimating $\nabla_\theta J(\theta)$.

That is exactly where the two core ideas fit:

1. **Score-function terms** for stochastic nodes whose probabilities depend on $\theta$:
   $$
   f(x)\nabla_\theta \log P_\theta(x).
   $$

2. **Pathwise derivative terms** for sampled quantities that can be rewritten as differentiable functions of fixed noise:
   $$
   \nabla_\theta f(g_\theta(\epsilon)).
   $$

Schulman et al. unify these ideas in [Gradient Estimation Using Stochastic Computation Graphs](https://arxiv.org/pdf/1506.05254). Their perspective is useful because real models often contain both kinds of structure: some stochastic nodes are discrete and need score-function terms, while others are continuous and can be reparameterized.

So the reparameterization trick is best viewed not as an isolated Gaussian trick, but as one tool inside the broader area of stochastic optimisation.

-----

# Summary

The reparameterization trick is not just a small Gaussian trick. It solves a specific computational problem, similarly for REINFORCE. In a regular objective such as

$$
\mathbb{E}_{(x,y)\sim\mathcal{D}}
\left[(y-\hat{y}_\theta(x))^2\right],
$$

samples from the data distribution are fixed inputs, and $\theta$ appears inside a deterministic differentiable model. Minibatch training works naturally with backpropagation.

But in objectives such as

$$
\mathbb{E}_{x\sim P_\theta}[f(x)],
$$

$\theta$ controls the sampling distribution itself. Analytically, we can expand the expectation as

$$
\sum_x P_\theta(x)f(x)
$$

and differentiate it directly. The mathematical expectation is not the problem.

The problem appears when we estimate the expectation using Monte Carlo samples. A realised sample $x$ is just a value. Changing $\theta$ changes the **probability** of seeing that value, but not the value itself. So the naive pathwise quantity $dx/d\theta$ is either zero, undefined, or simply the wrong object.

To optimise these objectives, we need a sample-based way to estimate the derivative of the expectation.

- For discrete samples, use the score-function estimator / REINFORCE:

$$
\nabla_\theta \mathbb{E}_{x\sim P_\theta}[f(x)]
=
\mathbb{E}_{x\sim P_\theta}
\left[
f(x)\nabla_\theta\log P_\theta(x)
\right].
$$

- For continuous differentiable samples, use the pathwise estimator / reparameterization trick:

$$
x=g_\theta(\epsilon),
\qquad
\epsilon\sim P(\epsilon),
$$

so that

$$
\nabla_\theta \mathbb{E}_\epsilon[f(g_\theta(\epsilon))]
=
\mathbb{E}_\epsilon
\left[
\nabla_\theta f(g_\theta(\epsilon))
\right].
$$

The shortest version is:

> Backpropagation differentiates through realised computational paths. If the path contains a raw parameter-dependent sampling step, ordinary backpropagation has no useful derivative through that step. REINFORCE differentiates the probability of the sampled path. Reparameterization rewrites the sampled path so that it becomes differentiable.

That is the core reason these methods exist.

-----

# References

- [OpenAI Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
- [Lilian Weng: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Schulman et al., Gradient Estimation Using Stochastic Computation Graphs](https://arxiv.org/pdf/1506.05254)
- [Kingma and Welling, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Heess et al., Learning Continuous Control Policies by Stochastic Value Gradients](https://arxiv.org/abs/1510.09142)
- [Haarnoja et al., Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
