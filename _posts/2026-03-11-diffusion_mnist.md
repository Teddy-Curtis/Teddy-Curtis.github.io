---
layout: post
title: Diffusion models with MNIST
date: 2026-03-11 10:45:00
description: Fundamentals of diffusion models exemplified with MNIST 
tags: Diffusion MNIST PyTorch 
categories: Diffusion
disqus_comments: true
toc:
  beginning: true
tabs: true
---

# Diffusion Models from Scratch: Generating MNIST Digits

Diffusion models have recently become one of the most successful classes
of generative models in machine learning. They are used in systems such
as **image generation models (e.g. Stable Diffusion, DALL·E)**, **audio
synthesis**, **video generation**, and even **protein structure
modelling**. Their strength lies in stable training, strong sample
quality, and a mathematically elegant probabilistic formulation.

In this post, we will:

1.  Introduce what diffusion models are used for
2.  Derive the core mathematics behind diffusion models
3.  Explain how the models are trained in practice
4.  Walk through a **minimal MNIST implementation.**
5.  Discuss improvements and limitations

The target audience is readers comfortable with machine learning and
probability, but new to diffusion models. This post also includes an accompanying Python script, which you can find in the Google [Colab notebook](https://colab.research.google.com/drive/1B5VAw2z6iG9qK1sVW6YQFekXL4qy3XWz?usp=sharing). 

------------------------------------------------------------------------

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/posts/2026-03-011-diffusion_mnist/overview.png" class="img-fluid rounded z-depth-1 w-100 d-block mx-auto" zoomable=true caption="Figure 1: overview of the diffusion pipeline, where we know the forward diffusion process given by $q(x_t|x_{t-1})$, and we want to train a model to learn the opposite process $ p_{\theta}(x_{t-1}| x_{t}) $." %}
    </div>
</div>

# **What Are Diffusion Models?**

Diffusion models are **generative models** that learn how to transform
noise into data.

The idea, exemplified in Figure 1 above, is conceptually simple:

1.  Take real data (an image).
2.  **Forward diffusion**: Gradually add Gaussian noise following a known probability distribution $q(x_t \mid x_{t-1})$, until the image becomes pure noise.
3.  **Reverse diffusion**: Train a neural network to **reverse this noising process**, which is represented as $ p_\theta(x_{t-1} \mid x_t) $ where $$ \theta $$ are the learnable parameters.

After training, we can start from random noise and repeatedly denoise it to generate new images.

------------------------------------------------------------------------

## **The Forward Diffusion Process**

The goal of the forward diffusion process is to gradually destroy the structure of the data by adding Gaussian noise over many steps.

A first idea might be to define the process by simply adding Gaussian noise at each step:

$$
x_t = x_{t-1} + \epsilon_t
$$

where

$$
\epsilon_t \sim \mathcal{N}(0, I)
$$

This certainly makes the image noisier over time, but it has a serious problem: the variance keeps increasing. If we keep adding independent Gaussian noise, the total variance grows with the number of steps, so the scale of the sample becomes uncontrolled.

To prevent this, diffusion models do two things at every step:

1. shrink the previous sample slightly
2. add back a controlled amount of fresh Gaussian noise

This gives the forward update

$$
x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}
$$

with

$$
\epsilon_{t-1} \sim \mathcal{N}(0,I)
$$

and

$$
\alpha_t = 1 - \beta_t \quad \text{with} \quad 0 < \alpha_t < 1 
$$

where $$ \beta_t $$ is a small positive number that controls how much noise is injected at step $$t$$.

We now assume that $$ x_{t-1} $$ has unit variance in coordinate, so that 

$$
\mathrm{Var}(\sqrt{\alpha_t}\,x_{t-1}) = \alpha_t \,\mathrm{Var}(x_{t-1}) = \alpha_t.
$$

and because $$ \epsilon_{t-1} \sim \mathcal{N}(0,I) $$, we get 

$$
\mathrm{Var}\!\left(\sqrt{1-\alpha_t}\,\epsilon_{t-1}\right) = (1-\alpha_t)I.
$$

As $$ x_{t-1} $$ and $$ \epsilon_{t-1} $$ are independent, we see that the variance of $$ x_t $$ is indeed constant:

$$
\mathrm{Var}(x_t) = \alpha_t + (1-\alpha_t) = 1.
$$

Essentially, the factor $$ \sqrt{\alpha_t} $$ damps the signal from the previous step, reducing the variance carried forward from $$ x_{t-1} $$, and the factor $$ \sqrt{1-\alpha_t} $$ means that the missing fraction of variance is recovered without growing uncontrollably. 


## **Equivalent Gaussian form**

The update

$$
x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}
$$

is equivalent to sampling from a Gaussian distribution with mean $$ \sqrt{\alpha_t}\,x_{t-1} $$ and covariance $$ (1-\alpha_t)I $$:

$$
q(x_t \mid x_{t-1}) =
\mathcal{N}\!\left(\sqrt{\alpha_t}\,x_{t-1}, (1-\alpha_t)I\right)
$$

Using $$ \alpha_t = 1-\beta_t $$, this is usually written as

$$
q(x_t \mid x_{t-1}) =
\mathcal{N}\!\left(\sqrt{1-\beta_t}\,x_{t-1}, \beta_t I\right)
$$

This is the standard forward diffusion transition used in DDPM.

---

## **Sampling Any Timestep Directly**

A very useful property of the forward process is that we can sample $$x_t$$ directly from the original image $$x_0$$, without simulating every intermediate step.


Start from the one-step forward update:

$$
x_t = \sqrt{\alpha_t}x_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t-1}
$$

Substitute in the expression for $$x_{t-1}$$:

$$
x_{t-1} = \sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}
$$

Then

$$
x_t
= \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}\right)
+ \sqrt{1-\alpha_t}\epsilon_{t-1}
$$

so

\begin{equation}
\label{beforeGaussSum}
x_t = \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-2} + \sqrt{1-\alpha_t}\,\epsilon_{t-1}
\end{equation}


At this point we have two independent Gaussian noise terms:

$$
\sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-2}
\quad\text{and}\quad
\sqrt{1-\alpha_t}\,\epsilon_{t-1}
$$

Because both are zero-mean independent Gaussians, their sum is also Gaussian. This is the Gaussian-merging step. Suppose

$$
z_1 \sim \mathcal{N}(0,\sigma_1^2 I), \qquad
z_2 \sim \mathcal{N}(0,\sigma_2^2 I)
$$

and $$z_1$$ and $$z_2$$ are independent. Then

$$
z_1 + z_2 \sim \mathcal{N}(0,(\sigma_1^2+\sigma_2^2)I)
$$

Here,

$$
z_1 = \sqrt{\alpha_t(1-\alpha_{t-1})}\,\epsilon_{t-2}
\quad\Rightarrow\quad
z_1 \sim \mathcal{N}(0,\alpha_t(1-\alpha_{t-1})I)
$$

and

$$
z_2 = \sqrt{1-\alpha_t}\,\epsilon_{t-1}
\quad\Rightarrow\quad
z_2 \sim \mathcal{N}(0,(1-\alpha_t)I)
$$

So their sum is

$$
z_1 + z_2 \sim
\mathcal{N}\!\left(0,
\left[\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t)\right]I\right)
$$

Simplifying the variance:

$$
\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t)
= \alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t
= 1 - \alpha_t\alpha_{t-1}
$$

Therefore, we can rewrite the two-noise expression, equation $$ \eqref{beforeGaussSum} $$ as a single Gaussian noise term:

$$
x_t
= \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2}
+ \sqrt{1-\alpha_t\alpha_{t-1}}\,\bar{\epsilon}
$$

where

$$
\bar{\epsilon} \sim \mathcal{N}(0,I)
$$

Repeating this argument recursively gives

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon
$$

with

$$
\epsilon \sim \mathcal{N}(0,I) \quad \text{and} \quad \bar{\alpha}_t = \prod_{i=1}^t \alpha_i
$$

Therefore the full conditional distribution of $$x_t$$ given the original image $$x_0$$ is

$$
q(x_t \mid x_0) =
\mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}\,x_0, (1-\bar{\alpha}_t)I\right)
$$

## **Why this matters for training**

This closed-form expression is one of the main reasons diffusion models are practical to train.

Instead of simulating the entire Markov chain

$$
x_0 \to x_1 \to x_2 \to \cdots \to x_t
$$

we can sample any noisy version $$x_t$$ directly in a single step:

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon
$$

This is exactly the quantity computed by functions such as `q_sample` in a PyTorch diffusion implementation.

------------------------------------------------------------------------

# **Why Predict Noise?**

Instead of predicting $x_0$ directly, diffusion models typically predict
the **noise component** $\epsilon$. That is, given a noisy image, the model tries to predict the noise that was added to the original image. 

A simplified training objective would then be 

$$
L = E_{x_0,t,\epsilon}
||\epsilon - \epsilon_\theta(x_t,t)||^2
$$

This is simply **mean squared error between the true noise and predicted
noise**.

Predicting noise works well because:

-   the noise distribution is simple (Gaussian)
-   the model avoids learning the raw data distribution directly


# **How the Model Generates an Image**

Once the model has been trained to predict the noise

$$
\epsilon_\theta(x_t, t)
$$

we can use it to **reverse the diffusion process** and generate new images.

The goal is to sample from the reverse transition

$$
p_\theta(x_{t-1} \mid x_t)
$$

which gradually removes noise from the image.

---



### **Reverse Diffusion Distribution**

The reverse process is modeled as a Gaussian:

$$
p_\theta(x_{t-1} \mid x_t)
=
\mathcal{N}(\mu_\theta(x_t,t), \sigma_t^2 I)
$$

The key result from the DDPM derivation is that the mean of this Gaussian can be written in terms of the predicted noise:

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t)
\right)
$$

Here:

- $x_t$ is the noisy image at timestep $t$
- $\epsilon_\theta(x_t,t)$ is the noise predicted by the neural network
- $\alpha_t = 1-\beta_t$
- $\bar{\alpha}_t = \alpha_0 *\alpha_1... * \alpha_t$

This equation tells us how to compute the **mean of the previous timestep** given the current noisy image and the predicted noise.

An important intuition is that a noisy image $x_t$ does **not uniquely determine** the previous image $x_{t-1}$. Many slightly less noisy images could plausibly have produced the current noisy observation.

Because of this ambiguity, the reverse diffusion step is modeled as a **probability distribution** rather than a deterministic function. The neural network therefore predicts the **mean of this distribution**—the most likely estimate of what the previous image could have been. The variance of the Gaussian captures the remaining uncertainty in that estimate.

---

### **How the Variance is Determined**

The variance of the reverse transition is not learned in the original DDPM formulation. Instead, it is taken from the **exact posterior variance of the forward diffusion process**.

Recall that the forward diffusion step is

$$
q(x_t \mid x_{t-1}) =
\mathcal{N}(\sqrt{\alpha_t}x_{t-1}, \beta_t I)
$$

and that the marginal distribution of a noisy image given the original image is

$$
q(x_t \mid x_0) =
\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)
$$

From these two expressions we can derive the posterior distribution

$$
q(x_{t-1} \mid x_t, x_0)
$$

which describes the distribution of the previous timestep given both the current noisy image and the original clean image. The full derivation can be found in the extremely useful blog by [Lilian Weng](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).

This posterior is also Gaussian:

$$
q(x_{t-1} \mid x_t, x_0)
=
\mathcal{N}(\tilde{\mu}_t(x_t,x_0), \tilde{\beta}_t I)
$$

where the variance is

$$
\tilde{\beta}_t
=
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$

This quantity is known as the **posterior variance** of the forward diffusion process.

In the DDPM model we do not know $x_0$ during generation, so we cannot use the exact posterior mean $\tilde{\mu}_t(x_t,x_0)$. Instead, we approximate it using the neural network's noise prediction.

However, the posterior *variance* does not depend on $x_0$, so it can be used directly as the variance of the learned reverse transition. Therefore, the reverse diffusion distribution becomes

$$
p_\theta(x_{t-1} \mid x_t)
=
\mathcal{N}(\mu_\theta(x_t,t), \tilde{\beta}_t I)
$$

where

$$
\tilde{\beta}_t
=
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t.
$$

This is the variance used when sampling the previous timestep during the reverse diffusion process.



### **Sampling the Previous Step**

Once we compute the mean of the reverse distribution, we sample the next image in the reverse chain:

$$
x_{t-1}
=
\mu_\theta(x_t,t)
+
\sigma_t z
$$

where

$$
z \sim \mathcal{N}(0,I)
$$

At first glance this may look like we are adding noise again after removing it, but this is **not the same as the forward diffusion process**.

In the forward process, noise is added to **destroy information** in the image. In the reverse process, the Gaussian term represents the **uncertainty in which previous image could have produced the current noisy image**.

A given noisy image $x_t$ could have come from many possible slightly cleaner images $x_{t-1}$. The model predicts the center of this distribution (the mean), and the random term samples one plausible image from that distribution.

This stochastic sampling is important because the reverse process is inherently probabilistic. If we removed the random term and always set

$$
x_{t-1} = \mu_\theta(x_t,t)
$$

the sampler would become deterministic. Some diffusion samplers such as DDIM take advantage of this idea, but the original DDPM formulation samples from the Gaussian at each step.

## **Full Image Generation Algorithm**

To generate a new image:

1. Start from pure Gaussian noise $$ x_T \sim \mathcal{N}(0,I) $$

2. Repeatedly apply the reverse diffusion step $$ x_{t-1} \sim p_\theta(x_{t-1} \mid x_t) $$ for $$ t = T, T-1,..., 1 $$

3. After the final step we obtain $$ x_0 $$ which is a generated sample from the learned data distribution.

---

### **Intuition**

The forward process gradually **destroys structure** by adding noise.

The neural network learns to **predict the noise that was added at each step**.

Using this prediction, we can iteratively remove noise, transforming random noise into a coherent image.

In practice, this means that starting from pure noise, the model slowly reveals structure until a recognizable image (such as a handwritten MNIST digit) emerges.


------------------------------------------------------------------------
# **Connecting the Theory to the Code**

Now we will connect the diffusion equations from the previous sections to the MNIST implementation.. The goal of this part of the post is not just to show the code, but to make explicit exactly which mathematical object each block of code is computing.

Throughout this section, the key theoretical quantities are:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(\sqrt{\alpha_t}x_{t-1}, \beta_t I\right)
$$

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I\right)
$$

$$
L = \mathbb{E}_{x_0,t,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t,t)\|^2\right]
$$

and

$$
p_\theta(x_{t-1} \mid x_t)
=
\mathcal{N}\!\left(
\mu_\theta(x_t,t),
\tilde{\beta}_t I
\right)
$$

with

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t)
\right)
$$

and

$$
\tilde{\beta}_t
=
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t.
$$

Each code block below implements one piece of this pipeline.

---

## Hyperparameters and Diffusion Schedule

```python
image_size  = 28
channels    = 1
batch_size  = 128
epochs      = 20
lr          = 2e-4
T           = 1000

beta_start = 1e-4
beta_end = 0.02

betas = torch.linspace(beta_start, beta_end, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

sqrt_reciprocal_alphas = torch.sqrt(1.0 / alphas)
alpha_bars_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bars[:-1]], dim=0)
posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
```

This block constructs the diffusion schedule. The line

```python
betas = torch.linspace(beta_start, beta_end, T, device=device)
```

defines the sequence

$$
\beta_1, \beta_2, \dots, \beta_T
$$

which controls how much noise is added at each forward step.

The code then defines

$$
\alpha_t = 1-\beta_t
$$

via

```python
alphas = 1.0 - betas
```

and the cumulative product

$$
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s
$$

via

```python
alpha_bars = torch.cumprod(alphas, dim=0)
```

These quantities appear in the closed-form forward noising equation

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon
$$

which is why the code precomputes

$$
\sqrt{\bar{\alpha}_t}
\qquad \text{and} \qquad
\sqrt{1-\bar{\alpha}_t}.
$$

Finally, the line

```python
posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
```

implements the reverse-process variance

$$
\tilde{\beta}_t
=
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t,
$$

which is the variance of the exact posterior

$$
q(x_{t-1}\mid x_t, x_0).
$$

So this whole block is setting up the scalar coefficients that appear throughout both the forward and reverse diffusion equations.

---

## Gathering Timestep-Dependent Scalars

```python
def gatherAndExpand(a, timesteps):
    out = a.gather(0, timesteps)
    return out.view(-1, 1, 1, 1)
```

In the mathematics, quantities such as $$ \beta_t $$, $$ \alpha_t $$, and $$ \bar{\alpha}_t $$ are indexed by a timestep $$t$$. In the code, each image in a batch may have a different timestep, so we need a way to extract the correct scalar for each batch element and reshape it so it can multiply an image tensor.

This function takes a tensor such as

$$
[\beta_1, \beta_2, \dots, \beta_T]
$$

and a batch of timesteps

$$
[t^{(1)}, t^{(2)}, \dots, t^{(B)}],
$$

and returns

$$
[\beta_{t^{(1)}}, \beta_{t^{(2)}}, \dots, \beta_{t^{(B)}}]
$$

reshaped to broadcast across the spatial dimensions of the image tensor.

So whenever the code writes something like

```python
betas_t = gatherAndExpand(betas, t)
```

it is turning the abstract mathematical quantity $$ \beta_t $$ into a tensor that can be multiplied with each image in the batch.

---

## MNIST Dataset

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2.0 - 1.0),
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
```

The main mathematical point here is the normalization step

```python
transforms.Lambda(lambda x: x * 2.0 - 1.0)
```

which rescales MNIST images from $$[0,1]$$ to $$[-1,1]$$.

That means the clean training image satisfies

$$
x_0 \in [-1,1]^{28\times 28}.
$$

This is useful because the diffusion process repeatedly adds Gaussian noise, so it is convenient for the clean data to already live on a roughly zero-centered scale.

---

## Forward Noising: Implementing $$q(x_t \mid x_0)$$

```python
def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_ab_t = gatherAndExpand(sqrt_alpha_bars, t)
    sqrt_1mab_t = gatherAndExpand(sqrt_one_minus_alpha_bars, t)

    xt = sqrt_ab_t * x0 + sqrt_1mab_t * noise
    return xt, noise
```

This function is one of the cleanest places where the code matches the theory exactly. It implements the closed-form sampling equation

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon
$$

with

$$
\epsilon \sim \mathcal{N}(0,I).
$$

The correspondence is:

$$
\text{x0} \leftrightarrow x_0
$$

$$
\texttt{noise} \leftrightarrow \epsilon
$$

$$
\texttt{sqrt_ab_t} \leftrightarrow \sqrt{\bar{\alpha}_t}
$$

$$
\texttt{sqrt_1mab_t} \leftrightarrow \sqrt{1-\bar{\alpha}_t}
$$

$$
\texttt{xt} \leftrightarrow x_t.
$$

So the line

```python
xt = sqrt_ab_t * x0 + sqrt_1mab_t * noise
```

is exactly

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon.
$$

The reason this function is called `q_sample` is that it samples from the forward diffusion distribution

$$
q(x_t \mid x_0).
$$

This is the closed-form version of the forward noising process derived earlier, and it is what makes diffusion training practical: we can jump directly to any timestep $$t$$ without explicitly simulating all intermediate steps.

---

## Visualizing the Forward Process

```python
x0 = train_dataset[0][0].unsqueeze(0).to(device)
ts = torch.tensor([0, 50, 200, 500], device=device)
x_noisy, _ = q_sample(x0.repeat(4,1,1,1), ts)

# plot images
fig, axs = plt.subplots(2, 2, figsize=(5, 4))
axs = axs.flatten()
for i in range(4):
    axs[i].imshow(x_noisy[i].cpu().squeeze(), cmap='gray')
    axs[i].set_title(f"Noisy Image (t={ts[i].item()})")
    axs[i].axis('off')
plt.savefig("imagine_noising_example.png", bbox_inches="tight")
plt.show()
```

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/posts/2026-03-011-diffusion_mnist/image_noising_example.png" class="img-fluid rounded z-depth-1 w-50 d-block mx-auto" zoomable=true caption="Figure 2: Example of an image at different noise timesteps." %}
    </div>
</div>


This is just a visualization, but it is worth noting what it means mathematically. The code is taking one clean image $$x_0$$ and evaluating

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon
$$

at several different values of $$t$$.

As $$t$$ increases, $$ \bar{\alpha}_t $$ becomes smaller, so the coefficient on $$x_0$$ decreases while the coefficient on the noise increases. In other words,

- early timesteps retain most of the original image,
- later timesteps are dominated by Gaussian noise.

This matches the theoretical picture of the forward process gradually destroying information.

---

## The Conditional U-Net

```python

# ============================================================
# 5. Conditional U-Net
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class ConditionalUNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, base=128, out_channels=1):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, 16)
        self.label_proj = nn.Linear(16, image_size * image_size)

        self.enc1 = ConvBlock(in_channels, base)
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base, base * 2)
        self.down2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 2, base * 4)

        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 4, base * 2)

        self.up2 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x, t, y):
        t = t.float() / (T - 1)
        t_channel = t[:, None, None, None].expand(-1, 1, x.shape[2], x.shape[3])

        y_emb = self.label_emb(y)                              # [B,16]
        y_proj = self.label_proj(y_emb)                        # [B,784]
        y_channel = y_proj.view(-1, 1, image_size, image_size)

        x_in = torch.cat([x, t_channel, y_channel], dim=1)    # [B,3,28,28]

        e1 = self.enc1(x_in)
        e2 = self.enc2(self.down1(e1))
        b = self.bottleneck(self.down2(e2))

        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return self.out(d2)

# ============================================================
# 6. Initialise model and optimizer
# ============================================================

model = ConditionalUNet(num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

The purpose of the neural network is to approximate

$$
\epsilon_\theta(x_t,t,y),
$$

that is, the noise present in the current noisy image $$x_t$$, conditioned on both the timestep $$t$$ and the class label $$y$$.

So the mathematical input-output relation of the model is

$$
(x_t, t, y) \mapsto \epsilon_\theta(x_t,t,y).
$$

This implementation supplies the model with three input channels:

1. the noisy image $$x_t$$,
2. a channel containing the normalized timestep $$t/(T-1)$$,
3. a channel containing an embedding of the digit label $$y$$.

Thus the line

```python
x_in = torch.cat([x, t_channel, y_channel], dim=1)
```

constructs the network input

$$
x_{\text{in}} = \text{concat}(x_t,\; \text{time channel},\; \text{label channel}).
$$

The U-Net architecture then maps this back to a single output image of shape $$28 \times 28$$, which is interpreted as the predicted noise field

$$
\epsilon_\theta(x_t,t,y).
$$

This is why the network output has the same shape as the input image: the model predicts one noise value per pixel.

---

## Training Objective: Predicting the Noise

```python
def getBatchLoss(model, x0, t, y):
    xt, noise = q_sample(x0, t)
    noise_pred = model(xt, t, y)
    return F.mse_loss(noise_pred, noise)
```

This is the training objective in its most compact form. First the code samples a noisy image

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon.
$$

Then it passes $$x_t$$, $$t$$, and $$y$$ into the model to get

$$
\epsilon_\theta(x_t,t,y).
$$

Finally it minimizes the mean-squared error between the true noise and the predicted noise:

$$
\|\epsilon - \epsilon_\theta(x_t,t,y)\|^2.
$$

Averaged over the data, timestep, and Gaussian noise, this becomes the familiar DDPM objective

$$
L = \mathbb{E}_{x_0,t,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t,t,y)\|^2\right].
$$

So the code line

```python
return F.mse_loss(noise_pred, noise)
```

is directly implementing the noise-prediction objective discussed earlier.

---

## The Training Loop

```python
for step, (x0, y) in enumerate(train_loader):
    x0 = x0.to(device)
    y = y.to(device)

    t = torch.randint(0, T, (x0.shape[0],), device=device).long()

    loss = getBatchLoss(model, x0, t, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

This loop makes explicit why training uses pairs $$(x_0, x_t)$$ rather than $$(x_{t-1}, x_t)$$.

The line

```python
t = torch.randint(0, T, (x0.shape[0],), device=device).long()
```

samples a random timestep for each image in the batch. Then `q_sample` uses the closed-form forward equation to jump directly from $$x_0$$ to $$x_t$$.

So for each training example, the model is really learning from triplets

$$
(x_0,\; x_t,\; t)
$$

together with the true noise $$ \epsilon $$.

Mathematically, this is efficient because the closed-form expression

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon
$$

lets us sample any timestep in one step. This is exactly the practical advantage discussed in the theory section.

---

## Reverse Sampling: Implementing $$p_\theta(x_{t-1}\mid x_t)$$

```python
@torch.no_grad()
def p_sample(model, x, t, y):
    betas_t = gatherAndExpand(betas, t)
    sqrt_one_minus_ab_t = gatherAndExpand(sqrt_one_minus_alpha_bars, t)
    sqrt_recip_alpha_t = gatherAndExpand(sqrt_reciprocal_alphas, t)

    eps_theta = model(x, t, y)

    model_mean = sqrt_recip_alpha_t * (
        x - (betas_t / sqrt_one_minus_ab_t) * eps_theta
    )

    posterior_var_t = gatherAndExpand(posterior_variance, t)

    nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
    noise = torch.randn_like(x)

    return model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise
```

This function implements one reverse diffusion step. In theory, the reverse transition is

$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}\!\left(\mu_\theta(x_t,t), \tilde{\beta}_t I\right).
$$

The network first predicts the noise:

$$
\epsilon_\theta(x_t,t,y).
$$

In code this is

```python
eps_theta = model(x, t, y)
```

Then the mean of the reverse Gaussian is computed as

```python
model_mean = sqrt_recip_alpha_t * (
    x - (betas_t / sqrt_one_minus_ab_t) * eps_theta
)
```

which is exactly

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t,y)
\right).
$$

The variance used for sampling is

```python
posterior_var_t = gatherAndExpand(posterior_variance, t)
```

which corresponds to

$$
\tilde{\beta}_t
=
\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t.
$$

Finally, the code samples

```python
return model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise
```

which is mathematically

$$
x_{t-1}
=
\mu_\theta(x_t,t)
+
\mathbf{1}_{t>0}\sqrt{\tilde{\beta}_t}\,z
$$

with

$$
z \sim \mathcal{N}(0,I).
$$

The mask $$ \mathbf{1}_{t>0} $$ ensures that on the final step no additional Gaussian noise is added.

This is exactly the reverse-sampling rule derived earlier: compute the mean of the reverse distribution from the predicted noise, then sample one plausible previous image from that Gaussian.

---

## Generating a Single Digit Step by Step

```python
@torch.no_grad()
def generate_digit_and_intermediates(model, digit):
    model.eval()

    x = torch.randn(1, 1, image_size, image_size, device=device)
    y = torch.full((1,), digit, device=device, dtype=torch.long)

    x_intermediates = []
    for i in reversed(range(T)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, y)
        x_intermediates.append(x.cpu().squeeze().numpy())

    x = (x.clamp(-1, 1) + 1) / 2
    return x, x_intermediates

```

This function is the full generative algorithm in code form.

It starts from pure Gaussian noise:

```python
x = torch.randn(1, 1, image_size, image_size, device=device)
```

which corresponds to

$$
x_T \sim \mathcal{N}(0,I).
$$

It then fixes a class label $$y$$, for example the digit 7, and repeatedly applies the reverse transition

$$
x_{t-1} \sim p_\theta(x_{t-1}\mid x_t, y)
$$

for

$$
t = T-1, T-2, \dots, 0.
$$

So the loop

```python
for i in reversed(range(T)):
    t = torch.full((1,), i, device=device, dtype=torch.long)
    x = p_sample(model, x, t, y)
```

is implementing the chain

$$
x_T \to x_{T-1} \to x_{T-2} \to \cdots \to x_0.
$$

The list `x_intermediates` stores the intermediate states so that we can visualize the gradual denoising process.

Finally, the line

```python
x = (x.clamp(-1, 1) + 1) / 2
```

rescales the generated image back from $$[-1,1]$$ to $$[0,1]$$ for display.

As an example, take the generation of the number 7:

```python
# example
x_generated, x_intermediates = generate_digit_and_intermediates(model, digit=7)

# now plot from the initial noise to the final generated image
t_to_plot = [0, 249, 499, 699, 899, 999]
fig, axs = plt.subplots(2, 3, figsize=(6, 4))
axs = axs.flatten()
for i in range(6):
    t = t_to_plot[i]
    axs[i].imshow(x_intermediates[t], cmap='gray')
    axs[i].set_title(f"t={T-1-t}")
    axs[i].axis('off')
plt.savefig("image_generation_intermediates.png", bbox_inches="tight")
plt.show()
```

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/posts/2026-03-011-diffusion_mnist/createSingleDigit.png" class="img-fluid rounded z-depth-1 w-50 d-block mx-auto" zoomable=true caption="Figure 3: Example of a generated 7, including a few intermediate steps." %}
    </div>
</div>

---

## Generating a Batch of Conditional Samples


```python
@torch.no_grad()
def generate_sample_of_digit(model, digit, n=16):
    model.eval()

    x = torch.randn(n, 1, image_size, image_size, device=device)
    y = torch.full((n,), digit, device=device, dtype=torch.long)

    for i in reversed(range(T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, y)

    x = (x.clamp(-1, 1) + 1) / 2
    return x
```

This is the same reverse diffusion process as above, but run on a batch of $$n$$ samples in parallel.

Mathematically, it starts from

$$
x_T^{(1)}, \dots, x_T^{(n)} \sim \mathcal{N}(0,I)
$$

and applies the reverse transition to each sample while conditioning them all on the same label $$y$$.

Because the label is fixed, this function samples from the model's learned class-conditional distribution for a given digit. In other words, for a chosen digit label $$y$$, it approximately draws samples from

$$
p_\theta(x_0 \mid y).
$$

This is why the model can generate a whole grid of different handwritten 7s, or 3s, or 9s: the stochastic reverse process produces different samples, while the class conditioning guides them toward the same digit identity.

We can then plot these as such:

```python 
def plot_sample(samples, digit):
    # plot the samples
    # it is a 4x4 grid
    fig, axs = plt.subplots(4, 4, figsize=(4, 4))
    axs = axs.flatten()
    for i in range(samples.size(0)):
        axs[i].imshow(samples[i].cpu().squeeze().numpy(), cmap='gray')
        axs[i].axis('off')
    fig.suptitle(f"Samples for digit {digit}")
    plt.show()

for digit in range(10):
    samples = generate_sample_of_digit(model, digit=digit, n=16)
    save_image(samples, f"mnist_diffusion_digit_{digit}.png", nrow=4)
    plot_sample(samples, digit)
    print(f"Saved samples for digit {digit}")
```

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/2026-03-011-diffusion_mnist/generated_0s.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/2026-03-011-diffusion_mnist/generated_1s.png" class="img-fluid rounded z-depth-1" %}
    </div>
    
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/2026-03-011-diffusion_mnist/generated_2s.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/2026-03-011-diffusion_mnist/generated_3s.png" class="img-fluid rounded z-depth-1" %}
    </div>

    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/2026-03-011-diffusion_mnist/generated_4s.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/2026-03-011-diffusion_mnist/generated_5s.png" class="img-fluid rounded z-depth-1" %}
    </div>

    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/2026-03-011-diffusion_mnist/generated_6s.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/2026-03-011-diffusion_mnist/generated_7s.png" class="img-fluid rounded z-depth-1" %}
    </div>

    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/2026-03-011-diffusion_mnist/generated_8s.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/2026-03-011-diffusion_mnist/generated_9s.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>


<div class="caption">
    Figure 4: Batch of generated digits. 
</div>

---

## Summary of the Theory-to-Code Mapping

The implementation closely mirrors the mathematical derivation:

1. The code precomputes the scalar schedule
$$
\beta_t,\; \alpha_t,\; \bar{\alpha}_t,\; \tilde{\beta}_t.
$$

2. The function `q_sample` implements
$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon.
$$

3. The U-Net learns the mapping
$$
(x_t,t,y) \mapsto \epsilon_\theta(x_t,t,y).
$$

4. The training loss implements
$$
\|\epsilon - \epsilon_\theta(x_t,t,y)\|^2.
$$

5. The function `p_sample` implements the reverse Gaussian
$$
p_\theta(x_{t-1}\mid x_t)
=
\mathcal{N}\!\left(
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t,y)
\right),
\tilde{\beta}_t I
\right).
$$

6. The generation loop repeatedly applies this reverse step, starting from
$$
x_T \sim \mathcal{N}(0,I),
$$
until a final image $$x_0$$ is produced.

This is the core diffusion pipeline in practice: sample noise, predict the noise at each timestep, and use that prediction to iteratively denoise toward a realistic image.
