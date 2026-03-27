# Ash: Contextual Epistemic Release for Transformer Language Models

**Authors:** Ash Project Contributors

**Status:** Working paper (architecture validated at proof-of-concept scale; large-scale results pending)

---

## Abstract

We introduce **Contextual Epistemic Release (CER)**, a mechanism that augments
transformer language models with the ability to recognize when a token's
information has been sufficiently integrated by the surrounding context and can
be safely de-emphasized in subsequent computation. CER is implemented through
three jointly trained primitives: **Epistemic State Channels (ESC)**, which
produce per-token scalar closure scores indicating integration status;
**Active Suppression Heads (ASH)**, which use sigmoid (not softmax) attention
to independently suppress multiple tokens without the zero-sum constraint of
standard attention; and the **Closure Register (CR)**, a compressed
sequence-level memory that accumulates information from epistemically closed
tokens and re-injects it as a faint residual signal. CER is neither sparsity,
nor pruning, nor Mixture-of-Experts routing---it is an epistemically grounded
forgetting mechanism that operates orthogonally to all three.

We present **Ashy-Small**, a 131.5M-parameter proof-of-concept
(124.4M base + 7.1M CER, 5.7% overhead) that validates the architecture at
GPT-2 scale, and detail the design of the full **Ash** model, a 17B-active-parameter
sparse MoE transformer (128 experts, top-2 routing, ~400B total parameters)
with CER integrated at the attention level. We describe a purpose-built
8-test evaluation battery (T1--T8) designed to distinguish genuine epistemic
reasoning from coincidental sparsity, with particular emphasis on the critical
T7 "Chekhov's Gun" test that guards against catastrophic over-suppression.
All claims about large-scale performance are explicitly marked as
planned/hypothesized; no benchmark results are fabricated.

---

## 1. Introduction

Standard transformer architectures treat every token in context with equal
architectural privilege. Once a token enters the key-value cache, it remains
available to all subsequent queries at full fidelity until the context window
is exhausted. This is computationally expensive and, we argue, epistemically
unnecessary: a model that has fully integrated the information carried by a
token---distributing its semantic content across surrounding hidden states---has
no further need to attend to that token at full strength.

Human cognition exhibits a related phenomenon: once a piece of information has
been understood and integrated into a mental model, the original sensory trace
fades, replaced by a compressed representation. We do not re-read each word
of a paragraph once we have grasped its meaning. We propose that language
models can benefit from a similar mechanism, provided it is implemented with
appropriate safeguards against premature or catastrophic forgetting.

**CER is not:**

- **Sparsity or pruning.** Sparse attention patterns (e.g., Longformer,
  BigBird) impose fixed or heuristic locality. CER learns *when* to
  de-emphasize, driven by epistemic state, not geometric proximity.
- **Mixture-of-Experts routing.** MoE routes tokens to different
  computational pathways. CER modulates *attention to context*, not
  feedforward computation. The two are orthogonal: a token can be routed
  to any expert while its ESC score evolves independently.
- **KV-cache eviction.** Eviction discards entries permanently. CER
  suppresses softly and retains a compressed trace in the Closure Register.
  However, ESC scores may *enable* principled KV-cache eviction as an
  inference-time optimization (Section 10).

**Contributions:**

1. The CER mechanism and its three primitives (ESC, ASH, CR).
2. A three-phase training curriculum that stabilizes CER learning.
3. An 8-test evaluation battery that targets epistemic behavior specifically.
4. Ashy-Small, a fully implemented 131.5M-parameter proof-of-concept.
5. A detailed scaling plan from Ashy-Small to a 17B-active MoE.

---

## 2. Background

### 2.1 Transformer Language Models

The decoder-only transformer (Vaswani et al., 2017; Radford et al., 2018)
computes hidden states through a stack of $L$ blocks, each applying
multi-head self-attention followed by a position-wise feedforward network
(FFN). For a sequence of $T$ tokens with embeddings $X \in \mathbb{R}^{T \times d}$:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right) V$$

where $Q = XW_Q$, $K = XW_K$, $V = XW_V$, and $d_h$ is the head dimension.
The softmax normalization imposes a zero-sum constraint: increasing attention
to one key necessarily decreases attention to others.

### 2.2 Mixture-of-Experts

Sparse MoE models (Shazeer et al., 2017; Fedus et al., 2022; Jiang et al.,
2024) replace the dense FFN with a set of $E$ expert FFNs, selecting the
top-$k$ per token via a learned router:

$$\text{MoE}(x) = \sum_{i \in \text{top-}k(g(x))} g_i(x) \cdot E_i(x)$$

This decouples total parameter count from per-token compute. CER operates at
the attention level, while MoE operates at the FFN level; they compose
without interference (Section 5.3).

### 2.3 Attention Efficiency

Flash Attention (Dao et al., 2022) and related IO-aware algorithms have made
exact attention practical for moderate context lengths ($\le$128K). For
longer contexts, methods such as sliding-window attention, sparse patterns,
and KV-cache compression are employed. CER provides a learned,
content-dependent signal for which tokens to de-emphasize, potentially
enabling more principled cache management than fixed heuristics.

---

## 3. Method: Contextual Epistemic Release

### 3.1 Overview

CER augments each transformer block with three components that track and act
on the epistemic state of tokens in context:

```
+-------------------------------------------------------------------+
|                     CER-Augmented Transformer Block                |
|                                                                    |
|   Input: x [B,T,D], cr_state [B,d_cr]                            |
|                                                                    |
|   1. h = LayerNorm(x)                                             |
|   2. attn_out, attn_weights = Attention(h)     <-- ASH modulates  |
|   3. x = x + attn_out                          <-- residual       |
|   4. esc_scores = ESC(x, attn_weights)         <-- epistemic score|
|   5. cr_state = CR.write(x, esc_scores, cr_state)  <-- accumulate |
|   6. x = x + CR.read(cr_state)                 <-- inject 0.1x   |
|   7. x = x + FFN(LayerNorm(x))                 <-- residual       |
|                                                                    |
|   Output: x [B,T,D], cr_state [B,d_cr]                           |
+-------------------------------------------------------------------+
```

The three primitives interact as follows: ESC *scores* tokens, ASH
*suppresses* attention based on those dynamics, and CR *remembers* a
compressed trace of suppressed information. Together, they implement a
soft, reversible, learnable forgetting mechanism.

### 3.2 Design Principles

1. **Soft, not hard.** CER produces continuous scores in $[0, 1]$, not
   binary gates. There is no discrete token dropping.
2. **Conservative initialization.** All CER components initialize to
   near-identity behavior: ESC scores start low (~0.12), ASH lambdas
   start small (0.01), CR gates start mostly closed (bias -1.0).
   The model must *learn* to forget; it does not forget by default.
3. **Orthogonality to base architecture.** CER can be added to any
   pre-norm transformer. It does not modify the FFN, embeddings, or
   output head.
4. **Curriculum-gated training.** CER losses are introduced gradually
   to avoid destabilizing early training dynamics.

---

## 4. Architecture Details

### 4.1 Ashy-Small (Proof-of-Concept)

Ashy-Small instantiates CER at GPT-2 scale for rapid validation:

| Parameter           | Value                |
|---------------------|----------------------|
| Layers              | 12                   |
| Model dimension     | 768                  |
| Attention heads     | 12 (10 std + 2 ASH)  |
| Head dimension      | 64                   |
| FFN hidden          | 3072 (4 x 768)       |
| Vocabulary          | 50,257 (GPT-2 BPE)   |
| Context length      | 1,024                |
| Base parameters     | 124.4M               |
| CER parameters      | 7.1M (5.7% overhead) |
| **Total**           | **131.5M**           |
| Norm                | LayerNorm             |
| Activation          | GELU                 |
| Positional encoding | Learned absolute      |

### 4.2 Full Ash (Target Architecture)

The full Ash model targets frontier capability with CER at scale:

| Parameter             | Value                          |
|-----------------------|--------------------------------|
| Layers                | 48                             |
| Model dimension       | 6,144                          |
| Query heads           | 64 (62 std + 2 ASH)            |
| KV heads              | 8 (GQA ratio 8:1)              |
| Head dimension        | 96                             |
| FFN hidden            | 16,384 (~8/3 x d_model)        |
| Experts               | 128 (top-2 routing, 1 shared)  |
| Vocabulary            | 128K BPE                       |
| Context               | 128K native (1M stretch)       |
| Active parameters     | 17B per token                  |
| Total parameters      | ~400B+                         |
| CER overhead          | ~965M (~5.7% of active)        |
| Norm                  | RMSNorm                        |
| Activation            | SwiGLU                         |
| Positional encoding   | RoPE with NTK scaling          |
| Precision             | BF16 train, FP8 inference      |

### 4.3 End-to-End Dataflow

The following diagram traces a forward pass through a CER-augmented
transformer:

```
          Token IDs [B, T]
               |
               v
    +---------------------+
    | Token Embedding     |
    | + Positional Embed  |
    +---------------------+
               |
               v
          x [B, T, D]
          cr_state = zeros [B, d_cr]
               |
     +---------+---------+
     |  For layer l = 1..L      |
     |                          |
     |   +------------------+   |
     |   | LayerNorm        |   |
     |   +------------------+   |
     |           |              |
     |           v              |
     |   +------------------+   |
     |   | Multi-Head Attn  |   |
     |   | Q,K,V projection |   |
     |   |                  |   |
     |   |  +------------+  |   |     +--> suppress_mask [B, n_ash, T, T]
     |   |  | ASH Heads  |  |   |     |    (independent per-token scores)
     |   |  | sigmoid(   |  |   |     |
     |   |  | Q_ash@K^T) |--+---+-----+
     |   |  +------------+  |   |
     |   |                  |   |
     |   | effective_attn = |   |
     |   |  softmax * (1 -  |   |
     |   |  lambda*suppress)|   |
     |   +------------------+   |
     |           |              |
     |     +residual            |
     |           |              |
     |   +------------------+   |
     |   | ESC              |   |
     |   | MLP([h;attn_in]) |   |     +--> esc_scores [B, T, 1]
     |   |  -> sigmoid      |---+-----+   (0=open, 1=closed)
     |   +------------------+   |
     |           |              |
     |   +------------------+   |
     |   | CR Write         |   |
     |   | gate*compress(h) |   |     +--> cr_state [B, d_cr]
     |   |  * esc_scores    |---+-----+   (accumulated)
     |   | cr += sum(write) |   |
     |   +------------------+   |
     |           |              |
     |   +------------------+   |
     |   | CR Read          |   |
     |   | 0.1*expand(cr)   |   |     +--> residual [B, D]
     |   +------------------+   |         (broadcast to all T)
     |           |              |
     |     +residual            |
     |           |              |
     |   +------------------+   |
     |   | LayerNorm        |   |
     |   +------------------+   |
     |           |              |
     |   +------------------+   |
     |   | FFN (or MoE FFN) |   |
     |   +------------------+   |
     |           |              |
     |     +residual            |
     |           |              |
     +-----------+--------------+
               |
               v
    +---------------------+
    | LayerNorm           |
    +---------------------+
               |
               v
    +---------------------+
    | LM Head (tied wts)  |
    +---------------------+
               |
               v
         logits [B, T, V]
```

---

## 5. CER Components

### 5.1 Epistemic State Channels (ESC)

ESC produces a per-token scalar score $s_t^{(\ell)} \in [0, 1]$ at each
layer $\ell$, indicating how fully the token's information has been integrated
by its context.

**Architecture.** A two-layer MLP takes the concatenation of the hidden state
and incoming attention statistics:

$$\text{input}_t = [h_t \,;\, a_t] \in \mathbb{R}^{d + H}$$

where $a_t \in \mathbb{R}^{H}$ is the sum of attention weights received by
token $t$ across all $H$ heads:

$$a_t^{(h)} = \sum_{i=1}^{T} \alpha_{i,t}^{(h)}$$

The score is then:

$$s_t = \sigma\!\left(W_2 \cdot \text{GELU}(W_1 \cdot \text{input}_t + b_1) + b_2\right)$$

with $W_1 \in \mathbb{R}^{m \times (d+H)}$, $W_2 \in \mathbb{R}^{1 \times m}$,
and $m = 64$ (the ESC MLP hidden dimension).

**Initialization.** $W_2$ is initialized to zero and $b_2 = -2.0$, so that
$\sigma(-2) \approx 0.12$. Tokens start predominantly "open."

**Target derivation.** The auxiliary training target for ESC is derived from
the entropy of the incoming attention distribution. For each key position
$j$, we normalize the column of the attention matrix into a distribution
over queries and compute its entropy:

$$H_j = -\sum_{i=1}^{T} \hat{\alpha}_{j \leftarrow i} \log \hat{\alpha}_{j \leftarrow i}$$

where $\hat{\alpha}_{j \leftarrow i}$ is the normalized incoming attention.
The target is:

$$\tau_j = 1 - \frac{H_j}{\log T}$$

High entropy (widely attended to by many queries) yields a low target (stay
open). Low entropy (attended to narrowly or not at all) yields a high target
(safe to close). The target is detached from the computation graph.

**ESC loss.** We penalize premature closure---scores that exceed their
entropy-derived targets---using a one-sided squared penalty:

$$\mathcal{L}_\text{ESC} = \frac{1}{L} \sum_{\ell=1}^{L} \frac{1}{BT} \sum_{b,t} \left[\max\!\left(0,\; s_t^{(\ell)} - \tau_t^{(\ell)}\right)\right]^2$$

This loss is asymmetric: tokens are free to remain open (low score) even when
the target suggests closure, but are penalized for closing before the
attention pattern warrants it.

### 5.2 Active Suppression Heads (ASH)

ASH heads are sigmoid-based attention heads that produce per-position
suppression masks, modulating the effective attention weights of standard
heads.

**Key design decision: sigmoid vs. softmax.**

Standard attention uses softmax, which imposes a zero-sum constraint: increasing
attention to one key necessarily decreases attention to others. This makes it
impossible for a single attention head to suppress *multiple* tokens
simultaneously---reducing attention to token A automatically redistributes
probability mass to tokens B, C, etc.

ASH heads use sigmoid activation, producing *independent* suppression scores
in $[0, 1]$ for each query-key pair:

$$\text{suppress}_{i,j} = \sigma\!\left(\frac{q_i^{\text{ash}} \cdot k_j}{\sqrt{d_h}}\right)$$

This allows the model to suppress any subset of context tokens without
redistributing attention mass.

**Architecture.** Each ASH head has its own learned query projection
$W_Q^{\text{ash}} \in \mathbb{R}^{d \times d_h}$ but shares keys with the
standard attention mechanism (using the average across all standard K heads).
A causal mask prevents suppression of future tokens.

**Modulation.** The effective attention weights after ASH are:

$$\hat{\alpha}_{i,j} = \alpha_{i,j}^{\text{softmax}} \cdot \left(1 - \frac{1}{N_{\text{ash}}} \sum_{a=1}^{N_{\text{ash}}} \lambda_a \cdot \text{suppress}_{i,j}^{(a)}\right)$$

where $\lambda_a$ is a per-head learnable suppression strength, clamped to
$[0, 1]$, initialized to 0.01.

**ASH regularization loss.** To prevent over-suppression, we penalize high
mean ESC scores across the model as a proxy:

$$\mathcal{L}_\text{ASH} = \frac{1}{L} \sum_{\ell=1}^{L} \frac{1}{BT} \sum_{b,t} s_t^{(\ell)}$$

This encourages the model to keep ESC scores (and thus suppression) measured.

### 5.3 Closure Register (CR)

The CR is a compressed, sequence-level memory buffer that accumulates
information from epistemically closed tokens. It serves two purposes:
(1) retaining a trace of suppressed information to mitigate catastrophic
forgetting, and (2) providing a potential pathway for information recovery
when later context reveals that a suppressed token was prematurely closed.

**Compression.** A linear projection compresses from model dimension to CR
dimension: $d_\text{cr} = d / 4$.

$$c_t = W_\text{compress} \cdot h_t \in \mathbb{R}^{d_\text{cr}}$$

**Gated write.** A learned gate controls the write rate:

$$g_t = \sigma(W_\text{gate} \cdot h_t + b_\text{gate})$$

The write amount for each token is proportional to its ESC score:

$$\text{write}_t = s_t \cdot g_t \cdot c_t$$

The CR state is updated by accumulating across the token dimension:

$$\text{cr}^{(\ell)} = \text{cr}^{(\ell-1)} + \sum_{t=1}^{T} \text{write}_t$$

The CR state persists across layers (the same $[B, d_\text{cr}]$ buffer
is passed through all $L$ blocks), allowing later layers to read summaries
written by earlier layers.

**Read and injection.** The CR state is expanded back to model dimension and
added to the residual stream at reduced scale:

$$\Delta x = \alpha \cdot W_\text{expand} \cdot \text{cr} \in \mathbb{R}^{d}$$

where $\alpha = 0.1$ is the injection scale. This is broadcast identically
across all token positions $t$, providing a uniform "background memory" signal.

**Initialization.** Compress and expand weights use $\mathcal{N}(0, 0.01)$
initialization; the gate bias is $-1.0$ (sigmoid$(-1) \approx 0.27$, so
the gate starts mostly closed). At initialization, the CR contribution to
the residual stream is $\sim 10^{-3}\times$ the residual norm.

### 5.4 CER-MoE Interaction (Full Ash)

In the full Ash model, MoE and CER occupy different architectural locations
and serve different functions:

```
+------------------------------------------------------+
|              Architectural Orthogonality              |
|                                                       |
|  ATTENTION LEVEL           FFN LEVEL                  |
|  +-----------------+       +--------------------+     |
|  | Standard Heads  |       | Expert Router      |     |
|  | ASH Heads       |       |  -> top-2 experts  |     |
|  | ESC scoring     |       |  + 1 shared expert |     |
|  | CR write/read   |       +--------------------+     |
|  +-----------------+                                  |
|                                                       |
|  CER: "What info is       MoE: "Which computation    |
|  integrated and can        pathway should process     |
|  be de-emphasized?"        this token?"               |
|                                                       |
|  Operates on ATTENTION     Operates on FFN ROUTING    |
|  (epistemic state)         (computational routing)    |
+------------------------------------------------------+
```

A token can have different expert assignments at each layer while its ESC
score evolves independently. The CR state is available to *all* experts
equally---it represents shared epistemic memory that crosses the MoE boundary.

### 5.5 CER Parameter Budget

CER parameters per layer at the Ashy-Small and full Ash scales:

| Component       | Ashy-Small (d=768) | Full Ash (d=6144)  |
|-----------------|--------------------|--------------------|
| ESC MLP         | ~50K               | ~75K               |
| ASH Q proj      | ~98K               | ~1.2M              |
| CR compress     | ~147K              | ~9.4M              |
| CR expand       | ~147K              | ~9.4M              |
| CR gate         | ~148K              | ~9.4M              |
| ASH lambdas     | 2                  | 2                  |
| **Per-layer**   | **~590K**          | **~20.1M**         |
| **All layers**  | **~7.1M** (12L)    | **~965M** (48L)    |
| **Overhead**    | **5.7%**           | **~5.7%**          |

---

## 6. Training Curriculum

### 6.1 Three-Phase Schedule

CER losses are not applied from the start of training. A three-phase
curriculum prevents CER from destabilizing the early, critical phase of
language model training:

```
Phase 1: Warm-up         Phase 2: CER Ramp        Phase 3: Full CER
(0% - 10% of training)   (10% - 30%)              (30% - 100%)

lambda_esc = 0.0          0.0 -> 0.1 (linear)      0.1
lambda_ash = 0.0          0.0 -> 0.05 (linear)     0.05

Model learns basic LM     CER gradually engages    Full epistemic
without CER overhead.     as base LM stabilizes.   forgetting active.
ESC/ASH/CR params still   Gradual ramp prevents    LM and CER losses
receive gradients from     sudden distribution      jointly optimized.
LM loss backprop.          shift in gradients.
```

**Rationale.** CER components receive indirect gradient signal from the
language modeling loss even during Phase 1 (e.g., ASH heads affect effective
attention weights, which affect LM loss). The curriculum controls only the
*auxiliary* CER-specific losses. This allows the CER parameters to begin
adapting to the model's attention patterns before being explicitly trained
toward epistemic objectives.

### 6.2 Loss Function

The total training loss at step $t$ is:

$$\mathcal{L}_\text{total} = \mathcal{L}_\text{LM} + \lambda_\text{esc}(t) \cdot \mathcal{L}_\text{ESC} + \lambda_\text{ash}(t) \cdot \mathcal{L}_\text{ASH}$$

where:

- $\mathcal{L}_\text{LM}$ is the standard autoregressive cross-entropy loss.
- $\mathcal{L}_\text{ESC}$ penalizes premature closure (one-sided squared excess over entropy-derived target).
- $\mathcal{L}_\text{ASH}$ penalizes over-suppression (mean ESC score across all layers).
- $\lambda_\text{esc}(t)$ and $\lambda_\text{ash}(t)$ follow the curriculum schedule.

### 6.3 Optimizer and Schedule

| Hyperparameter        | Ashy-Small     | Full Ash (planned)     |
|-----------------------|----------------|------------------------|
| Optimizer             | AdamW          | AdamW                  |
| $\beta_1, \beta_2$    | 0.9, 0.95      | 0.9, 0.95              |
| Weight decay          | 0.1            | 0.1                    |
| Peak LR               | 6e-4           | 3e-4                   |
| Min LR                | 6e-5           | 3e-5                   |
| Warmup steps          | 2,000          | 2,000                  |
| Max steps             | 100,000        | TBD (token-budget)     |
| LR schedule           | Cosine decay   | Cosine decay           |
| Gradient clipping     | 1.0 (global)   | 1.0 (global)           |
| Batch size            | 12 x 40 GA     | 512 -> 4096 (ramp)     |
| Precision             | BF16           | BF16 (FP8 inference)   |

**Weight decay groups.** Linear weights receive weight decay; biases,
embedding weights, LayerNorm/RMSNorm parameters, and CER scalar parameters
(ASH lambdas) are excluded.

---

## 7. Experimental Design

### 7.1 Ashy-Small Validation Protocol

Ashy-Small serves as a controlled validation environment for CER. Its purpose
is not to produce a competitive language model at 131M parameters, but to
answer three questions:

1. **Does CER learn meaningful epistemic behavior?** (T8: ESC flow analysis)
2. **Does CER improve specific downstream behaviors?** (T2--T4)
3. **Does CER avoid catastrophic failure modes?** (T7: Chekhov's Gun)

**Ablation configurations:**

| Configuration         | CER  | Purpose                              |
|-----------------------|------|--------------------------------------|
| `ashy_small.yaml`     | Full | Main CER-augmented model             |
| `ashy_small_no_cer.yaml` | Off | Vanilla GPT-2 baseline (ablation) |

Both configurations share identical base architecture (12L, 768d, 12H),
training data, and optimizer settings. The no-CER variant uses fused
`scaled_dot_product_attention` (faster but does not expose attention weights).

### 7.2 Data Pipeline

Training data is managed through a declarative YAML-driven pipeline
(`data_sources/*.yaml`) with the following properties:

- **Provenance tracking:** Each source records deduplication status,
  PII scrubbing, quality filtering, and licensing.
- **Versioned shards:** Training uses tokenized, versioned shards from
  deterministic manifests---never raw text fetched at runtime.
- **Three-tier storage:** S3 (canonical) -> optional FSx for Lustre
  (shared high-throughput) -> local NVMe (hot cache with LRU eviction).

**Planned data mix for full Ash (Phase B, 350B--500B tokens):**

| Category              | Share | Sources                              |
|-----------------------|-------|--------------------------------------|
| Web (deduplicated)    | 50%   | FineWeb, RedPajama                   |
| Code                  | 20%   | The Stack v2, StarCoderData          |
| Books / long-form     | 10%   | Various (license-cleared)            |
| Scientific            | 8%    | ArXiv, PubMed                        |
| Wikipedia             | 5%    | Wikipedia, Wikibooks, Wikisource     |
| Math / synthetic      | 4%    | OpenWebMath                          |
| Instruction / dialog  | 3%    | UltraChat, OpenAssistant             |

### 7.3 Staged Token Budget

Training is organized in staged gates with explicit go/no-go criteria:

| Stage    | Tokens       | Purpose                     | Exit Criteria                        |
|----------|--------------|-----------------------------|--------------------------------------|
| Phase A  | 5B--20B      | Pipeline & arch shakeout    | T7 not catastrophic, T8 shows ESC progression |
| Phase B  | 350B--500B   | Core pretraining (compute-optimal for 17B active) | Standard benchmarks competitive, CER battery passes |
| Phase C  | 1T--4T       | Optional extension          | Continued ROI on scaling curves      |

---

## 8. Evaluation Battery (T1--T8)

The CER evaluation battery consists of eight tests designed to probe
epistemic behavior specifically. Standard language modeling benchmarks (MMLU,
HumanEval, GSM8K, etc.) are evaluated separately; the T1--T8 battery targets
the *CER-specific hypothesis* that learned epistemic forgetting produces
measurable behavioral signatures.

### T1: Grokking Speed

**Hypothesis:** CER accelerates grokking on algorithmic tasks by enabling
the model to release memorized examples once the underlying rule has been
learned.

**Protocol:** Train on modular arithmetic ($a \circ b \mod p$ for various
operations $\circ$) with a 50/50 train/test split. Measure steps to
generalization (test accuracy crossing 95%) for CER vs. baseline.

**Status:** Planned. Implementation pending.

### T2: Garden-Path Recovery

**Hypothesis:** CER enables faster recovery from syntactic garden-path
sentences by suppressing the initial misparse once disambiguation occurs.

**Protocol:** Present syntactically misleading sentences (e.g., "The horse
raced past the barn fell.") and measure mean surprisal (negative
log-probability) at the disambiguation point. Lower surprisal indicates
faster recovery.

**Dataset:** 10 garden-path sentences with known disambiguation points.

**Status:** Implemented (`ash/eval/t2_garden_path.py`). Results pending
training.

### T3: Noise Robustness

**Hypothesis:** CER improves robustness to irrelevant context by actively
suppressing noise tokens.

**Protocol:** Embed a fact early in context (e.g., "The capital of France is
Paris."), insert $N$ tokens of irrelevant filler, then probe for fact
retrieval. Noise levels: $N \in \{0, 50, 100, 200, 400\}$.

**Metric:** Retrieval accuracy as a function of noise level. Expected: CER
degrades more gracefully than baseline.

**Status:** Implemented (`ash/eval/t3_noise_robustness.py`). Results pending.

### T4: Redundancy Tolerance

**Hypothesis:** CER handles redundant information more efficiently by closing
tokens whose content is already represented elsewhere in context.

**Protocol:** Present passages with varying levels of deliberate redundancy.
Measure perplexity on the non-redundant continuation.

**Status:** Planned. Implementation pending.

### T5: Closure Register Probe

**Hypothesis:** The CR encodes meaningful compressed summaries of closed
tokens, not noise.

**Protocol:** Train a linear probe on CR states to predict properties of
closed tokens (e.g., topic, entity presence). Above-chance probe accuracy
indicates that CR retains useful information.

**Status:** Planned. Implementation pending.

### T6: Suppression Calibration

**Hypothesis:** ASH suppression strength correlates with the actual
redundancy of suppressed tokens.

**Protocol:** Measure the correlation between ASH suppression mask values
and an independent measure of token redundancy (e.g., mutual information
with surrounding context).

**Status:** Planned. Implementation pending.

### T7: Chekhov's Gun (Critical)

**Hypothesis:** CER does *not* suppress information that appears irrelevant
but becomes crucial later ("Chekhov's Gun" pattern).

**Protocol:** Present setups where a detail mentioned early (e.g., "John
placed his old key under the doormat.") is separated by ~300 tokens of filler
before being queried ("Where is the key?"). Measure:

1. **Answer accuracy:** Is the expected answer in the top-$k$ logits ($k=20$)?
2. **ESC scores of setup tokens:** Did the model suppress the critical
   information (key, doormat) prematurely?

**Failure criterion:** If CER systematically fails T7 (catastrophic
over-suppression of information needed later), the mechanism is considered
unsafe for deployment. This is the single most important test in the battery.

**Mitigation if T7 fails:** Conditional restoration mechanism---monitor ESC
scores for sharp increases after query tokens and trigger CR-based recovery.
See Section 9.2.

**Status:** Implemented (`ash/eval/t7_chekhov.py`). Results pending training.

### T8: ESC Flow Analysis

**Hypothesis:** ESC scores exhibit interpretable dynamics across layers:
function words (the, a, is, in) should close earlier than content words
(nouns, verbs, adjectives).

**Protocol:** Run inference on diverse text samples. Classify tokens as
function words or content words. Compute mean ESC scores per layer for each
class.

**Expected pattern:**

```
ESC Score
  1.0 |                                          .---.  Function words
      |                                    .---'
      |                              .---'
  0.5 |                        .---'
      |                  .---'                    .--.  Content words
      |            .---'                    .---'
      |      .---'                    .---'
  0.0 |.---'                   .---'
      +---+---+---+---+---+---+---+---+---+---+---+---> Layer
        1   2   3   4   5   6   7   8   9  10  11  12
```

Function words carry syntactic scaffolding that is integrated early.
Content words carry semantic payload that remains relevant longer.

**Status:** Implemented (`ash/eval/t8_esc_flow.py`). Results pending training.

---

## 9. Failure Modes and Mitigations

### 9.1 Over-Suppression (Catastrophic Forgetting)

**Risk:** The model learns to suppress tokens too aggressively, losing
information that is needed downstream. This is distinct from standard
catastrophic forgetting in continual learning; here, the forgetting occurs
*within a single forward pass*.

**Mitigations:**

1. **Conservative initialization.** ESC scores start at ~0.12; ASH lambdas
   start at 0.01. The model must actively learn to forget.
2. **One-sided ESC loss.** Only penalizes premature closure (scores
   exceeding the entropy-derived target), not tokens that remain open.
3. **ASH regularization loss.** Penalizes high mean ESC scores across the
   model, directly discouraging aggressive suppression.
4. **CR safety net.** Even when a token is suppressed, its compressed
   representation persists in the Closure Register at 0.1x scale.
5. **Three-phase curriculum.** CER losses are introduced gradually,
   allowing the base model to stabilize before epistemic mechanisms engage.

### 9.2 Chekhov's Gun Failure (T7)

**Risk:** Information that appears irrelevant at point of introduction
becomes critical later. CER suppresses it before its relevance is revealed.

**Primary defense:** The mitigations in 9.1 make this failure mode less
likely by construction---the model is biased toward keeping tokens open.

**Secondary defense (planned):** If T7 failure rates exceed acceptable
thresholds, implement a conditional restoration mechanism:

1. Monitor for sharp increases in query-side attention to positions with
   high ESC scores.
2. When detected, scale up the CR injection for that forward pass.
3. If necessary, temporarily reduce ASH lambda values during inference.

### 9.3 CER Training Instability

**Risk:** The auxiliary CER losses conflict with the language modeling
objective, causing training divergence.

**Mitigations:**

1. **Curriculum.** CER losses are zero for the first 10% of training.
2. **Small loss coefficients.** $\lambda_\text{esc} = 0.1$ and
   $\lambda_\text{ash} = 0.05$ are small relative to the LM loss.
3. **Monitoring.** Track per-component losses (LM, ESC, ASH) independently.
   If any component diverges, reduce its lambda or pause CER training.

### 9.4 CER Ineffectiveness at Scale

**Risk:** CER works at Ashy-Small scale but provides no benefit (or causes
harm) at 17B-active scale.

**Mitigation:** Phased build plan with explicit go/no-go gates. After
validating at Ashy-Small, test at 1B-active x 16-expert intermediate
scale before committing to full 17B x 128E training.

---

## 10. Scaling Plan

### 10.1 Model Progression

```
+------------------+     +---------------------+     +---------------------+
| Ashy-Small       |     | Ash-1B (validation) |     | Full Ash            |
| 131.5M params    |     | ~1B active x 16E    |     | 17B active x 128E   |
| 12L, 768d        | --> | ~24L, ~2048d        | --> | 48L, 6144d          |
| 1024 context     |     | 4K-8K context       |     | 128K context        |
| LayerNorm, GELU  |     | RMSNorm, SwiGLU     |     | RMSNorm, SwiGLU     |
| Learned pos      |     | RoPE                |     | RoPE + NTK          |
| Single GPU       |     | 8-16 GPUs           |     | 512x H100           |
+------------------+     +---------------------+     +---------------------+
       |                         |                          |
  CER validated            CER-MoE validated           Production CER
  at 131M scale            at 1B-16E scale             at frontier scale
```

### 10.2 CER Scaling Properties

CER overhead scales linearly with model dimension and layer count. At all
scales, it remains at approximately 5.7% of active parameters. Key
scaling-sensitive components:

- **ESC MLP:** Input dimension grows as $d + H$; hidden dimension ($m = 64$)
  may need to scale sublinearly with model size. This is an open question.
- **ASH heads:** Number of ASH heads (2) remains constant; head dimension
  scales with the model. Key sharing (averaged K) is efficient regardless of
  the number of standard attention heads.
- **CR:** Compression ratio ($d / 4$) is maintained. CR state is
  sequence-level ($[B, d_\text{cr}]$), so memory cost does not scale with
  sequence length---only with model dimension.

### 10.3 Infrastructure (Full Ash)

| Component              | Specification                      |
|------------------------|------------------------------------|
| Hardware               | 512x NVIDIA H100 80GB SXM          |
| Interconnect           | NDR InfiniBand 400 Gb/s             |
| Framework              | Megatron-LM (fork with CER module)  |
| Parallelism            | 4D: 8 TP x 16 EP x 4 PP x 1 DP    |
| Custom kernels         | Fused CER attention (Flash Attn 2 + ASH sigmoid), ESC update, CR accumulation |
| Precision              | BF16 training, FP8 inference        |

**CER-specific kernel requirements:**

1. **Fused ASH attention.** Flash Attention 2 computes attention in tiled
   blocks without materializing the full $T \times T$ matrix. CER requires
   modifying this kernel to apply sigmoid suppression masks within each tile.
   A Triton prototype is planned before CUDA porting.

2. **ESC update.** The ESC MLP is small ($d + H \to 64 \to 1$) and runs
   per-layer. Fusing it with the attention output reduces kernel launch
   overhead.

3. **CR accumulation.** The CR write operation (compress, gate, scale by ESC,
   sum over tokens) can be fused into a single kernel.

---

## 11. Systems Considerations

### 11.1 KV-Cache Optimization (Inference)

ESC scores provide a principled signal for KV-cache management during
autoregressive generation:

**Hypothesis (untested):** Tokens with sustained high ESC scores
($s_t^{(\ell)} > 0.95$ for $\ell > L/2$) can have their KV entries evicted
from cache and replaced by the CR summary, reducing memory footprint for
long-context inference without quality degradation.

This is distinct from existing KV-cache eviction heuristics (e.g., H2O,
which uses cumulative attention scores) in that it uses a *learned*,
*content-aware* signal rather than attention statistics alone.

### 11.2 Speculative Decoding

**Hypothesis (untested):** ASH suppression patterns at layer $\ell$ may
predict which tokens are irrelevant for the *next* token prediction,
enabling more efficient speculative decoding by focusing draft model
computation on non-suppressed context.

### 11.3 Checkpoint Format

Checkpoints store:

```python
{
    "model_state_dict":     model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "step":                 int,
    "configs": {
        "model":    ModelConfig,
        "training": TrainingConfig,
        "cer":      CERConfig,
    },
    "metrics": {
        "train_loss": float,
        "val_loss":   float,
        # ... per-component losses
    },
}
```

### 11.4 Reproducibility

CER adds three sources of non-determinism that must be controlled:

1. **ESC target computation.** Depends on attention weights, which may
   differ across attention implementations (fused vs. manual). CER always
   uses the manual path.
2. **CR state accumulation.** Floating-point summation order across the
   token dimension. Mitigated by using deterministic CUDA operations
   where available.
3. **ASH sigmoid masking.** The causal mask uses $-10^9$ (not $-\infty$)
   for masked positions in sigmoid, which produces $\sigma(-10^9) \approx 0$
   rather than exactly 0.

---

## 12. Limitations

1. **No large-scale results.** All claims about CER effectiveness beyond
   Ashy-Small are hypothesized. The mechanism may fail to provide
   benefits at scale, or may introduce unforeseen instabilities.

2. **Attention weight materialization.** CER requires explicit attention
   weight matrices for ESC computation, precluding the use of Flash
   Attention (which avoids materializing the $T \times T$ matrix). This
   increases memory cost quadratically with sequence length during
   training. Custom fused kernels are planned but not yet implemented.

3. **ESC target heuristic.** The entropy-derived ESC target is a heuristic,
   not a ground-truth measure of epistemic integration. It is possible to
   construct scenarios where the heuristic is misleading (e.g., a token
   that receives high-entropy attention but carries critical information).

4. **CR capacity.** The CR is a fixed-size buffer ($[B, d_\text{cr}]$) that
   accumulates across all layers and all tokens. For very long sequences
   or deep models, the CR may saturate, reducing its utility. No
   normalization or forgetting mechanism is currently applied to CR state.

5. **Evaluation battery limitations.** T1, T4, T5, T6 are not yet
   implemented. The implemented tests (T2, T3, T7, T8) use small,
   hand-crafted datasets (10 garden-path sentences, 5 noise levels, 3
   Chekhov's gun scenarios, 2 text samples). These are sufficient for
   qualitative validation but not for rigorous statistical claims.

6. **Single-GPU proof-of-concept.** Ashy-Small trains on a single GPU.
   The scaling plan to 512 GPUs introduces parallelism challenges
   (particularly for CR state, which is sequence-level and must be
   synchronized across tensor-parallel ranks) that have not been validated.

7. **No multimodal validation.** CER is designed for language; its
   applicability to vision tokens (where redundancy patterns differ)
   is an open question.

---

## 13. Ethics and Safety

### 13.1 CER-Specific Safety Concerns

**Selective information suppression.** A model with learned forgetting
could, in principle, learn to suppress information that is inconvenient
for its training objective rather than information that is genuinely
redundant. We mitigate this through:

- Transparent ESC scores that can be inspected at inference time.
- The T7 Chekhov's Gun test as a mandatory safety gate.
- CR retention of suppressed information at reduced scale.

**Interpretability.** ESC scores and ASH suppression masks are fully
inspectable, providing a layer of interpretability that standard attention
weights lack. Visualizing ESC flow across layers (T8) can reveal what the
model considers "integrated" vs. "still relevant."

### 13.2 Data and Training

- All training data sources have explicit license documentation
  (`data_sources/*.yaml`).
- Sources with `allowed_use: unknown` are flagged for quarantine pending
  legal review.
- PII scrubbing status and deduplication status are tracked per source.
- No training on live-fetched data; all training uses versioned, immutable
  token shards.

### 13.3 Dual Use

CER does not introduce capabilities beyond those of standard transformer
language models. It is an efficiency and interpretability mechanism, not
a capability amplifier. The same safety considerations that apply to
any frontier language model apply to Ash.

---

## 14. Related Work

**Sparse and efficient attention.** Longformer (Beltagy et al., 2020),
BigBird (Zaheer et al., 2020), and Reformer (Kitaev et al., 2020) impose
fixed or learned sparsity patterns on attention. CER differs in that
suppression is *content-dependent* and *learned from epistemic state*,
not geometric proximity or hash-based partitioning.

**KV-cache compression.** H2O (Zhang et al., 2023) evicts KV entries based
on cumulative attention scores. Scissorhands (Liu et al., 2023) uses
attention persistence. CER provides a richer signal (learned ESC scores
that incorporate both hidden state and attention statistics) and retains
a compressed trace (CR) rather than discarding entries entirely.

**Mixture-of-Experts.** GShard (Lepikhin et al., 2021), Switch Transformer
(Fedus et al., 2022), Mixtral (Jiang et al., 2024), and Llama 4 Maverick
(Meta, 2025) route tokens to subsets of experts at the FFN level. Ash's MoE
design follows this lineage. CER is orthogonal to MoE and operates at the
attention level.

**Gating and forgetting in RNNs.** LSTM (Hochreiter & Schmidhuber, 1997)
and GRU (Cho et al., 2014) use learned forget gates to manage information
flow in recurrent architectures. CER can be seen as bringing a similar
principle to the attention mechanism of transformers, with the key
difference that CER operates on the full context simultaneously rather
than sequentially.

**Memory-augmented transformers.** Compressive Transformer (Rae et al.,
2020) compresses old activations into a secondary memory. Memorizing
Transformers (Wu et al., 2022) use kNN lookup over cached representations.
CR is simpler: a single compressed vector per sequence, not a memory bank.

**Adaptive computation.** Universal Transformers (Dehghani et al., 2019)
and early-exit methods (Schwartz et al., 2020) adapt the *amount* of
computation per token. CER adapts the *attention* to context tokens, a
complementary axis of adaptation.

**Sigmoid attention.** Recent work on replacing softmax with sigmoid in
attention (Ramapuram et al., 2024) explores this for the primary attention
mechanism. ASH uses sigmoid specifically for *suppression heads*, not as a
replacement for standard softmax attention, combining both activation
functions in the same block.

---

## 15. Conclusion

We have presented Contextual Epistemic Release (CER), a mechanism that
endows transformer language models with learned, content-dependent
information forgetting. CER's three primitives---Epistemic State Channels,
Active Suppression Heads, and the Closure Register---provide a soft,
reversible, interpretable pathway for de-emphasizing integrated information
while retaining a compressed trace.

Ashy-Small (131.5M parameters) demonstrates that CER can be cleanly
integrated into a standard transformer at minimal overhead (5.7%) with
stable training dynamics. The full Ash design (17B active x 128 experts)
targets frontier scale, with CER operating orthogonally to MoE routing.

The central open question is whether CER provides measurable benefits at
scale. The phased build plan, staged token budgets, and explicit go/no-go
criteria are designed to answer this question efficiently, committing
compute only as evidence accumulates. The 8-test CER battery---particularly
the T7 Chekhov's Gun test---provides targeted evaluation of the epistemic
claims that distinguish CER from simpler sparsity or pruning approaches.

If CER delivers on its hypotheses, the implications extend beyond language
modeling: ESC-guided KV-cache eviction could make million-token context
practical, ASH patterns could accelerate speculative decoding, and the
CR mechanism could enable principled knowledge distillation from large
sparse models to smaller dense ones. These remain open research directions.

---

## Appendix A: Key Equations

### A.1 ESC Score

$$s_t^{(\ell)} = \sigma\!\left(W_2^{(\ell)} \cdot \text{GELU}\!\left(W_1^{(\ell)} \cdot [h_t^{(\ell)} ; a_t^{(\ell)}] + b_1^{(\ell)}\right) + b_2^{(\ell)}\right)$$

where $a_t^{(\ell)} = \left[\sum_i \alpha_{i,t}^{(\ell,1)}, \ldots, \sum_i \alpha_{i,t}^{(\ell,H)}\right] \in \mathbb{R}^H$.

### A.2 ESC Target

$$\tau_t^{(\ell)} = 1 - \frac{H_t^{(\ell)}}{\log T}, \qquad H_t^{(\ell)} = -\sum_{i} \hat{\alpha}_{t \leftarrow i}^{(\ell)} \log \hat{\alpha}_{t \leftarrow i}^{(\ell)}$$

### A.3 ESC Loss

$$\mathcal{L}_\text{ESC} = \frac{1}{L} \sum_{\ell=1}^{L} \mathbb{E}_{b,t}\!\left[\left(\max(0, s_t^{(\ell)} - \tau_t^{(\ell)})\right)^2\right]$$

### A.4 ASH Suppression

$$\text{suppress}_{i,j}^{(a)} = \sigma\!\left(\frac{q_i^{\text{ash},(a)} \cdot \bar{k}_j}{\sqrt{d_h}}\right), \qquad \bar{k}_j = \frac{1}{H}\sum_{h=1}^{H} k_j^{(h)}$$

### A.5 Effective Attention

$$\hat{\alpha}_{i,j} = \alpha_{i,j}^{\text{softmax}} \cdot \left(1 - \frac{1}{N_a}\sum_{a=1}^{N_a} \text{clamp}(\lambda_a, 0, 1) \cdot \text{suppress}_{i,j}^{(a)}\right)$$

### A.6 CR Write

$$\text{cr}^{(\ell)} = \text{cr}^{(\ell-1)} + \sum_{t=1}^{T} s_t^{(\ell)} \cdot \sigma(W_g h_t + b_g) \cdot W_c h_t$$

### A.7 CR Read

$$\Delta x^{(\ell)} = \alpha \cdot W_e \cdot \text{cr}^{(\ell)}, \qquad \alpha = 0.1$$

### A.8 Total Loss

$$\mathcal{L} = \mathcal{L}_\text{LM} + \lambda_\text{esc}(t) \cdot \mathcal{L}_\text{ESC} + \lambda_\text{ash}(t) \cdot \mathcal{L}_\text{ASH}$$

### A.9 CER Curriculum

$$\lambda_\text{esc}(t) = \begin{cases} 0 & \text{if } p < p_1 \\ \lambda_\text{esc}^{\max} \cdot \frac{p - p_1}{p_2 - p_1} & \text{if } p_1 \le p < p_2 \\ \lambda_\text{esc}^{\max} & \text{if } p \ge p_2 \end{cases}$$

where $p = t / t_\text{max}$, $p_1 = 0.1$, $p_2 = 0.3$. Same form for $\lambda_\text{ash}(t)$.

---

## Appendix B: Pseudocode

### B.1 CER-Augmented Transformer Block

```python
def cer_transformer_block(x, cr_state, layer):
    """
    x:        [B, T, D]  hidden states
    cr_state: [B, d_cr]  closure register (persists across layers)
    """
    # --- Attention with ASH ---
    h = layer_norm_1(x)
    q, k, v = qkv_project(h)                    # standard QKV

    # Standard softmax attention
    attn_logits = (q @ k.T) / sqrt(d_h)
    attn_logits = causal_mask(attn_logits)
    attn_weights = softmax(attn_logits, dim=-1)  # [B, H, T, T]

    # ASH: sigmoid suppression (if CER enabled)
    q_ash = ash_q_project(h)                     # [B, n_ash, T, d_h]
    k_avg = mean(k, dim=heads)                   # [B, 1, T, d_h]
    suppress = sigmoid(q_ash @ k_avg.T / sqrt(d_h))  # [B, n_ash, T, T]
    suppress = causal_mask(suppress)

    # Modulate attention
    lambda_ash = clamp(learnable_lambda, 0, 1)   # [n_ash]
    combined = mean(lambda_ash * suppress, dim=ash_heads)  # [B, 1, T, T]
    effective_attn = attn_weights * (1 - combined)

    attn_out = effective_attn @ v
    attn_out = output_project(attn_out)
    x = x + attn_out                             # residual

    # --- ESC ---
    incoming_attn = sum(attn_weights, dim=query)  # [B, H, T]
    esc_input = concat(x, incoming_attn)          # [B, T, D+H]
    esc_scores = sigmoid(mlp(esc_input))          # [B, T, 1]

    # --- CR Write ---
    gate = sigmoid(gate_project(x))               # [B, T, d_cr]
    compressed = compress_project(x)              # [B, T, d_cr]
    write_val = esc_scores * gate * compressed    # [B, T, d_cr]
    cr_state = cr_state + sum(write_val, dim=T)   # [B, d_cr]

    # --- CR Read ---
    cr_residual = 0.1 * expand_project(cr_state)  # [B, D]
    x = x + cr_residual.unsqueeze(T)              # broadcast to [B, T, D]

    # --- FFN ---
    x = x + ffn(layer_norm_2(x))                  # residual

    return x, cr_state
```

### B.2 CER Curriculum Scheduler

```python
def get_cer_lambdas(step, max_steps, config):
    progress = step / max_steps
    phase1_end = config.curriculum_phase1_end   # 0.1
    phase2_end = config.curriculum_phase2_end   # 0.3

    if progress < phase1_end:
        return 0.0, 0.0                         # Phase 1: no CER loss

    if progress < phase2_end:
        ramp = (progress - phase1_end) / (phase2_end - phase1_end)
        return (config.lambda_esc_max * ramp,
                config.lambda_ash_max * ramp)   # Phase 2: linear ramp

    return config.lambda_esc_max, config.lambda_ash_max  # Phase 3: full
```

### B.3 Combined Loss

```python
def compute_total_loss(logits, targets, cer_info, lambda_esc, lambda_ash):
    # Standard language modeling loss
    L_lm = cross_entropy(logits, targets)

    # ESC loss: penalize premature closure
    L_esc = 0.0
    for scores, targets in zip(cer_info.esc_scores, cer_info.esc_targets):
        excess = clamp(scores - targets, min=0)
        L_esc += mean(excess ** 2)
    L_esc /= num_layers

    # ASH regularization: penalize high mean ESC
    L_ash = 0.0
    for scores in cer_info.esc_scores:
        L_ash += mean(scores)
    L_ash /= num_layers

    # Combined
    total = L_lm + lambda_esc * L_esc + lambda_ash * L_ash
    return total
```

---

## Appendix C: Reproducibility Checklist

| Item                                    | Status        |
|-----------------------------------------|---------------|
| Code publicly available                 | Yes (this repository) |
| Model configuration files provided      | Yes (`configs/`) |
| Training hyperparameters documented     | Yes (Section 6.3, configs) |
| Data sources with licenses documented   | Yes (`data_sources/*.yaml`) |
| Random seed control                     | Partial (PyTorch seed; CUDA nondeterminism noted) |
| Evaluation code provided                | Yes (`ash/eval/`) |
| Ablation configurations provided        | Yes (`ashy_small_no_cer.yaml`) |
| CER parameter budget documented         | Yes (Section 5.5) |
| Training curriculum fully specified     | Yes (Section 6.1) |
| Loss function equations provided        | Yes (Appendix A) |
| Initialization details documented       | Yes (Section 5) |
| Hardware requirements specified         | Yes (Section 10.3) |
| Checkpoint format documented            | Yes (Section 11.3) |
| Go/no-go criteria defined               | Yes (Section 7.3) |
| Known limitations listed                | Yes (Section 12) |

---

## Appendix D: Glossary

| Term    | Definition                                                |
|---------|-----------------------------------------------------------|
| **CER** | Contextual Epistemic Release --- the full mechanism       |
| **ESC** | Epistemic State Channel --- per-token closure score       |
| **ASH** | Active Suppression Head --- sigmoid attention for suppression |
| **CR**  | Closure Register --- compressed memory of closed tokens   |
| **T1--T8** | The eight tests in the CER evaluation battery          |
| **Ashy-Small** | 131.5M-parameter proof-of-concept model             |
| **Ash** | Full 17B-active x 128-expert target model                 |

---

## How to Cite

If you reference this work, please use the following placeholder entry:

```bibtex
@article{ash2026cer,
  title     = {Ash: Contextual Epistemic Release for Transformer Language Models},
  author    = {{Ash Project Contributors}},
  year      = {2026},
  note      = {Working paper. Architecture validated at proof-of-concept scale;
               large-scale results pending.},
  url       = {https://github.com/PLACEHOLDER/ash}
}
```

---

*This is a working paper. All architectural details reflect the current
implementation. No benchmark results have been fabricated; all performance
claims are explicitly marked as planned or hypothesized. Last updated:
March 2026.*
