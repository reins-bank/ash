# Ash: SOTA Frontier LLM Build Plan

## Project Summary

**Ash** is a sparse Mixture-of-Experts (MoE) frontier language model augmented with the novel **Contextual Epistemic Release (CER)** framework. The architecture is loosely based on Llama 4 Maverick (17B active parameters × 128 experts) but introduces three architectural primitives — Epistemic State Channels (ESC), Active Suppression Heads (ASH), and the Closure Register (CR) — that give the model the ability to _forget well_: to recognise when information has been fully integrated and can be safely released from active attention without loss.

This is not a Llama fork. We are building from scratch, using Llama 4 as an architectural reference — particularly for CUDA-level optimizations (Flash Attention 2, RoPE, GQA) and MoE routing strategies.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Base Transformer Stack](#2-base-transformer-stack)
3. [Mixture of Experts Layer](#3-mixture-of-experts-layer)
4. [CER Framework Integration](#4-cer-framework-integration)
5. [Tokenizer and Vocabulary](#5-tokenizer-and-vocabulary)
6. [Training Infrastructure](#6-training-infrastructure)
7. [Data Pipeline](#7-data-pipeline)
8. [Training Protocol](#8-training-protocol)
9. [Evaluation and Test Battery](#9-evaluation-and-test-battery)
10. [Phased Build Plan](#10-phased-build-plan)
11. [Technology Stack](#11-technology-stack)
12. [Risk Register](#12-risk-register)

---

## 1. Architecture Overview

### 1.1 High-Level Specs

| Parameter | Value |
|---|---|
| Model name | Ash |
| Architecture | Sparse MoE Transformer + CER |
| Active params per token | ~17B |
| Total params (all experts) | ~400B+ (depends on expert sizing) |
| Expert count | 128 |
| Experts active per token | 2 (top-k routing) |
| Context length | 128K tokens (target), 1M stretch goal |
| Precision | BF16 training, FP8 inference |
| Positional encoding | RoPE (Rotary Position Embeddings) |
| Attention | GQA (Grouped Query Attention) + Flash Attention 2 |
| Novel components | ESC, ASH heads, Closure Register |
| CER overhead | ~4.6% additional parameters per block |

### 1.2 Why This Architecture

The thesis from the _Ashes to Attention_ paper is that standard attention is a mechanism for _inclusion_ — it learns what to look at. But reasoning also requires its complement: _informed forgetting_, the ability to recognise that information, once integrated, can be released from active consideration. Current transformers have no mechanism for this. Sparsity, pruning, and MoE are efficiency strategies — they don't represent epistemic closure.

CER fills this gap. MoE provides the scale efficiency (128 experts, 2 active = you only pay for 17B of compute per token). CER provides the _reasoning quality_ improvement by allowing the model to manage its own attention budget epistemically, not just computationally.

### 1.3 How CER Interacts with MoE

This is architecturally novel and needs careful thought:

- **MoE operates at the FFN level** — selecting which expert sub-networks activate per token. This is a _computational routing_ decision.
- **CER operates at the attention level** — tracking which tokens' information has been fully integrated and can be suppressed. This is an _epistemic state_ decision.
- **They are complementary, not competing.** MoE decides _which computation_ to run. CER decides _which information_ to attend to. A token can be routed to different experts at different layers while its ESC state simultaneously evolves from "open" to "closed."
- **Interaction point:** The Closure Register (CR) content is available to _all_ experts, not just the active ones. This means information that has been "released" from attention is still available as a compressed residual signal to any expert that needs it. The CR acts as a shared epistemic memory across the MoE routing boundary.

---

## 2. Base Transformer Stack

### 2.1 Transformer Block

Each layer follows this order (pre-norm architecture):

```
Input
  → RMSNorm
  → GQA Multi-Head Attention (with CER augmentation — see §4)
  → Residual connection
  → RMSNorm
  → MoE FFN (sparse, top-2 routing)
  → Residual connection
Output
```

### 2.2 Attention Mechanism

**Grouped Query Attention (GQA):**
- Reduces KV-cache memory by sharing key/value heads across query heads
- Llama 4 uses 8 KV heads for 64 query heads (8:1 ratio) — we adopt similar
- GQA maintains quality close to full MHA while dramatically reducing inference memory

**Rotary Position Embeddings (RoPE):**
- Applied to queries and keys, not values
- Encodes relative position through rotation in complex space
- Supports context length extension via NTK-aware interpolation or YaRN
- We target 128K native, with dynamic NTK scaling to 1M

**Flash Attention 2:**
- IO-aware exact attention algorithm — no approximation
- Fuses softmax, masking, and dropout into a single CUDA kernel
- Reduces memory from O(N²) to O(N) by tiling and recomputation
- Critical for 128K+ context training
- We use the Tri Dao implementation via `flash-attn` package, or integrate into our custom CUDA kernels

### 2.3 Normalization

- **RMSNorm** (not LayerNorm) — simpler, faster, empirically equivalent
- Pre-norm architecture (norm before attention and FFN, not after)
- No bias terms anywhere in the model (following Llama/PaLM convention)

### 2.4 Activation Function

- **SwiGLU** in the FFN: `SwiGLU(x) = Swish(xW₁) ⊙ (xW₂)` followed by projection `W₃`
- This is gated and has been shown to outperform ReLU/GELU at scale
- FFN hidden dim = 8/3 × model dim (rounded to nearest multiple of 256 for hardware efficiency)

### 2.5 Layer Count and Dimensions

| Component | Value | Notes |
|---|---|---|
| Layers | 48 | Deep enough for complex reasoning |
| Model dim (d_model) | 6144 | |
| Attention heads (query) | 64 | |
| KV heads | 8 | 8:1 GQA ratio |
| Head dim | 96 | d_model / n_heads |
| FFN hidden dim | 16384 | ~8/3 × 6144, rounded |
| Vocab size | 128K | BPE tokenizer |

Active parameters per token ≈ 17B (attention layers are dense; only FFN is sparse).

---

## 3. Mixture of Experts Layer

### 3.1 MoE Design

Each MoE layer replaces the standard dense FFN with:

```
Router(x) → top-k expert indices + gates
Expert_i(x) = SwiGLU FFN (per-expert weights)
Output = Σ gate_i × Expert_i(x) for selected experts
```

| MoE Parameter | Value |
|---|---|
| Total experts per layer | 128 |
| Active experts per token | 2 (top-2) |
| Router type | Token-choice with learned gating |
| Load balancing | Auxiliary loss (Switch Transformer style) + expert capacity factor |
| Expert parallelism | Yes (across GPUs) |
| Shared expert | 1 shared expert always active (Llama 4 approach) |

### 3.2 Router Design

- Linear projection from d_model to n_experts, followed by softmax
- Top-2 selection with renormalized gates
- **Auxiliary load-balancing loss** to prevent expert collapse (tokens all routing to same experts):
  ```
  L_balance = α × n_experts × Σ(f_i × p_i)
  ```
  where f_i = fraction of tokens routed to expert i, p_i = mean routing probability for expert i
- α = 0.01 (typical, tuned empirically)

### 3.3 Shared Expert

Following Llama 4 Maverick, one expert is always activated for every token regardless of routing. This:
- Provides a stable base capacity that all tokens benefit from
- Reduces the "expert lottery" effect where some tokens get poor expert matches
- Acts as a regularizer during training

### 3.4 Expert Capacity and Dropping

- Capacity factor = 1.25 (each expert can handle 25% more tokens than perfect balance)
- Tokens exceeding capacity are handled via auxiliary routing (routed to next-best expert), not dropped
- No token dropping in training — every token gets processed

### 3.5 Expert Parallelism Strategy

- Experts distributed across GPUs using Expert Parallelism (EP)
- All-to-all communication pattern for token dispatch/combine
- Co-designed with Tensor Parallelism and Pipeline Parallelism (see §6)

---

## 4. CER Framework Integration

This is the novel contribution. The CER framework from _Ashes to Attention_ introduces three architectural primitives that augment the standard attention mechanism.

### 4.1 Epistemic State Channels (ESC)

**What it does:** Maintains a per-token "epistemic score" that tracks whether the information carried by that token has been sufficiently integrated by the rest of the sequence.

**Implementation:**
- Each token position gets an additional scalar channel: `esc_score ∈ [0, 1]`
- Computed by a small MLP per layer: `esc_score = σ(W_esc · [h_t; attn_stats_t] + b_esc)`
  - `h_t` = hidden state of token t
  - `attn_stats_t` = aggregated attention statistics (how much other tokens attend to t)
  - σ = sigmoid
- Score evolves across layers: starts near 0 (open/active), approaches 1 (closed/integrated)
- A score of 1.0 means "this token's information has been fully absorbed — it can be released"

**Auxiliary Loss:**
```
L_esc = λ_esc × Σ_t max(0, esc_score_t - esc_target_t)²
```
- Encourages ESC scores to be _justified_ — tokens shouldn't be marked closed prematurely
- The target is derived from attention entropy: tokens that many other tokens still attend to should remain open
- λ_esc is curriculum-scheduled (starts at 0, ramps up during training)

**Expected dynamics across layers:**
- Early layers (1-12): Most ESC scores near 0. Model is still parsing.
- Middle layers (13-32): Functional tokens (articles, prepositions, resolved references) begin closing. ESC scores climb.
- Late layers (33-48): Substantial closure. Only tokens carrying unresolved semantic content remain open.

### 4.2 Active Suppression Heads (ASH)

**What it does:** Dedicated attention heads that learn to _suppress_ rather than attend. Where standard heads ask "what should I look at?", ASH heads ask "what should I stop looking at?"

**Key design decision — sigmoid, not softmax:**
- Standard attention uses softmax, which creates a probability distribution — attending _more_ to one token means attending _less_ to others. It's zero-sum.
- ASH heads use **sigmoid** activation, producing independent per-token suppression scores ∈ [0, 1]
- This allows the model to suppress multiple tokens simultaneously without the zero-sum constraint
- Suppression is not the inverse of attention — it's a separate, independently learned signal

**Integration into standard attention:**
```python
# Standard attention output
attn_out = softmax(QK^T / √d_k) @ V

# ASH suppression mask (per ASH head)
suppress = sigmoid(Q_ash @ K_ash^T / √d_k)  # shape: [seq_len, seq_len]

# Combined: element-wise gating
effective_attn = attn_out * (1 - λ_ash * suppress)
```
- λ_ash is a learnable per-head scalar, initialized small (0.01)
- This means ASH _modulates_ attention, it doesn't replace it
- The model learns when and how aggressively to suppress

**Head allocation:**
- 2 of the 64 query heads per layer are designated ASH heads
- These share the KV heads (via GQA) but have separate Q projections
- Parameter overhead: minimal — just additional Q projection weights + sigmoid vs softmax

### 4.3 Closure Register (CR)

**What it does:** When a token's ESC score crosses the closure threshold (e.g., 0.95), its information is "released" from active attention — but not destroyed. The CR captures a compressed summary of what was released, making it available as a low-bandwidth residual signal.

**Soft-write mechanism:**
```python
# For each token t where esc_score_t > threshold:
cr_write_gate = esc_score_t * sigmoid(W_cr_gate @ h_t)
cr_content = W_cr_compress @ h_t  # compress from d_model to d_cr
CR = CR + cr_write_gate * cr_content  # soft accumulate
```
- d_cr = d_model // 4 (compressed representation)
- Write is soft (gated by ESC score), not hard — allows gradient flow

**Residual injection:**
- The CR state is projected back to d_model and added to the residual stream
- Injection scale = 0.1 (small, so CR is a gentle bias, not a dominant signal)
- This means "forgotten" information is still faintly available — the model can reconstruct if needed
```python
residual = residual + 0.1 * W_cr_expand @ CR
```

**KV-cache extension:**
- CR state persists across the KV-cache during inference
- This is critical: during autoregressive generation, the CR accumulates across the full context
- Enables efficient long-context inference — tokens that have been "closed" need reduced KV-cache entries
- Potential for **adaptive KV-cache eviction**: tokens with ESC > 0.95 can have their full KV entries replaced by the CR summary, dramatically reducing memory

### 4.4 The CER Block: Full Data Flow

```
Input hidden state h
  │
  ├─→ RMSNorm → GQA Attention (62 standard heads + 2 ASH heads)
  │     │
  │     ├─→ Standard heads: softmax attention → attn_out_standard
  │     ├─→ ASH heads: sigmoid suppression → suppress_mask
  │     └─→ effective_attn = attn_out_standard ⊙ (1 - λ·suppress_mask)
  │
  ├─→ ESC Update: esc_score = σ(MLP([h; attn_stats]))
  │
  ├─→ Closure Register Write: CR += gate(esc) × compress(h)
  │
  ├─→ Closure Register Read: residual += 0.1 × expand(CR)
  │
  └─→ Residual add → RMSNorm → MoE FFN → Residual add → Output
```

### 4.5 CER Parameter Overhead

Per layer (approximate, at d_model=6144):
| Component | Parameters | Notes |
|---|---|---|
| ESC MLP | ~75K | Small 2-layer MLP |
| ASH Q projection | ~1.2M | 2 heads × separate Q weights |
| CR compress/expand | ~18.8M | d_model → d_cr → d_model |
| CR gate | ~37K | Gating MLP |
| **Per-layer total** | ~20.1M | |
| **All 48 layers** | ~965M | |
| **% of 17B active** | ~5.7% | Acceptable overhead |

This is a meaningful but manageable overhead. The paper demonstrated 4.6% on GPT-2 (117M); at our scale the relative overhead is similar.

---

## 5. Tokenizer and Vocabulary

### 5.1 Tokenizer Choice

| Decision | Choice | Rationale |
|---|---|---|
| Algorithm | BPE (Byte-Pair Encoding) | Industry standard, well-understood |
| Library | `sentencepiece` or `tiktoken` | Fast, battle-tested |
| Vocab size | 128,000 | Matches Llama 4; good balance of compression and granularity |
| Special tokens | `<bos>`, `<eos>`, `<pad>`, `<unk>`, `<|im_start|>`, `<|im_end|>` | Chat-template compatible |
| Byte fallback | Yes | Handle any UTF-8 input gracefully |
| Pre-tokenization | Whitespace + digit splitting | Improves arithmetic and code handling |

### 5.2 Training the Tokenizer

- Train on a representative sample of the training corpus (~10B tokens)
- Ensure good coverage of: English, code (Python, JS, C++, Rust, etc.), math notation, multilingual text
- Add domain-specific tokens for common programming constructs if needed

---

## 6. Training Infrastructure

### 6.1 Hardware Requirements

For a 17Bx128E model (~400B+ total params):

| Resource | Minimum | Target |
|---|---|---|
| GPUs | 256 × H100 80GB | 512 × H100 80GB |
| Interconnect | NVLink + InfiniBand 400Gb/s | NDR InfiniBand 400Gb/s |
| Storage | 100TB NVMe | 200TB NVMe (fast checkpointing) |
| Training time | ~4-8 weeks | Depends on data budget |
| Total compute | ~10²⁴ FLOPs | Chinchilla-optimal for this scale |

### 6.2 Parallelism Strategy

We need a 4D parallelism approach:

1. **Tensor Parallelism (TP):** Split attention heads and FFN columns across 8 GPUs within a node
2. **Expert Parallelism (EP):** Distribute 128 experts across 16 EP ranks (8 experts per rank)
3. **Pipeline Parallelism (PP):** Split 48 layers across 4 pipeline stages (12 layers each)
4. **Data Parallelism (DP):** Replicate across remaining GPU dimension with ZeRO Stage 1

Effective layout on 512 H100s:
```
512 GPUs = 8 (TP) × 16 (EP) × 4 (PP) × 1 (DP implied by EP overlap)
```

### 6.3 Framework Choice

| Option | Pros | Cons | Decision |
|---|---|---|---|
| **Megatron-LM** | Best MoE support, NVIDIA-optimized, proven at scale | Complex, NVIDIA-coupled | **Primary choice** |
| **DeepSpeed** | ZeRO, good EP support | MoE support less mature than Megatron | Secondary/hybrid |
| **FSDP (PyTorch native)** | Simpler, upstream PyTorch | Weaker MoE parallelism | Not for this scale |
| **JAX/Pax** | Excellent TPU support | We're targeting H100s | Not applicable |

**Decision: Megatron-LM** as the core training framework, with custom extensions for CER components. We will fork Megatron-Core and add CER as a first-class module.

### 6.4 Custom CUDA Kernels

We need custom kernels for:

1. **Fused CER attention kernel** — Extends Flash Attention 2 to include ASH sigmoid suppression in the same tiled kernel. Without this, ASH requires a separate attention pass (2× memory).
2. **ESC update kernel** — Fused sigmoid MLP + attention statistics aggregation. Small but called every layer.
3. **CR accumulation kernel** — Fused compress + gate + accumulate. Must be cache-friendly.
4. **MoE all-to-all with CER state** — Extend the standard MoE dispatch to include CR state propagation.

Libraries to build on:
- **Triton** for rapid kernel prototyping (then port to raw CUDA if perf-critical)
- **Flash Attention 2** codebase as starting point for the fused CER attention kernel
- **CUTLASS** for high-performance GEMM primitives

### 6.5 Mixed Precision Strategy

| Stage | Precision | Notes |
|---|---|---|
| Weights (master) | FP32 | Full precision for optimizer states |
| Forward/backward | BF16 | Standard for H100 training |
| Communication | BF16 | Reduce bandwidth |
| ESC scores | FP32 | Precision matters for epistemic thresholds |
| CR accumulation | FP32 | Soft-write accumulation needs precision |
| Inference | FP8 (E4M3) | H100 FP8 tensor cores for fast inference |

---

## 7. Data Pipeline

### 7.1 Training Data Budget

Following (modified) Chinchilla scaling:
- 17B active params × ~20 tokens/param = ~340B tokens minimum
- We target **2-4T tokens** for over-training (common practice for frontier models post-Chinchilla)

### 7.2 Data Mix

| Source | % of Mix | Purpose |
|---|---|---|
| Web crawl (deduplicated) | 50% | General knowledge, language patterns |
| Code (GitHub, StackOverflow) | 20% | Programming, logical reasoning |
| Books and long-form text | 10% | Extended reasoning, narrative coherence |
| Scientific papers (ArXiv, PubMed) | 8% | Technical reasoning, math |
| Wikipedia + encyclopedic | 5% | Factual grounding |
| Math datasets (synthetic) | 4% | Numerical reasoning |
| Instruction/dialogue (synthetic) | 3% | Conversational capability |

### 7.3 Data Processing Pipeline

```
Raw sources
  → Language detection (fastText lid)
  → Deduplication (MinHash + exact dedup)
  → Quality filtering (perplexity filter, heuristic rules)
  → PII removal (regex + NER-based)
  → Tokenization (BPE, 128K vocab)
  → Packing into sequences (128K tokens, with document boundaries)
  → Shuffle and shard for distributed training
```

### 7.4 Libraries

- **DataTrove** (HuggingFace) — scalable data processing pipeline
- **dolma** (AI2) — deduplication and filtering
- Custom PII scrubbing pipeline

---

## 8. Training Protocol

### 8.1 Optimizer

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| β₁, β₂ | 0.9, 0.95 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 (global norm) |
| Learning rate (peak) | 3e-4 |
| Warmup | 2000 steps (linear) |
| Schedule | Cosine decay to 3e-5 |
| Batch size | Ramp from 512 → 4096 sequences |

### 8.2 CER Training Curriculum

This is critical and novel — CER components need curriculum scheduling:

**Phase 1: Base model warm-up (0-10% of training)**
- CER components present but λ_esc = 0, λ_ash = 0
- Standard transformer training, MoE routing stabilizes
- ESC scores will be random/untrained — that's fine

**Phase 2: CER activation (10-30% of training)**
- Linear ramp: λ_esc from 0 → 0.1, λ_ash from 0 → 0.5
- ESC begins learning to track epistemic state
- ASH heads begin learning suppression patterns
- CR begins accumulating (but influence is minimal due to low ESC scores)

**Phase 3: Full CER (30-100% of training)**
- λ_esc = 0.1 (stable), λ_ash = 0.5 → 1.0
- Full CER dynamics active
- Monitor for mode collapse (all tokens closed, or none closed)
- Adaptive curriculum: if ESC scores saturate, reduce λ_esc temporarily

**Total loss:**
```
L = L_lm + α·L_balance + λ_esc·L_esc + λ_ash·L_ash_reg
```
where:
- L_lm = standard cross-entropy language modeling loss
- L_balance = MoE load-balancing auxiliary loss
- L_esc = ESC calibration loss
- L_ash_reg = ASH regularization (prevent ASH from suppressing everything)

### 8.3 Stability Monitoring

CER introduces new failure modes to watch for:

| Failure Mode | Detection | Mitigation |
|---|---|---|
| ESC collapse (all closed) | Mean ESC > 0.9 early in layers | Reduce λ_esc, increase regularization |
| ESC stagnation (none close) | Mean ESC < 0.1 in late layers | Increase λ_esc, check gradient flow |
| ASH over-suppression | Effective attention near zero | Clip ASH λ scalars, add L2 penalty |
| CR saturation | CR norm grows unbounded | Add CR norm penalty, decay old CR state |
| Expert collapse | < 10% of experts used | Standard MoE balancing fixes |
| Loss spikes | NaN or >5× sudden increase | LR reduction, gradient skip, checkpoint rollback |

### 8.4 Checkpointing

- Full checkpoint every 1000 steps (~2TB per checkpoint)
- Lightweight checkpoint (optimizer states only) every 100 steps
- Async checkpointing to NVMe to avoid training stalls
- Keep last 10 full checkpoints for rollback capability

---

## 9. Evaluation and Test Battery

### 9.1 Standard Benchmarks

| Benchmark | What it tests | Target |
|---|---|---|
| MMLU | Broad knowledge | >85% (5-shot) |
| HumanEval / MBPP | Code generation | >80% pass@1 |
| GSM8K | Math reasoning | >90% |
| MATH | Hard math | >60% |
| HellaSwag | Common sense | >90% |
| ARC-Challenge | Science reasoning | >85% |
| TruthfulQA | Hallucination resistance | >70% |
| WinoGrande | Coreference | >85% |
| GPQA | Expert-level QA | >45% |

### 9.2 CER-Specific Test Battery (from the paper)

These 8 tests are specifically designed to measure whether CER is working — whether the model is genuinely _forgetting well_ rather than just exhibiting coincidental sparsity:

| Test | What It Measures | Expected CER Advantage |
|---|---|---|
| **T1: Grokking Speed** | How fast the model achieves generalization on algorithmic tasks | CER should grok faster — it can release training-distribution artifacts |
| **T2: Garden Path Recovery** | Recovery from syntactically misleading sentences | CER should recover faster — it can release the initial misparse |
| **T3: Long Context Noise** | Performance degradation as irrelevant context grows | CER should degrade less — it actively suppresses noise tokens |
| **T4: Redundancy Tolerance** | Handling of repeated information | CER should handle better — redundant tokens get closed after first integration |
| **T5: Closure Register Probe** | Whether CR actually contains useful compressed information | Probing CR should recover token-level information above chance |
| **T6: Suppression Calibration** | Whether ASH suppresses the "right" tokens | Suppressed tokens should be ones a human judge deems irrelevant |
| **T7: Chekhov's Gun (Adversarial)** | Whether CER wrongly suppresses information that becomes relevant later | CER must NOT suppress too aggressively — this is the critical failure mode |
| **T8: Cross-Layer ESC Flow** | Whether ESC scores evolve meaningfully across layers | Should see clear open→closed trajectories for function words, stable-open for content words |

### 9.3 Long-Context Evaluation

| Test | Description |
|---|---|
| RULER | Synthetic retrieval at various context lengths |
| LongBench | Real-world long-context tasks |
| Needle-in-a-Haystack | Passkey retrieval at 128K+ |
| BABILong | Multi-hop reasoning over long contexts |

### 9.4 CER vs. Ablation Comparisons

We must train ablation models to prove CER's value:
1. **Ash-NoCER** — Same architecture, MoE, scale — but no ESC/ASH/CR
2. **Ash-ESConly** — ESC channels but no ASH heads or CR
3. **Ash-ASHonly** — ASH heads but no ESC tracking or CR
4. **Ash-NoCR** — ESC + ASH but no Closure Register
5. **Ash-Full** — Complete CER

This isolates the contribution of each component.

---

## 10. Phased Build Plan

### Phase 0: Foundations (Weeks 1-4)

- [ ] Set up the repo structure: `ash/model/`, `ash/training/`, `ash/data/`, `ash/eval/`, `ash/kernels/`
- [ ] Implement base transformer block (RMSNorm, GQA attention, SwiGLU FFN)
- [ ] Implement RoPE with NTK-aware scaling
- [ ] Implement MoE layer with top-2 routing and load balancing
- [ ] Unit tests for each component in isolation
- [ ] Train a tiny model (125M, 8 experts) on a small dataset to validate forward/backward pass
- [ ] Tokenizer training on representative corpus sample

### Phase 1: CER Implementation (Weeks 5-8)

- [ ] Implement Epistemic State Channels (ESC) module
- [ ] Implement Active Suppression Heads (ASH) with sigmoid attention
- [ ] Implement Closure Register (CR) with soft-write and residual injection
- [ ] Integrate CER into the transformer block (the CER Block from §4.4)
- [ ] Implement CER auxiliary losses (L_esc, L_ash_reg)
- [ ] Implement curriculum scheduling for CER loss weights
- [ ] Validate CER on the Ashy-Small proof-of-concept (GPT-2 117M scale + CER)
- [ ] Run the 8-test CER evaluation battery on Ashy-Small
- [ ] Confirm CER is learning meaningful patterns before scaling up

### Phase 2: CUDA Kernel Development (Weeks 6-10, parallel with Phase 1)

- [ ] Write fused CER attention kernel (Flash Attention 2 + ASH sigmoid) in Triton
- [ ] Write ESC update kernel
- [ ] Write CR accumulation kernel
- [ ] Benchmark kernels vs. naive PyTorch implementation
- [ ] Port performance-critical kernels from Triton to raw CUDA if needed
- [ ] Integration tests: fused kernels produce same results as reference implementation (within BF16 tolerance)

### Phase 3: Scaling Infrastructure (Weeks 9-12)

- [ ] Fork Megatron-Core, add CER module as first-class component
- [ ] Implement 4D parallelism (TP + EP + PP + DP) with CER state propagation
- [ ] Implement async checkpointing with CR state persistence
- [ ] Implement data pipeline (DataTrove-based) with quality filtering and dedup
- [ ] Scale test: train 1B × 16E model on 50B tokens with full CER
- [ ] Profile and optimize communication overhead from CER state
- [ ] Verify MoE + CER interaction at scale (expert utilization, ESC dynamics)

### Phase 4: Full-Scale Training (Weeks 13-24)

- [ ] Finalize data mix and tokenizer
- [ ] Begin Phase 1 training (base warm-up, no CER loss)
- [ ] Transition to Phase 2 training (CER activation ramp)
- [ ] Full CER training with monitoring dashboard
- [ ] Continuous evaluation on benchmark suite
- [ ] CER-specific probing throughout training (T5, T8)
- [ ] Handle stability issues as they arise (see §8.3)
- [ ] Ablation models trained in parallel on smaller compute allocation

### Phase 5: Post-Training (Weeks 25-28)

- [ ] Supervised Fine-Tuning (SFT) on instruction-following data
- [ ] RLHF or DPO alignment
- [ ] Evaluate whether CER dynamics survive/improve through post-training
- [ ] FP8 quantization for inference
- [ ] Adaptive KV-cache eviction using ESC scores (inference optimization)
- [ ] Final benchmark evaluation
- [ ] CER test battery final results

### Phase 6: Release and Documentation (Weeks 29-30)

- [ ] Model card and technical report
- [ ] Publish CER evaluation results (update the _Ashes to Attention_ paper)
- [ ] Open-weight release (if applicable)
- [ ] Inference serving setup (vLLM with CER-aware KV-cache management)

---

## 11. Technology Stack

### 11.1 Core Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ (training), CUDA C++ (kernels), Triton (kernel prototyping) |
| Framework | PyTorch 2.x + Megatron-Core |
| Distributed | NCCL, Megatron parallelism, DeepSpeed (selective) |
| Kernels | Flash Attention 2, CUTLASS, Triton, custom CUDA |
| Data | DataTrove, Apache Arrow, WebDataset |
| Tokenizer | SentencePiece or tiktoken |
| Monitoring | Weights & Biases (training curves, ESC dynamics, expert utilization) |
| Evaluation | lm-evaluation-harness (EleutherAI), custom CER battery |
| Inference | vLLM (extended with CER-aware KV-cache) |
| Hardware | NVIDIA H100 80GB SXM5, NVLink, InfiniBand |

### 11.2 Key Dependencies

```
torch >= 2.3
megatron-core >= 0.9
flash-attn >= 2.5
triton >= 3.0
sentencepiece >= 0.2
wandb
datatrove
lm-eval >= 0.4
```

---

## 12. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CER doesn't improve quality at scale | Medium | High | Early validation at 1B scale (Phase 3); ablation models; the 8-test battery provides clear go/no-go signal |
| CER destabilizes training | Medium | High | Curriculum scheduling; conservative λ ramps; extensive stability monitoring; checkpoint rollback |
| T7 (Chekhov's Gun) failure — model suppresses too aggressively | Medium | Critical | This is the core risk. Mitigation: conditional restoration mechanism in CR; adversarial training examples; aggressive T7 testing throughout |
| MoE + CER interaction causes expert collapse | Low | High | CER operates at attention level, MoE at FFN — largely orthogonal. Monitor expert utilization with CER active vs. ablation |
| Custom CUDA kernels have correctness bugs | Medium | Medium | Reference PyTorch implementation for bit-exact comparison; extensive numerical testing |
| Compute budget insufficient | Low | Critical | Phase 1-3 validation at smaller scales gives go/no-go before committing full compute |
| Data quality issues | Low | Medium | Standard mitigations: dedup, quality filtering, held-out validation set monitoring |
| ESC scores don't generalize to new domains | Medium | Medium | Evaluate ESC dynamics across diverse benchmarks; the auxiliary loss should force generalization |

---

## Open Questions

1. **CER at inference scale:** Can ESC-guided KV-cache eviction reduce memory enough to make 1M context practical at 17B active params? This could be a breakthrough inference optimization.

2. **CER + speculative decoding:** Do ASH heads provide a natural signal for which tokens are "safe" to speculate past? If so, CER could improve inference speed as well as quality.

3. **CER transfer:** If we pre-train Ash at 17Bx128E and then distill to a dense 7B model, do the CER behaviors transfer? Or is CER only useful at the scale where attention budget management matters?

4. **CER for multimodal:** Vision tokens in multimodal models are massively redundant (adjacent image patches carry similar information). ESC could learn to aggressively close redundant vision tokens, dramatically reducing cross-modal attention cost.

5. **Expert specialization and epistemic state:** Do certain experts become specialists at processing tokens that are in particular ESC states? E.g., do some experts specialize in "closing" information while others specialize in working with "open" tokens? This would be a fascinating emergent behavior.
