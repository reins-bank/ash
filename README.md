# Ash

Ash is a frontier language model built from scratch, augmented with **Contextual Epistemic Release (CER)** — a novel architectural framework that gives transformers the ability to _forget well_.

The full Ash model targets a sparse Mixture-of-Experts architecture (17B active parameters x 128 experts), loosely based on Llama 4 Maverick. **Ashy-Small** is the current proof-of-concept: a GPT-2 scale (124M + 7M CER = 131M params) dense model with all three CER primitives integrated and trainable on a single GPU.

---

## What is CER?

Standard transformer attention is a mechanism for _inclusion_ — it learns what to look at. But human cognition derives equal power from its complement: **deliberate, contextually-grounded forgetting** — knowing what to _stop_ looking at, and why.

This is not sparsity, pruning, or MoE routing. Those are computational efficiency strategies. CER is an epistemically richer operation: the ability to recognise that a piece of information, once fully understood, can be safely released from active consideration — and that retaining it actively distorts subsequent reasoning.

CER introduces three architectural primitives into the transformer:

### Epistemic State Channels (ESC)

Each token gets a per-layer score in [0, 1] tracking whether its information has been sufficiently integrated by the rest of the sequence. A score near 0 means "open/active" — this token still carries unprocessed information. A score near 1 means "closed" — the model has absorbed what it needs.

ESC scores are computed by a small MLP that takes the token's hidden state and attention statistics (how much other tokens attend to it) as input. An auxiliary loss prevents premature closure — tokens that are still being attended to are penalised for closing.

### Active Suppression Heads (ASH)

Dedicated attention heads (2 of 12 in Ashy-Small) that use **sigmoid** activation instead of softmax. Where standard heads ask "what should I look at?", ASH heads ask "what should I stop looking at?"

Sigmoid produces _independent_ suppression scores per token — suppressing one token doesn't force attention onto another (no zero-sum constraint). The suppression modulates standard attention: `effective_attn = standard_attn * (1 - lambda * suppress)`, where lambda is a learnable per-head scalar initialised small (0.01).

### Closure Register (CR)

When tokens close (high ESC), their information isn't destroyed. The CR compresses it (768 -> 192 dims) and accumulates it into a sequence-level register. This register is injected back into the residual stream at a small scale (0.1x), providing a faint signal of "forgotten" information that the model can reconstruct from if needed.

The CR state persists across layers — information released at layer 4 is faintly available at layer 12.

### How They Work Together

```
Input hidden state
  |
  +---> LayerNorm --> Attention (10 standard + 2 ASH heads)
  |       |
  |       +---> Standard heads: softmax attention
  |       +---> ASH heads: sigmoid suppression mask
  |       +---> effective_attn = standard * (1 - lambda * suppress)
  |
  +---> ESC Update: score = sigmoid(MLP([hidden; attn_stats]))
  |
  +---> CR Write: register += esc_score * gate * compress(hidden)
  +---> CR Read:  residual += 0.1 * expand(register)
  |
  +---> LayerNorm --> FFN --> Output
```

For the full theoretical treatment, see [docs/ashes_to_attention.docx](docs/ashes_to_attention.docx).

---

## Ashy-Small Architecture

| Parameter | Value |
|---|---|
| Base architecture | GPT-2 (dense, pre-norm) |
| Layers | 12 |
| Model dimension | 768 |
| Attention heads | 12 (10 standard + 2 ASH) |
| FFN hidden dimension | 3072 |
| Vocabulary | 50,257 (GPT-2 BPE via tiktoken) |
| Context length | 1024 tokens |
| Base parameters | 124.4M |
| CER parameters | 7.1M (5.7% overhead) |
| **Total parameters** | **131.5M** |

---

## Setup

```bash
# Clone
git clone <repo-url> && cd ash

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Optional: install Modal for cloud GPU training
pip install -e ".[modal]"
```

Requires Python 3.10+ and PyTorch 2.1+.

---

## Training

### 1. Prepare data

Tokenize a dataset into memmap format (nanoGPT-style):

```bash
# WikiText-103 (smaller, good for validation)
python scripts/prepare_data.py --dataset wikitext --out-dir data/wikitext

# OpenWebText (full pretraining corpus)
python scripts/prepare_data.py --dataset openwebtext --out-dir data/openwebtext
```

This produces `train.bin` and `val.bin` — pre-tokenized uint16 binary files.

### 2. Train

```bash
# Full Ashy-Small with CER (requires GPU, ~100K steps)
python scripts/train.py --config configs/ashy_small.yaml

# Vanilla GPT-2 baseline (no CER, for ablation comparison)
python scripts/train.py --config configs/ashy_small_no_cer.yaml

# Tiny debug model (runs in <1 min on CPU, for testing the pipeline)
python scripts/train.py --config configs/ashy_small_debug.yaml
```

Set `data_dir` in the config YAML to point at your prepared data, or it will fall back to random tokens.

### 3. What to expect during training

Training logs include standard metrics plus CER-specific diagnostics:

```
step    500 | loss 6.4231 | lr 1.50e-04 | grad_norm 0.82 | tok/s 45000
step   1000 | loss 5.8712 | lr 3.00e-04 | grad_norm 0.65 | tok/s 44800 | lambda_esc 0.0200 lambda_ash 0.0100
```

**CER curriculum phases** (visible as lambda values appear):

- **Steps 0 - 10K** (Phase 1): Standard transformer training. CER components are present but their auxiliary losses are zeroed out. The model learns basic language modelling. You should see `lambda_esc 0.0000`.
- **Steps 10K - 30K** (Phase 2): CER loss ramps in. `lambda_esc` and `lambda_ash` increase linearly. ESC begins learning to track epistemic state. You may see slight loss instability here — this is normal.
- **Steps 30K - 100K** (Phase 3): Full CER. `lambda_esc 0.1000 lambda_ash 0.0500`. ESC scores should show meaningful dynamics across layers.

**Evaluation logs** (every `eval_interval` steps):

```
  eval | step 5000 | loss 5.12 | ppl 167.34 | ESC mean: [0.12, 0.13, 0.15, 0.18, 0.22, 0.28, 0.35, 0.41, 0.48, 0.55, 0.62, 0.68]
```

The ESC mean array shows the average epistemic score per layer. After sufficient training, you should see a **gradient from low (early layers) to high (late layers)** — meaning the model learns to progressively close tokens as their information is absorbed. This is the primary signal that CER is working.

**W&B logging**: If `wandb_enabled: true`, all metrics are logged to Weights & Biases including per-layer ESC means, loss components, and learning rate.

### 4. Checkpoints

Checkpoints are saved to `checkpoints/` and include model weights, optimizer state, step count, and full config. Resume training with:

```bash
python scripts/train.py --config configs/ashy_small.yaml --resume checkpoints/ckpt_best.pt
```

### 5. Training on Modal (cloud GPU)

For larger training runs, you can hand off to [Modal](https://modal.com)'s serverless GPUs. Training launches from your terminal and logs stream back in real time.

**One-time setup:**

```bash
# Install Modal
pip install -e ".[modal]"

# Authenticate with Modal
modal setup

# (Optional) Set up W&B logging on Modal
modal secret create wandb-secret WANDB_API_KEY=<your-key>

# Upload pre-tokenized data to a Modal Volume
# Works with both legacy flat layout (data/train.bin) and
# pipeline output (data/clean/<source>/tokens.bin)
make modal-upload-data
```

**Run training on Modal:**

```bash
# Uses configs/ashy_small_modal.yaml (A10G GPU, 2hr timeout)
make modal-train

# Or directly:
python scripts/train.py --config configs/ashy_small_modal.yaml
```

This runs the exact same training pipeline on a remote GPU. Checkpoints are saved to a persistent Modal Volume (`ash-checkpoints`).

**Configuring GPU and timeout:**

The `modal` section in any YAML config controls the handoff:

```yaml
training:
  modal:
    enabled: true       # set to false to run locally
    gpu: "A10G"         # GPU tier: "T4", "A10G", "A100", "H100"
    timeout: 7200       # max runtime in seconds
    data_volume: "ash-data"           # Modal Volume for training data
    checkpoint_volume: "ash-checkpoints"  # Modal Volume for checkpoints
    wandb_secret: "wandb-secret"      # Modal Secret name for W&B API key
```

GPU tiers in order of cost/performance: `T4` (cheapest, fine for GPT-2 scale) < `A10G` < `A100` < `H100`.

**Using pipeline data with Modal:**

After running the data pipeline locally, upload the tokenized output to Modal:

```bash
# Run pipeline to fetch + clean + tokenize a source
python scripts/pipeline.py run --source wikipedia

# Upload all .bin files (preserves directory structure)
make modal-upload-data

# Set data_dir in your config to match the volume path, e.g.:
#   data_dir: "clean/wikipedia"  (maps to /data/clean/wikipedia/ on the volume)
```

---

## Evaluation

### Perplexity

```bash
python scripts/eval.py --checkpoint checkpoints/ckpt_best.pt --eval perplexity
```

### CER Test Battery

The CER battery is a suite of 8 tests designed to distinguish genuine epistemic suppression from coincidental sparsity:

```bash
python scripts/eval.py --checkpoint checkpoints/ckpt_best.pt --battery cer
```

| Test | What it measures | What to look for |
|---|---|---|
| **T1: Grokking Speed** | Generalisation speed on algorithmic tasks | CER model groks faster |
| **T2: Garden Path** | Recovery from syntactic misparsing | Lower surprisal at disambiguation point |
| **T3: Noise Robustness** | Performance with irrelevant context padding | Less accuracy degradation as noise grows |
| **T4: Redundancy** | Handling of repeated information | Stable loss regardless of repetition count |
| **T5: CR Probe** | Whether the Closure Register encodes useful info | Probe accuracy above chance |
| **T6: Suppression Cal** | Whether ASH suppresses the right tokens | Function words suppressed more than content |
| **T7: Chekhov's Gun** | Whether CER wrongly suppresses info needed later | **Critical** — must NOT fail this test |
| **T8: ESC Flow** | Whether ESC scores evolve meaningfully across layers | Function words close early, content stays open |

T7 (Chekhov's Gun) is the most important failure mode test. If the model suppresses early information that becomes relevant later, CER is too aggressive and needs tuning.

### Visualise CER Dynamics

Generate an ESC heatmap for a given input:

```bash
python scripts/visualize_cer.py \
  --checkpoint checkpoints/ckpt_best.pt \
  --text "The horse raced past the barn fell." \
  --out esc_heatmap.png
```

This produces a [layers x tokens] heatmap showing how each token's epistemic state evolves through the network.

---

## Data Pipeline

The data pipeline framework manages training data sources declaratively via YAML. Each source defines its location, licensing/provenance metadata, fetch settings, and cleaning/standardization steps.

### Source definitions

Source YAMLs live in `data_sources/`. Shipped examples:

| Source | Type | Description |
|---|---|---|
| `fineweb` | HuggingFace | Curated web text from CommonCrawl |
| `fineweb_edu` | HuggingFace | Higher-quality educational web text |
| `redpajama_commoncrawl` | HuggingFace | Large general web shard (CC) |
| `the_stack_v2` | HuggingFace | Permissively-licensed source code |
| `redpajama_stackexchange` | HuggingFace | Technical Q&A style reasoning text |
| `redpajama_book` | HuggingFace | Long-form book-like corpus |
| `wikipedia` | HuggingFace | English Wikipedia articles |
| `arxiv` | HuggingFace | ArXiv scientific papers |
| `openwebmath` | HuggingFace | Math-heavy corpus for quantitative reasoning |
| `ultrachat_200k` | HuggingFace | Instruction/dialogue data (use low %) |

### YAML schema

```yaml
name: my_source
description: "What this source contains"

source:
  type: huggingface   # huggingface | url | local
  path: "org/dataset"
  subset: "default"   # optional
  split: "train"

license:
  name: "MIT"
  allowed_use: research  # research | commercial | unknown

provenance:
  dedup_status: "document-level"
  pii_scrub: "none"
  quality_filter: "custom v1"

fetch:
  max_documents: null   # null = all
  streaming: true
  output_dir: "data/raw/my_source"

cleaning:
  output_dir: "data/clean/my_source"
  steps:
    - name: strip_html
    - name: normalize_whitespace
    - name: min_length_filter
      params:
        min_chars: 100
    - name: dedup_exact
    - name: tokenize
      params:
        tokenizer: gpt2
```

Sources with `allowed_use: unknown` are flagged for quarantine per the [TRAINING.md](TRAINING.md) sourcing policy.

### CLI usage

```bash
# List all defined sources
python scripts/pipeline.py list

# Fetch a single source (dry run)
python scripts/pipeline.py fetch --source fineweb --dry-run

# Fetch all sources
python scripts/pipeline.py fetch --all

# Clean/standardize a source
python scripts/pipeline.py clean --source wikipedia

# Run full pipeline (fetch + clean) for all sources
python scripts/pipeline.py run --all

# Dry run the full pipeline
python scripts/pipeline.py run --all --dry-run

# Custom sources directory and output base
python scripts/pipeline.py run --all --sources-dir my_sources/ --base-dir /data/
```

Available cleaning steps: `strip_html`, `strip_latex_commands`, `normalize_whitespace`, `min_length_filter`, `max_length_filter`, `dedup_exact`, `tokenize`.

### Makefile shortcuts

```bash
make pipeline-list     # List sources
make pipeline-fetch    # Dry-run fetch all
make pipeline-clean    # Dry-run clean all
make pipeline-run SOURCE=fineweb  # Run pipeline for one source
```

---

## Running Tests

```bash
# Full test suite (42 tests)
pytest tests/ -v

# Individual component tests
pytest tests/test_esc.py -v          # Epistemic State Channels
pytest tests/test_ash_heads.py -v    # Active Suppression Heads
pytest tests/test_closure_register.py -v  # Closure Register
pytest tests/test_model.py -v        # Full model integration
pytest tests/test_training.py -v     # Training loop smoke test
```

Tests cover: output shapes, value ranges, gradient flow, causal masking, parameter counts, CER auxiliary losses, checkpoint roundtrip, and a 50-step training smoke test.

---

## Project Structure

```
ash/
├── ash/
│   ├── config.py              # ModelConfig, CERConfig, TrainingConfig, ModalConfig
│   ├── model/
│   │   ├── embeddings.py      # Token + positional embeddings
│   │   ├── attention.py       # Causal self-attention + ASH integration
│   │   ├── ffn.py             # Feed-forward network
│   │   ├── block.py           # Transformer block (attention + CER + FFN)
│   │   ├── gpt.py             # Full GPT model
│   │   └── cer/
│   │       ├── esc.py         # Epistemic State Channels
│   │       ├── ash_heads.py   # Active Suppression Heads
│   │       └── closure_register.py  # Closure Register
│   ├── training/
│   │   ├── losses.py          # CER auxiliary losses
│   │   ├── optimizer.py       # AdamW configuration
│   │   ├── scheduler.py       # LR schedule + CER curriculum
│   │   ├── checkpoint.py      # Save/load
│   │   ├── trainer.py         # Main training loop
│   │   └── runner.py          # Training orchestration (local + Modal)
│   ├── infra/
│   │   ├── modal_runner.py    # Modal GPU dispatch
│   │   └── modal_data.py      # Upload data to Modal Volumes
│   ├── pipeline/
│   │   ├── source.py          # YAML source loader and validator
│   │   ├── fetch.py           # Fetch from HuggingFace, URL, or local
│   │   ├── clean.py           # Cleaning step registry and executor
│   │   └── runner.py          # Pipeline orchestrator
│   ├── data/
│   │   ├── tokenizer.py       # tiktoken GPT-2 wrapper
│   │   └── dataset.py         # Memmap dataset
│   └── eval/
│       ├── cer_battery.py     # CER test suite runner
│       └── t1-t8 tests        # Individual CER evaluation tests
├── data_sources/              # YAML data source definitions (pipeline)
├── configs/                   # YAML training configs
├── scripts/                   # Entry points (train, eval, prepare_data, visualize)
├── tests/                     # 42 unit and integration tests
├── plans/
│   └── SOTA.md               # Full architecture plan for Ash at scale
└── docs/
    └── ashes_to_attention.docx  # CER theory paper
```

---

## Scaling to Full Ash

Ashy-Small is the proof-of-concept. The [full Ash architecture](plans/SOTA.md) targets:

| | Ashy-Small | Ash |
|---|---|---|
| Architecture | Dense GPT-2 | Sparse MoE |
| Active parameters | 131M | 17B |
| Total parameters | 131M | ~400B+ |
| Experts | — | 128 (top-2 + 1 shared) |
| Context length | 1,024 | 128K (1M stretch) |
| Attention | Learned positional + manual | RoPE + GQA + Flash Attention 2 |
| FFN | GELU | SwiGLU (per-expert) |
| Normalisation | LayerNorm | RMSNorm |
| Framework | Pure PyTorch | Megatron-Core (fork) |
| Hardware | 1x GPU | 512x H100 |

**What carries over from Ashy-Small:**
- The `ash/model/cer/` module (ESC, ASH, CR) lifts directly into the Megatron integration
- CER curriculum scheduling and auxiliary losses
- The 8-test CER evaluation battery

**What changes at scale:**
- MoE replaces dense FFN (128 experts, top-2 routing, 1 shared)
- CER operates at attention level, MoE at FFN level — they are orthogonal
- Custom fused CUDA kernel for Flash Attention 2 + ASH sigmoid (avoids 2x memory)
- The CR becomes shared across the MoE routing boundary — epistemic memory that all experts can access
- ESC-guided KV-cache eviction for efficient long-context inference

The go/no-go decision for the full scale build depends on Ashy-Small's CER battery results. Specifically: T7 (Chekhov's Gun) must not catastrophically fail, and T8 (ESC Flow) must show meaningful cross-layer dynamics on real training data.

---

## References

- **Ashes to Attention: On the Virtue of Forgetting Well** — A. Cole (working paper, [docs/ashes_to_attention.docx](docs/ashes_to_attention.docx))
- Llama 4 Maverick — [meta-llama/llama-models](https://github.com/meta-llama/llama-models/tree/main/models/llama4)
- nanoGPT — [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) (architectural inspiration for Ashy-Small)

---

## License

Proprietary. Copyright Reins Financial, Inc.
