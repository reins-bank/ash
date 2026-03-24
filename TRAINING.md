# Ash Training Token Plan

This document defines a non-random token strategy for both:
- **Ashy-Small (current repo model, ~131M params)**
- **Full Ash (target ~17B active MoE, 128 experts)**

## TL;DR

- Do **not** train on random tokens except for smoke tests.
- For real training, use curated, deduplicated corpora with fixed train/val/test splits.
- Treat token budget as staged gates, not a single giant run.
- For Full Ash, start with a **core target of 350B-500B tokens** before considering multi-trillion-token extension.

---

## 1) Correction: random tokens are not "hard mode"

You are right to avoid random data.

Random-token training is only useful to verify plumbing (throughput, shape checks, loss wiring). It does **not** teach language structure, and can hide whether CER is learning meaningful suppression/closure dynamics.

So: random fallback is for debug only, never for serious training.

---

## 2) Token-budget principles

1. **Quality > raw token count** (especially early). Dedup and filtering matter.
2. **Freeze evaluation sets early** to avoid moving targets.
3. **Track unique-token exposure** (repeat rate) so we do not just recycle the same data too often.
4. **Gate progression by CER battery results** (especially T7 Chekhov's Gun and T8 ESC Flow), not by spend alone.

---

## 3) Recommended staged plan (Full Ash)

### Phase A — Pipeline + architecture shakeout
- **Budget:** 5B-20B tokens
- **Goal:** stabilize distributed training + CER behavior + MoE routing
- **Exit criteria:**
  - No persistent instability (NaNs, router collapse, CR saturation)
  - T7 not catastrophic
  - T8 shows meaningful layerwise ESC progression

### Phase B — Core pretraining run
- **Budget:** 350B-500B tokens
- **Why this range:** roughly compute-optimal territory for ~17B active parameters
- **Goal:** establish strong base model quality and CER signal validity
- **Exit criteria:**
  - PPL trends still improving at acceptable slope
  - CER battery improvements persist, especially on long-context/noise tests

### Phase C — Frontier extension (optional, expensive)
- **Budget:** extend to 1T-2T tokens first, then evaluate up to 4T only if ROI holds
- **Goal:** push frontier quality and robustness if gains justify cost
- **Stop if:** marginal gains flatten, or CER regressions emerge under longer curricula

Note: Going straight to 2T-4T from day 1 is high-risk for a new architecture. Better to earn the right to scale.

---

## 4) Data mix recommendation (starting point)

Use the planned mix, with strict dedup + filtering:

- Web (deduped): 50%
- Code: 20%
- Books/long-form: 10%
- Scientific: 8%
- Wikipedia/encyclopedic: 5%
- Math/synthetic reasoning: 4%
- Instruction/dialogue synthetic: 3%

Additions:
- Keep a **high-quality reserve set** (~2-5%) that is never used in training, only for periodic deep eval.
- Enforce document-level and near-duplicate dedup across sources.

### 4.1 Corpora we can use now (to reduce internal data burden)

The goal is to avoid building everything from scratch. Start from mature public corpora, then layer your own curation.

#### Immediate bootstrap set (usable now)

- **FineWeb / FineWeb-Edu** (high-quality web text)
- **Dolma** (large curated text mixture)
- **Wikipedia + Wikibooks + Wikisource** dumps (encyclopedic + long-form)
- **The Stack v2 / StarCoderData** (code)
- **ArXiv + PubMed Open Access subsets** (scientific)
- **OpenWebMath / Proof-Pile-style math corpora** (math reasoning)
- **OpenAssistant / UltraChat-style instruction corpora** (for instruction flavor; keep low % in pretrain)

#### Datasets to avoid or quarantine by default

- Any corpus with unclear redistribution/training rights
- Legacy mixed corpora with known legal ambiguity (e.g., unlicensed book dumps)
- Data with weak provenance metadata

### 4.2 Practical sourcing policy (recommended)

For each dataset, record in a manifest:
- source URL / commit or snapshot ID
- license
- allowed use (research/commercial/unknown)
- dedup status
- quality filter version
- PII scrub version

Then enforce:
- only "allowed" data enters main pretrain
- "unknown" data goes to quarantine until reviewed

This reduces legal risk and prevents accidental contamination of the main run.

---

## 5) CER-aware curriculum mapped to token budget

CER schedule in code is phase-based by training progress:
- Phase 1: 0-10% (CER loss off)
- Phase 2: 10-30% (CER ramp)
- Phase 3: 30-100% (full CER)

Apply the same logic by tokens in each run:
- First 10% of run tokens: warm-up
- Next 20%: CER ramp
- Final 70%: full CER

Example for a 500B-token run:
- 0-50B: warm-up
- 50B-150B: CER ramp
- 150B-500B: full CER

---

## 6) Immediate practical plan for current Ashy-Small repo

Current default config processes approximately:

`tokens = max_steps * batch_size * gradient_accumulation_steps * block_size`

With defaults (100k, 12, 40, 1024):
- **~49.15B tokens processed**

For a 131M model, that is substantial and can imply heavy repetition depending on corpus size. Recommended:

1. Start with a **10B-20B token pilot** to validate CER and data pipeline.
2. Run full 49B only if pilot signals are clean (T7/T8, no pathological suppression).
3. If using OpenWebText-scale data, monitor repeat exposure closely.

---

## 7) Token accounting and reporting (must-have)

Log these continuously:
- Total tokens processed
- Estimated unique tokens seen
- Repeat ratio (processed / unique)
- Mix percentages actually sampled vs target
- CER diagnostics by layer (ESC means), suppression stats, CR norms
- MoE load-balance stats (for full Ash)

If repeat ratio rises too early, refresh with new shards before extending run length.

---

## 8) Recommended go/no-go gates

Proceed to larger token budgets only if all hold:
1. T7 (Chekhov's Gun) does not fail catastrophically
2. T8 (ESC Flow) shows meaningful non-trivial dynamics
3. No sustained training instabilities
4. Validation improvements remain monotonic enough to justify spend

If any fail, pause scale-up and fix architecture/data first.

---

## 9) Concrete recommendation

- **Do now:** non-random pilot on curated corpus (10B-20B equivalent at Ashy-Small scale; 5B-20B at Full-Ash prototype scale).
- **Then:** one serious core run targeting **350B-500B tokens** for Full Ash.
- **Only then:** consider extension toward 1T+ tokens if eval ROI is still strong.

This strategy reduces risk, preserves budget, and gives CER a fair, measurable test.
