# Ash Storage and Data Marshaling Plan (AWS + 2TB Local per VPS)

This plan defines how Ash should store, version, move, and serve training data at scale, assuming:

- AWS services are available
- Each training VPS can attach up to **2TB local high-speed storage** (cache/staging)

The key design is: **S3 as durable source-of-truth, local 2TB as high-speed cache, training reads from local token shards only**.

---

## 1) Goals

1. Keep datasets reproducible and legally auditable.
2. Avoid retraining bottlenecks from remote reads.
3. Fit each node within 2TB local storage while supporting large global token budgets.
4. Make data movement deterministic via manifests and dataset versions.
5. Support resume/restart without losing preprocessing work.

---

## 2) High-Level Architecture

### Control plane
- Dataset manifests, mix weights, and run metadata in Git + S3 metadata files.
- Optional DynamoDB table for run registry and shard assignment state.

### Data plane (three tiers)
1. **Tier A (Durable canonical): S3**
   - Raw snapshots
   - Cleaned snapshots
   - Tokenized shards
   - Dataset manifests and checksums
   - Checkpoints and logs

2. **Tier B (Optional high-throughput shared): FSx for Lustre (if multi-node pressure grows)**
   - Used when many nodes need the same hot shard windows at once.
   - Linked to S3 as backing import/export.

3. **Tier C (Per-node hot cache): local 2TB NVMe/EBS**
   - Prefetched token shards for current + next training window.
   - Temporary staging for cleaning/tokenization workers.
   - Small local checkpoint buffer before S3 upload.

Training must consume Tier C only (local token files), never raw remote JSON.

---

## 3) S3 Bucket Layout

Use a dedicated bucket, e.g. `s3://ash-ml-data-prod/`.

```text
s3://ash-ml-data-prod/
  sources/
    raw/
      <source_name>/
        snapshot=<YYYYMMDD>/
          part-*.jsonl.zst
          _SUCCESS
          manifest.json
    clean/
      <source_name>/
        snapshot=<YYYYMMDD>/
          part-*.jsonl.zst
          manifest.json
    tokenized/
      tokenizer=<tokenizer_name>/
        dataset_version=<ver>/
          split=train/
            shard-000000.bin
            shard-000001.bin
            ...
            index.json
            checksums.sha256
          split=val/
            ...
  manifests/
    source_manifests/
      <source_name>.yaml
    dataset_versions/
      <dataset_version>.json
    training_mixes/
      <mix_name>.json
  runs/
    <run_id>/
      config.json
      sampled_shards.json
      metrics/
      logs/
  checkpoints/
    <run_id>/
      step-00010000/
      latest/
```

---

## 4) Local 2TB Allocation per VPS

Recommended partitioning:

- `1.4TB` token shard cache (active + lookahead)
- `300GB` preprocessing staging (download/decompress/intermediate)
- `200GB` checkpoint/log buffer
- `100GB` safety headroom (filesystem overhead + fragmentation)

Paths:

```text
/mnt/ash/
  cache/tokens/
  cache/prefetch/
  stage/raw/
  stage/clean/
  checkpoints/
  logs/
```

Eviction policy:
- LRU at shard level, pinned shards for current epoch window.
- Never evict shards currently leased by active dataloader workers.

---

## 5) Data Lifecycle

1. **Ingest**
   - Pull source-defined datasets using `data_sources/*.yaml`.
   - Store immutable raw snapshot in S3 with manifest and checksums.

2. **Clean + normalize**
   - Run declared cleaning pipeline.
   - Write compressed cleaned parts and metadata to S3.

3. **Tokenize + shard**
   - Convert cleaned text to uint16 token shards (`.bin`) + index metadata.
   - Target shard size: 0.5GB-2GB per shard (tunable by dataloader throughput).

4. **Publish dataset version**
   - Freeze exact shard list, source hashes, licenses, and mix constraints.
   - Write `manifests/dataset_versions/<version>.json`.

5. **Training prefetch + cache**
   - Node pulls only required shard window to local cache.
   - Background prefetch keeps N windows ahead.

6. **Checkpoint + metrics upload**
   - Local write first, async upload to S3.
   - Keep latest+N checkpoints local, full history in S3 lifecycle class.

---

## 6) Scripts/Tools to Build

These are the concrete tools needed to marshal data during runs.

### A) Source and manifest tools
1. `scripts/storage/build_source_manifest.py`
   - Input: source YAML + snapshot metadata
   - Output: normalized source manifest JSON (hashes, license, provenance)

2. `scripts/storage/validate_licenses.py`
   - Fails if `allowed_use: unknown` enters production dataset version.
   - Supports override allowlist with explicit approval file.

3. `scripts/storage/publish_dataset_version.py`
   - Freezes a dataset version from selected source snapshots.
   - Emits immutable `dataset_version.json`.

### B) Processing tools
4. `scripts/storage/fetch_to_s3.py`
   - Fetch from source YAML and write raw shards directly to S3.
   - Supports `--max-documents`, `--max-gb`, `--resume`.

5. `scripts/storage/clean_from_s3.py`
   - Read raw snapshot from S3, apply cleaning steps, write clean snapshot to S3.

6. `scripts/storage/tokenize_to_s3.py`
   - Read clean text from S3, tokenize, emit `.bin` shards + index to S3.
   - Supports parallel workers and deterministic sharding.

### C) Training-time data mover tools
7. `scripts/storage/prefetch_shards.py`
   - Given dataset version + run rank, prefetch shard window to local cache.
   - Maintains lookahead depth and cache watermark.

8. `scripts/storage/cache_manager.py`
   - LRU eviction, pin/unpin, disk watermark enforcement, cache stats.

9. `scripts/storage/lease_coordinator.py` (optional)
   - Prevents duplicate downloads across local workers.
   - Can be file-lock based; DynamoDB-backed for multi-host coordination.

10. `scripts/storage/sync_checkpoints.py`
   - Async upload/download, retry/backoff, verifies checksum.

### D) Operator tooling
11. `scripts/storage/storage_report.py`
   - Reports per-source size, per-snapshot size, local cache utilization.

12. `scripts/storage/prune_versions.py`
   - Retention policy for old snapshots/shards/checkpoints.

---

## 7) Runtime Training Data Flow

For each training VPS:

1. Read run config with `dataset_version` and `mix_name`.
2. Resolve shard sequence deterministically (seeded).
3. Prefetch worker downloads upcoming shards from S3 to local cache.
4. Dataloader consumes local cached shards only.
5. Cache manager evicts old shards when high-watermark reached.
6. On interruption, resume from local checkpoint + run state; refill cache from manifest.

This keeps training stable even with S3 latency spikes.

---

## 8) AWS Service Choices

### Required
- **S3**: canonical storage
- **IAM roles**: least-privilege read/write per environment
- **CloudWatch**: metrics/log alarms for data and training jobs

### Strongly recommended
- **S3 VPC endpoints**: private traffic, lower egress surprises
- **KMS encryption** for S3 objects
- **S3 Lifecycle policies**
  - recent shards/checkpoints in Standard
  - old artifacts transition to Infrequent Access / Glacier tiers

### Optional scale-up
- **FSx for Lustre**: if many nodes repeatedly read same shard sets and S3-only path bottlenecks
- **DynamoDB**: shard lease coordination and run metadata state
- **SQS**: queue-based preprocessing orchestration

---

## 9) Security and Compliance Controls

1. Enforce `allowed_use` policy at dataset publish time.
2. Attach dataset version metadata to every run for auditability.
3. Keep per-source provenance and hash manifests immutable.
4. Block production training if manifest has unresolved `unknown` license sources.
5. Maintain denylist for banned/ambiguous source IDs.

---

## 10) Capacity Planning Heuristics

### Tokenized storage
- uint16 token file size is ~`2 bytes/token`.
- 500B tokens ≈ ~1TB tokenized raw bytes.

### Working storage multiplier
Because of parallel staging, retries, and multiple representations, operational footprint is usually **5x-10x** tokenized active window globally.

With 2TB per node, avoid storing entire corpus locally. Keep only rolling shard windows.

---

## 11) Suggested Rollout

### Phase 1 (now)
- Implement scripts A/B/C with S3 backend.
- Keep single-node prefetch + cache manager simple (file locks + LRU).
- Start with capped snapshots and validate throughput.

### Phase 2
- Add robust checkpoint sync and retention policies.
- Add run-level metadata and storage report dashboards.

### Phase 3 (multi-node scaling)
- Add optional FSx for Lustre and distributed lease coordination.
- Tune shard size and prefetch depth with real throughput traces.

---

## 12) Non-Negotiable Rules

1. Do not train from raw text fetched live from internet sources.
2. Do not train directly from remote JSONL if avoidable.
3. Always train from versioned token shards with deterministic manifests.
4. Keep local 2TB for hot data only; canonical data belongs in S3.
5. Every run must record exact dataset version and shard list.
