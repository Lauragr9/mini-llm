# Reproducing the results

> Fill in each section as the corresponding step is completed. Every command here
> must be copy-pasteable on the cluster. Never run training on the login node.

## 0. Environment

```bash
module load pytorch/2.8.0
```

| Item             | Version / value |
|------------------|-----------------|
| PyTorch module   | pytorch/2.8.0   |
| CUDA             | TODO            |
| NCCL             | TODO            |
| Python           | TODO            |
| Extra packages   | see `env/requirements.txt` |
| Slurm account    | TODO (gpu-prior+? — confirm with instructor) |

Cluster details: `env/cluster.md`.

## 1. Data (cpu partition)

```bash
# TODO: sbatch slurm/prepare_data.sbatch
```

- Sources: TODO (list of public-domain works + URLs)
- Validation hold-out: TODO (one or two complete works)

## 2. Tokenizer and encoding (cpu partition)

```bash
sbatch slurm/tokenize.sbatch      # needs data/clean/ from step 1
```

Equivalent manual commands (inside a job, never on the login node):

```bash
module load pytorch/2.8.0
pip install --user tokenizers
python3 src/train_tokenizer.py --vocab_size 8192
python3 src/encode.py --tiny_tokens 1500000
python3 src/test_roundtrip.py
```

- BPE vocabulary: 8,192, byte-level, trained on the training texts only; EOT id 0
- Outputs: `data/tokenized/train.bin` (4,761,128 tokens), `val.bin` (518,383),
  `tiny.bin` (1,500,000), `meta.json`, `tokenizer.json`
- Deterministic: reruns produce byte-identical files (SHA-256 in `meta.json`)
- Tested with tokenizers 0.23.3, numpy 2.4.4

## 3. CPU smoke test

```bash
# TODO: sbatch slurm/cpu_smoke.sbatch
```

## 4. Experiments

| Job | Script                 | Seeds | Notes |
|-----|------------------------|-------|-------|
| A   | TODO                   | TODO  |       |
| B   | TODO                   | TODO  |       |
| C   | TODO                   | TODO  |       |
| D   | TODO                   | TODO  |       |
| E   | TODO                   | TODO  |       |

## 5. Analysis

```bash
# TODO: python src/analyze.py results/runs.csv
```

## 6. Final model and samples

```bash
# TODO
```

## Job log index

Record every submitted job here (job ID, script, date, outcome). Raw logs live in
`slurm/logs/`.

| Job ID | Script | Submitted | Started | Outcome |
|--------|--------|-----------|---------|---------|
|        |        |           |         |         |
