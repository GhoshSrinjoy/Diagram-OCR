# Diagram Extraction Training Pipeline TODO

## 0. Project Goal

Build an end-to-end training and inference pipeline for extracting structured diagram content from images.

Primary outputs:

- Flowchart / graph / diagram image to Mermaid code.
- Diagram image to TikZ or other diagram code when the dataset provides it.
- Chart / plot image question answering and structured chart understanding.
- Science / educational diagram understanding as a secondary multimodal task.
- Optional document/OCR tasks only after the core diagram pipeline is stable.

Primary rule:

- Do not download the full 827.68 GB corpus at the start. Build the pipeline on the smaller, highest-signal datasets first, then scale.

## 1. Current Dataset Inventory

### Flowcharts / Graphs / Diagram Code

| Dataset | Size | Initial Priority |
| --- | ---: | --- |
| OpenRaiser/DiagramGen | 50.2 MB | High |
| vgbench/VGQA | 6.0 MB | High |
| CoreFidelity/Mermaid_50K | 67.0 MB | High |
| rakitha/mermaid-flowchart-transformer | 352.8 MB | High |
| TIGER-Lab/VisCode-Multi-679K | 4.21 GB | Medium |
| nllg/datikz | 614.4 MB | Medium |
| Kamizuru00/diagram_image_to_text | 244.7 MB | High |

### Charts / Plots

| Dataset | Size | Initial Priority |
| --- | ---: | --- |
| HuggingFaceM4/ChartQA | 919.8 MB | High |
| ahmed-masry/ChartQA | 1.72 GB | Medium |
| lmms-lab/ChartQA | 340.2 MB | Medium |
| achang/plot_qa | 30.01 GB | Later |
| Dodon/plotqa-dataset | 14.72 GB | Later |
| vikhyatk/figureqa | 2.07 GB | Medium |
| jrc/cleaned-plotqa-v2 | 116.2 MB | High |
| jrc/cleaned-figureqa-v2 | 23.8 MB | High |

### Document / OCR / Mixed Corpus

| Dataset | Size | Initial Priority |
| --- | ---: | --- |
| lmms-lab/DocVQA | 108.59 GB | Later |
| HuggingFaceM4/the_cauldron | 281.77 GB | Later |
| DermaVLM/PMC-Clinical-VQA | 287.92 GB | Skip unless needed |

### Science / Educational Diagrams

| Dataset | Size | Initial Priority |
| --- | ---: | --- |
| lmms-lab/ai2d | 133.0 MB | High |
| lmms-lab/ai2d-no-mask | 225.7 MB | Medium |
| andito/ai2d-merged | 820.6 MB | Medium |
| abhayzala/AI2D-Caption | 781.9 MB | High |
| derek-thomas/ScienceQA | 7.40 GB | Later |
| lmms-lab/ScienceQA-IMG | 2.54 GB | Medium |
| AI4Math/MathVista | 25.31 GB | Later |

### Dataset Notes

- Total known inventory: 827,680,294,674 bytes, about 827.7 GB decimal or 770.8 GiB binary.
- CoreFidelity/Mermaid_500k is gated and anonymous access was rejected, so it is excluded for now.
- The three largest repositories are about 93 percent of the total: PMC-Clinical-VQA, the_cauldron, and DocVQA.
- Dropping those three leaves roughly 92 GiB across the remaining 22 datasets.
- The current shortlist is about 141 GiB, but DocVQA alone is about 108 GiB.
- Dropping DocVQA from the shortlist leaves roughly 32 GiB.
- `usedStorage` may count both original archives and generated Parquet. Avoid pulling duplicate formats unless required.

## 2. Recommended Download Strategy

### Stage A: Small Core Diagram Pipeline

Download first:

- OpenRaiser/DiagramGen
- vgbench/VGQA
- CoreFidelity/Mermaid_50K
- rakitha/mermaid-flowchart-transformer
- Kamizuru00/diagram_image_to_text
- nllg/datikz
- lmms-lab/ai2d
- abhayzala/AI2D-Caption
- HuggingFaceM4/ChartQA
- jrc/cleaned-plotqa-v2
- jrc/cleaned-figureqa-v2

Expected size: small enough to build and debug locally before scaling.

### Stage B: Medium Multitask Expansion

Add after Stage A conversion and training works:

- TIGER-Lab/VisCode-Multi-679K
- vikhyatk/figureqa
- lmms-lab/ScienceQA-IMG
- ahmed-masry/ChartQA or lmms-lab/ChartQA, but avoid duplicated ChartQA variants unless they provide useful differences.

### Stage C: Large Optional Corpora

Add only if the model needs broader OCR and document understanding:

- lmms-lab/DocVQA, preferably only the exact config needed.
- achang/plot_qa or Dodon/plotqa-dataset, but avoid downloading both full versions until the duplicate/format overlap is known.
- AI4Math/MathVista if mathematical diagram reasoning becomes a target task.

### Stage D: Skip Unless There Is a Clear Need

- HuggingFaceM4/the_cauldron
- DermaVLM/PMC-Clinical-VQA

These are too large for the first training cycle and are not focused on diagram-code extraction.

## 3. Repository Structure To Create

```text
data/
  manifests/
    datasets.yaml
    downloads.lock.json
  raw/
    hf/
  processed/
    diagram_code/
    chart_qa/
    science_diagrams/
    doc_qa/
  samples/
models/
  base/
  adapters/
  merged/
runs/
  train/
  eval/
  inference/
configs/
  datasets/
  training/
  inference/
docker/
  Dockerfile
  docker-compose.yml
scripts/
  download_datasets.py
  inspect_dataset.py
  build_manifest.py
  convert_dataset.py
  validate_examples.py
  render_mermaid.py
  render_tikz.py
  train_lora.py
  evaluate.py
  infer.py
src/
  diagram_extraction/
    data/
    models/
    training/
    evaluation/
    inference/
notebooks/
reports/
```

## 4. Data Download Tasks

- [ ] Create `data/manifests/datasets.yaml`.
- [ ] Add one manifest entry per dataset:
  - Hugging Face repo ID.
  - Dataset type: diagram-code, chart-qa, science-diagram, doc-qa, or mixed.
  - Expected size from the inventory above.
  - Priority: high, medium, later, skip.
  - Required split/config names.
  - Preferred format: Parquet, image folder, JSON, JSONL, CSV, or archive.
  - Ignore patterns to avoid duplicate archives.
  - License and gated-access status.
- [ ] Create `scripts/download_datasets.py`.
- [ ] Use `huggingface_hub.snapshot_download` for repositories that need raw files.
- [ ] Use `datasets.load_dataset` for datasets where Parquet/config loading is cleaner.
- [ ] Add `--dataset`, `--stage`, `--all-high-priority`, `--local-dir`, `--cache-dir`, and `--dry-run` arguments.
- [ ] Add support for `ignore_patterns` so duplicate archives or generated files can be skipped.
- [ ] Add support for Hugging Face token auth through `HF_TOKEN` or `huggingface-cli login`.
- [ ] Pin each download to a resolved revision hash.
- [ ] Write all resolved revisions to `data/manifests/downloads.lock.json`.
- [ ] Log actual downloaded bytes per dataset.
- [ ] Create a `--verify-only` mode that checks file count, total bytes, and required files.
- [ ] Create a `--streaming` option for very large datasets.
- [ ] Do not download gated datasets until access is confirmed.

Recommended environment variables:

```powershell
$env:HF_HOME="D:\hf_cache"
$env:HF_DATASETS_CACHE="D:\hf_cache\datasets"
$env:TRANSFORMERS_CACHE="D:\hf_cache\models"
```

## 5. Data Processing Tasks

- [ ] Create a canonical schema for diagram-code examples.

```json
{
  "id": "source-specific-id",
  "source_dataset": "CoreFidelity/Mermaid_50K",
  "image_path": "data/raw-or-processed/path.png",
  "task": "image_to_mermaid",
  "prompt": "Convert this diagram to Mermaid flowchart code.",
  "target_text": "graph TD; A-->B;",
  "target_format": "mermaid",
  "metadata": {}
}
```

- [ ] Create a canonical schema for visual QA examples.

```json
{
  "id": "source-specific-id",
  "source_dataset": "HuggingFaceM4/ChartQA",
  "image_path": "data/raw-or-processed/path.png",
  "task": "chart_qa",
  "question": "What is the largest value?",
  "answer": "42",
  "metadata": {}
}
```

- [ ] Create converters for each high-priority dataset.
- [ ] Normalize image paths and copy or symlink images into `data/processed/...`.
- [ ] Convert all outputs to JSONL for training.
- [ ] Validate that every example has a readable image.
- [ ] Validate image dimensions and remove corrupt images.
- [ ] Normalize Mermaid code:
  - Strip markdown fences.
  - Normalize line endings.
  - Remove repeated boilerplate.
  - Preserve labels and graph direction.
- [ ] Normalize TikZ code:
  - Strip unrelated LaTeX wrappers only when safe.
  - Preserve package requirements in metadata.
- [ ] Deduplicate near-identical examples by image hash and target hash.
- [ ] Build train/validation/test splits with fixed random seeds.
- [ ] Save split files:
  - `data/processed/diagram_code/train.jsonl`
  - `data/processed/diagram_code/val.jsonl`
  - `data/processed/diagram_code/test.jsonl`
  - `data/processed/chart_qa/train.jsonl`
  - `data/processed/chart_qa/val.jsonl`
  - `data/processed/chart_qa/test.jsonl`
- [ ] Generate a 100-example smoke-test split.
- [ ] Generate a 1,000-example pilot split.
- [ ] Generate a dataset quality report in `reports/data_quality.md`.

## 6. Rendering And Validation Tasks

- [ ] Add Mermaid rendering support.
- [ ] Add TikZ rendering support if TikZ datasets are included.
- [ ] Add Graphviz support if any dataset emits DOT or graph structures.
- [ ] For every Mermaid target:
  - Check parse/render success.
  - Save render failure logs.
  - Track invalid syntax rate by dataset.
- [ ] For every TikZ target:
  - Check compile success.
  - Track missing package errors.
  - Save compile logs.
- [ ] Create visual samples showing:
  - Original image.
  - Target code.
  - Rendered target output.
- [ ] Use render success as a data quality filter, but keep rejected examples logged for inspection.

## 7. Base Model Tasks

Important assumption:

- Start from pretrained vision-language models. Training a useful VLM from random initialization is not realistic for the current dataset size unless there is very large compute available.

Tasks:

- [ ] Decide the first model family after checking current availability and GPU memory.
- [ ] Prefer a small or medium pretrained VLM for the first cycle.
- [ ] Download base checkpoints into `models/base/`.
- [ ] Record model revision hashes.
- [ ] Record tokenizer and processor revisions.
- [ ] Create `configs/training/model.yaml`.
- [ ] Create one baseline config for LoRA or QLoRA fine-tuning.
- [ ] Create one inference-only config for the unmodified base model.
- [ ] Run a zero-shot baseline before training.
- [ ] Store zero-shot outputs in `runs/inference/base_zero_shot/`.

Candidate model categories to evaluate:

- Pretrained VLM with image input and text generation.
- OCR-oriented image-to-text model.
- Encoder-decoder document understanding model.
- Smaller model for local smoke tests.
- Larger model only after the training/eval loop is stable.

Model download checklist:

- [ ] Verify license.
- [ ] Verify commercial/research restrictions.
- [ ] Verify max image resolution and context length.
- [ ] Verify supported quantization.
- [ ] Verify LoRA compatibility.
- [ ] Verify multi-GPU or DeepSpeed support if needed.
- [ ] Verify inference latency and memory usage.

## 8. Docker Environment Tasks

- [ ] Create `docker/Dockerfile`.
- [ ] Use a CUDA-enabled base image if GPU training is planned.
- [ ] Pin Python version.
- [ ] Pin PyTorch and CUDA compatibility.
- [ ] Install core ML packages:
  - `torch`
  - `torchvision`
  - `transformers`
  - `datasets`
  - `accelerate`
  - `peft`
  - `bitsandbytes`
  - `deepspeed`
  - `sentencepiece`
  - `safetensors`
- [ ] Install image and OCR packages:
  - `pillow`
  - `opencv-python`
  - `pytesseract`
  - `pdf2image`
  - `imagehash`
- [ ] Install evaluation packages:
  - `evaluate`
  - `scikit-learn`
  - `jiwer`
  - `sacrebleu`
  - `rouge-score`
- [ ] Install data tooling:
  - `huggingface_hub`
  - `pyarrow`
  - `pandas`
  - `orjson`
  - `tqdm`
- [ ] Install rendering tools:
  - Node.js
  - `@mermaid-js/mermaid-cli`
  - Graphviz
  - LaTeX packages for TikZ if needed
- [ ] Install experiment tracking:
  - `wandb` or `mlflow`
  - TensorBoard
- [ ] Create `docker/docker-compose.yml`.
- [ ] Mount:
  - Project directory.
  - Data directory.
  - Hugging Face cache.
  - Model output directory.
- [ ] Add GPU runtime support.
- [ ] Add a container smoke test:
  - Import PyTorch.
  - Check CUDA.
  - Import Transformers.
  - Render a tiny Mermaid graph.
  - Load one image.

## 9. Training Plan

### Training Stage 0: Pipeline Smoke Test

- [ ] Use 100 examples.
- [ ] Train for a very small number of steps.
- [ ] Confirm loss decreases.
- [ ] Confirm checkpoints are written.
- [ ] Confirm inference runs from the checkpoint.
- [ ] Confirm generated text can be parsed.

### Training Stage 1: Diagram-Code LoRA

- [ ] Train on the high-priority diagram-code datasets.
- [ ] Target outputs:
  - Mermaid first.
  - TikZ second if enough clean data exists.
- [ ] Use LoRA or QLoRA.
- [ ] Track:
  - Training loss.
  - Validation loss.
  - Exact match after normalization.
  - Render success rate.
  - Syntax error rate.
  - Average generated token length.
- [ ] Save adapter checkpoints to `models/adapters/diagram_code/`.

### Training Stage 2: Mixed Diagram And Chart Understanding

- [ ] Add ChartQA and cleaned PlotQA/FigureQA.
- [ ] Use prompt templates to separate tasks:
  - `image_to_mermaid`
  - `image_to_tikz`
  - `chart_qa`
  - `diagram_caption`
  - `science_diagram_qa`
- [ ] Balance datasets so small diagram-code data is not drowned out.
- [ ] Track per-task validation metrics.
- [ ] Save adapter checkpoints to `models/adapters/multitask/`.

### Training Stage 3: Medium Data Expansion

- [ ] Add TIGER-Lab/VisCode-Multi-679K.
- [ ] Add ScienceQA-IMG if science diagram reasoning is useful.
- [ ] Add FigureQA if chart reasoning is still weak.
- [ ] Re-run data balance analysis.
- [ ] Train with a longer schedule.
- [ ] Compare against Stage 1 and Stage 2.

### Training Stage 4: Large Optional Data

- [ ] Add DocVQA only if OCR/document understanding is needed.
- [ ] Pull only the required DocVQA config.
- [ ] Avoid InfographicVQA and full-resolution images unless needed.
- [ ] Consider streaming or shard-by-shard training.
- [ ] Do not add the_cauldron or PMC-Clinical-VQA unless there is a specific measurable gap.

## 10. Training Configuration Tasks

- [ ] Create `configs/training/smoke.yaml`.
- [ ] Create `configs/training/diagram_lora.yaml`.
- [ ] Create `configs/training/multitask_lora.yaml`.
- [ ] Create `configs/training/large_optional.yaml`.
- [ ] Include:
  - Model ID.
  - Model revision.
  - Dataset split paths.
  - Image resolution.
  - Max input tokens.
  - Max output tokens.
  - Batch size.
  - Gradient accumulation.
  - Learning rate.
  - Warmup steps.
  - Scheduler.
  - Precision: bf16/fp16/fp32.
  - LoRA rank, alpha, dropout, and target modules.
  - Checkpoint interval.
  - Eval interval.
  - Early stopping criteria.
  - Seed.
- [ ] Add resume-from-checkpoint support.
- [ ] Add gradient checkpointing support.
- [ ] Add optional DeepSpeed config.
- [ ] Add optional FlashAttention config if supported.

## 11. Evaluation Plan

### Diagram-Code Metrics

- [ ] Normalized exact match.
- [ ] Character error rate.
- [ ] Token-level F1.
- [ ] Mermaid parse success rate.
- [ ] Mermaid render success rate.
- [ ] TikZ compile success rate.
- [ ] Graph node/edge F1 where graph extraction is possible.
- [ ] Rendered-image similarity for target render vs generated render.
- [ ] Human review sample for outputs that render but are semantically wrong.

### Chart / Plot QA Metrics

- [ ] Exact answer accuracy.
- [ ] Relaxed numeric accuracy.
- [ ] Text answer F1.
- [ ] Per-chart-type accuracy where metadata allows.

### Science Diagram Metrics

- [ ] Multiple-choice accuracy where applicable.
- [ ] Caption similarity for caption datasets.
- [ ] Per-source validation breakdown.

### Evaluation Artifacts

- [ ] Save predictions as JSONL.
- [ ] Save failed parses separately.
- [ ] Save rendered predictions for visual inspection.
- [ ] Save confusion/error categories.
- [ ] Generate `reports/eval_report.md` per run.
- [ ] Compare every trained model against the zero-shot base model.

## 12. Inference Tasks

- [ ] Create `scripts/infer.py`.
- [ ] Support single-image inference.
- [ ] Support batch inference over a folder or JSONL file.
- [ ] Support task selection:
  - `image_to_mermaid`
  - `image_to_tikz`
  - `chart_qa`
  - `diagram_caption`
- [ ] Support optional question input for QA tasks.
- [ ] Support rendering generated Mermaid output.
- [ ] Save:
  - Raw generated text.
  - Normalized generated text.
  - Rendered preview image when applicable.
  - Metadata with model ID and checkpoint.
- [ ] Add CLI examples.

Example target commands:

```powershell
python scripts/infer.py --image samples/flowchart.png --task image_to_mermaid --checkpoint models/adapters/diagram_code/latest
python scripts/infer.py --image samples/chart.png --task chart_qa --question "What is the highest value?" --checkpoint models/adapters/multitask/latest
```

## 13. Serving Tasks

- [ ] Create a lightweight inference API after CLI inference works.
- [ ] Use FastAPI or a similar minimal server.
- [ ] Add endpoints:
  - `POST /infer/mermaid`
  - `POST /infer/tikz`
  - `POST /infer/chart-qa`
  - `GET /health`
- [ ] Add GPU memory checks at startup.
- [ ] Add model warmup.
- [ ] Add request size limits.
- [ ] Add batch inference endpoint only after single inference is stable.
- [ ] Add Docker compose profile for inference serving.

## 14. Experiment Tracking Tasks

- [ ] Choose tracking backend: W&B, MLflow, or TensorBoard-only.
- [ ] Log:
  - Dataset manifest hash.
  - Model revision.
  - Training config.
  - Git commit if the repo becomes a Git repository.
  - Training metrics.
  - Eval metrics.
  - Example predictions.
  - Render failure samples.
- [ ] Store run outputs under `runs/`.
- [ ] Keep a human-readable run index in `reports/runs.md`.

## 15. Licensing And Access Tasks

- [ ] Record license for every dataset.
- [ ] Record license for every base model.
- [ ] Record gated-access requirements.
- [ ] Record whether commercial use is allowed.
- [ ] Keep dataset attribution in `reports/dataset_licenses.md`.
- [ ] Do not redistribute gated or restricted data.
- [ ] Do not publish merged datasets until licenses are checked.

## 16. Quality Gates

Do not move to the next stage until these pass:

- [ ] Docker smoke test passes.
- [ ] At least one high-priority dataset downloads and verifies.
- [ ] At least one diagram-code dataset converts to canonical JSONL.
- [ ] Mermaid target rendering works on a sample.
- [ ] Training smoke test writes a checkpoint.
- [ ] Inference from the checkpoint works.
- [ ] Evaluation produces a report.
- [ ] Stage 1 model beats the zero-shot base model on at least one meaningful diagram-code metric.

## 17. Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Downloading duplicate archives and Parquet files | Wastes hundreds of GB | Use manifests, ignore patterns, and verify actual bytes |
| Gated datasets unavailable | Blocks planned data | Mark gated datasets as optional and keep ungated alternatives |
| Training from scratch underperforms | Wastes compute | Start with pretrained VLM and LoRA |
| Mermaid/TikZ targets contain invalid code | Bad supervision | Render/compile validation and dataset quality filtering |
| ChartQA variants duplicate examples | Data leakage and wasted storage | Hash images/questions/answers and deduplicate |
| Large OCR datasets dominate training | Model drifts away from diagram-code task | Task-balanced sampling |
| Generated code renders but is semantically wrong | Weak exact-match metrics | Add graph-level and visual similarity checks |
| Dataset licenses conflict | Cannot publish model/data | Track licenses before large training |

## 18. First Implementation Sprint

- [ ] Create repository scaffold.
- [ ] Create dataset manifest.
- [ ] Implement dataset dry-run download.
- [ ] Download only Stage A high-priority datasets.
- [ ] Implement converters for:
  - CoreFidelity/Mermaid_50K
  - rakitha/mermaid-flowchart-transformer
  - Kamizuru00/diagram_image_to_text
  - HuggingFaceM4/ChartQA
  - lmms-lab/ai2d
  - abhayzala/AI2D-Caption
- [ ] Build 100-example smoke split.
- [ ] Build Mermaid render validation.
- [ ] Create Dockerfile.
- [ ] Run Docker smoke test.
- [ ] Download one small pretrained VLM.
- [ ] Run zero-shot baseline.
- [ ] Run LoRA smoke training.
- [ ] Run inference from the smoke checkpoint.
- [ ] Write first evaluation report.

## 19. Definition Of Done For Version 0

- [ ] Data download is reproducible from manifests.
- [ ] Dataset revisions are locked.
- [ ] Processed JSONL splits are deterministic.
- [ ] Docker environment builds from scratch.
- [ ] Training can run from a config file.
- [ ] Inference can run on a single image.
- [ ] Mermaid outputs can be rendered automatically.
- [ ] Evaluation report is generated automatically.
- [ ] At least one trained checkpoint improves over the base model.
- [ ] All licenses and gated datasets are documented.

