# LLM-based Description Enrichment for Short Video Clustering

> **KDMiLe 2026** — Symposium on Knowledge Discovery, Mining and Learning<br>
> Juliano Yugoshi · Ricardo Marcacini<br>
> Institute of Mathematics and Computer Science, University of São Paulo (ICMC-USP), Brazil<br>
> `{juliano.yugoshi, ricardo.marcacini}@usp.br`

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: MSR-VTT](https://img.shields.io/badge/Dataset-MSR--VTT-orange.svg)](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bp7nTGCt1mV5ayxkgSpS5SltCM-d23GB)

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Method](#method)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Generated Outputs](#generated-outputs)
- [Pre-computed Files](#pre-computed-files)
- [Dataset](#dataset)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

The rapid growth of short-video platforms (TikTok, YouTube Shorts, Instagram Reels) has
generated massive unlabeled video collections that are prohibitively expensive to organize
manually. **Video clustering** offers a scalable, unsupervised alternative, yet its quality
is fundamentally constrained by the richness of the video representations used.

Compact Vision-Language Models (VLMs) produce short, generic captions that capture only
immediately visible elements, creating a *semantic gap* between low-level visual signals and
high-level thematic meaning. This work addresses that gap by introducing an
**LLM-based semantic enrichment stage** between caption generation and embedding projection.

> **Core Question:** *Can LLM-based semantic enrichment of compact visual descriptions
> improve unsupervised short-video clustering quality?*

**Answer:** Yes — improvements of up to **13.18% in NMI** over the baseline are observed,
statistically validated via paired Wilcoxon tests (p < 1e-3) and Cohen's d effect size analysis.

---

## Key Contributions

| # | Contribution |
|---|---|
| (i) | A **semantic enrichment method** for video clustering that improves unsupervised performance without fine-tuning or annotation |
| (ii) | Analysis of how **heterogeneous LLMs** contribute complementary semantic views of the same visual content |
| (iii) | Systematic empirical comparison of baseline vs. enriched representations across LLM x embedding configurations |
| (iv) | Statistical validation via **paired Wilcoxon Signed-Rank tests** and **Cohen's d effect size** across 15 independent runs |

---

## Method

### Problem Definition

Let V = {v1, v2, ..., vn} be a collection of unlabeled short videos. The goal is to learn a
clustering assignment g : V -> {1, 2, ..., K} **without** class labels during grouping.
Clustering quality depends critically on the representation function f : V -> R^d.

### Pipeline

The full enriched representation is defined as:

    f_{m,e}(v_i) = psi_e( T_m( phi(v_i), q ) )

where `phi` is the compact VLM captioner (SmolVLM2-2.2B), `T_m` is the LLM enrichment
function for model `m` in {Gemini, Claude, Llama}, and `psi_e` is the text embedding
model `e` in {Qwen, OpenAI}.

The **baseline** skips enrichment: `f_{0,e}(v_i) = psi_e( phi(v_i) )`.

Before clustering, all feature vectors are standardized: `z = (x - mu) / sigma`.

**Clustering objective (K-Means):**

    min sum_{i=1}^{n} min_{k} || z_i^{(m,e)} - eta_k ||_2^2

### Models Used

| Role | Model | Details |
|---|---|---|
| VLM Captioner | SmolVLM2-2.2B-Instruct | Frame-level visual description (1 fps, max 8 frames) |
| LLM Enrichment | google/gemini-2.5-flash | Via OpenRouter API |
| LLM Enrichment | anthropic/claude-3-haiku | Via OpenRouter API |
| LLM Enrichment | meta-llama/llama-3.3-70b-instruct | Via OpenRouter API |
| Text Embedding | qwen/qwen3-embedding-8b | 4096-dim, via OpenRouter |
| Text Embedding | openai/text-embedding-3-large | 3072-dim, via OpenRouter |
| Clustering | KMeans (k-means++) | K=20, n_init=10, 15 independent runs |

---

## Results

### Clustering Performance (MSR-VTT, K = 20, 15 runs)

| Representation | Embedding | NMI (higher is better) | ARI (higher is better) |
|---|---|---|---|
| Baseline | Qwen | 0.2943 +/- 0.0074 | 0.1757 +/- 0.0129 |
| Baseline | OpenAI | 0.3538 +/- 0.0106 | 0.2429 +/- 0.0138 |
| Gemini enriched | Qwen | 0.3082 +/- 0.0054 | 0.1853 +/- 0.0100 |
| Gemini enriched | OpenAI | 0.3548 +/- 0.0051 | 0.2356 +/- 0.0124 |
| Claude enriched | Qwen | 0.3331 +/- 0.0045 | 0.2183 +/- 0.0119 |
| **Claude enriched** | **OpenAI** | **0.3740 +/- 0.0023** | **0.2564 +/- 0.0088** |
| Llama enriched | Qwen | 0.3246 +/- 0.0040 | 0.2027 +/- 0.0064 |
| Llama enriched | OpenAI | 0.3549 +/- 0.0054 | 0.2389 +/- 0.0142 |

> Best result: **Claude + OpenAI -> NMI = 0.3740** (+5.71% over OpenAI baseline)
> Largest relative gain: **Claude + Qwen -> +13.18% NMI** over Qwen baseline
> Confidence intervals from nonparametric bootstrap (B = 1,000 resamples, 95% CI)

### Statistical Validation (Wilcoxon + Cohen's d)

| Embedding | Enrichment | p-value | Significant | Delta NMI | Cohen's d |
|---|---|---|---|---|---|
| Qwen | Gemini | 6.1e-5 | Yes | +0.0140 | 9.85 |
| Qwen | Claude | 6.1e-5 | Yes | +0.0391 | **18.96** |
| Qwen | Llama | 6.1e-5 | Yes | +0.0306 | 12.49 |
| OpenAI | Gemini | 0.1688 | No | +0.0010 | 0.39 |
| OpenAI | Claude | 6.1e-5 | Yes | +0.0209 | 3.51 |
| OpenAI | Llama | 0.1514 | No | +0.0011 | 0.31 |

> Paired Wilcoxon Signed-Rank test; 4 of 6 enriched variants significantly outperform baselines.

### NMI Sensitivity to Number of Clusters (K = 2 to 30)

The figure below shows NMI scores for all 8 configurations (3 LLMs + baseline x 2 embedding
models) as the number of clusters K varies from 2 to 30. The vertical dashed line marks
K = 20, which corresponds to the MSR-VTT ground-truth taxonomy. Most configurations,
especially with Qwen embeddings, reach a natural peak near K = 20, confirming that enriched
representations organically capture the dataset's thematic structure. Claude-enriched curves
consistently sit above the baseline across the full K range, evidencing robustness of the
semantic enrichment benefit beyond the fixed evaluation point.

<p align="center">
  <img src="results/figures/09_k_sensitivity_overlay_2embeddings.png"
       alt="K-Sensitivity Analysis: NMI vs Number of Clusters (K=2 to 30)"
       width="820"/>
  <br/>
  <em>
    Figure 2 — NMI sensitivity analysis for K in [2, 30].
    Circles = Qwen (4096-dim); Triangles = OpenAI (3072-dim).
    The dashed vertical line marks K = 20 (MSR-VTT taxonomy).
    Claude-enriched representations consistently achieve higher NMI across both embedding
    families, peaking naturally near the ground-truth cluster count.
  </em>
</p>

---

## Repository Structure

    video-clustering-enrichment/
    |
    |-- code-llm_based_description_enrichment.ipynb   # Main pipeline (Google Colab)
    |-- requirements.txt                               # Python dependencies
    |-- README.md
    |-- LICENSE
    |
    |-- results/
    |   └── figures/
    |       |-- 08_hypothesis_testing_qwen.png
    |       |-- 08_hypothesis_testing_openai.png
    |       |-- 09_k_sensitivity_sidebyside_2embeddings.png
    |       └── 09_k_sensitivity_overlay_2embeddings.png
    |
    └── MSRVTT_Workspace/                  # Created automatically on Google Drive
        |-- MSRVTT_Videos.zip              # MSR-VTT video files (user-provided)
        |-- MSR_VTT.json                   # MSR-VTT annotation file (user-provided)
        └── outputs/
            |-- MSRVTT_base.csv
            |-- MSRVTT_dados_multillm.csv
            |-- embeddings/
            |   |-- smolvlm_qwen_embeddings.npy
            |   |-- smolvlm_openai_embeddings.npy
            |   |-- gemini_qwen_embeddings.npy
            |   └── ...
            |-- stage2_smolvlm_checkpoint.csv
            |-- stage3_llm_checkpoint.csv
            |-- 06_summary_metrics_consolidated.csv
            |-- 07_hypothesis_testing.csv
            |-- 07_stats_summary_QWEN.csv
            |-- 07_stats_summary_OPENAI.csv
            |-- 09_k_sensitivity_QWEN.csv
            |-- 09_k_sensitivity_OPENAI.csv
            |-- embeddings_metadata.json
            └── FINAL_REPORT.txt

---

## Installation

### Prerequisites

- Python 3.10+
- A Google account with Google Drive (recommended: run on Google Colab with GPU)
- An [OpenRouter](https://openrouter.ai/) API key (for LLM enrichment and embeddings)
- The MSR-VTT dataset files: `MSRVTT_Videos.zip` and `MSR_VTT.json`

> **Recommended:** Run the notebook on [Google Colab](https://colab.research.google.com/)
> with a T4 or A100 GPU runtime for the SmolVLM2 inference step (Cell 4).
> All other cells can run on CPU.

### Option A — Google Colab (Recommended)

#### 1. Open the notebook directly

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bp7nTGCt1mV5ayxkgSpS5SltCM-d23GB)

#### 2. Set up Google Drive workspace

On your Google Drive, create the following folder structure and upload the dataset files:

    MyDrive/
    └── MSRVTT_Workspace/
        |-- MSRVTT_Videos.zip     <- MSR-VTT video archive
        └── MSR_VTT.json          <- MSR-VTT annotation file

#### 3. Set your API key in Colab Secrets

In the Colab sidebar, go to **Secrets** and add:

    Name  : OPENROUTER_API_KEY
    Value : your_openrouter_key_here

#### 4. Run all cells in order

    Cell 0  -> Install dependencies
    Cell 1  -> Imports + Config + Paths
    Cell 2  -> Utility functions
    Cell 3  -> Mount Drive + Build base CSV
    Cell 4  -> SmolVLM2 visual descriptions    <- GPU recommended
    Cell 5  -> LLM enrichment (async)          <- API calls
    Cell 6  -> Text embeddings (async)         <- API calls
    Cell 7  -> Feature preparation
    Cell 8  -> Multi-run KMeans clustering
    Cell 9  -> Wilcoxon tests + Cohen's d
    Cell 10 -> K-Sensitivity analysis
    Cell 11 -> Final export + report

### Option B — Local Installation

#### 1. Clone the repository

    git clone https://github.com/julianoyugoshi/video-clustering-enrichment.git
    cd video-clustering-enrichment

#### 2. Create and activate a virtual environment

    python -m venv venv
    source venv/bin/activate        # Linux/macOS
    # venv\Scripts\activate         # Windows

#### 3. Install dependencies

    pip install -r requirements.txt

#### 4. Configure environment variables

Create a `.env` file in the project root:

    OPENROUTER_API_KEY=your_openrouter_key_here

#### 5. Set up the workspace directories

    mkdir -p MSRVTT_Workspace/outputs/embeddings
    mkdir -p results/figures

Place your dataset files in `MSRVTT_Workspace/`:

    MSRVTT_Workspace/
    |-- MSRVTT_Videos.zip
    └── MSR_VTT.json

#### 6. Convert the notebook to a script (optional)

    pip install jupyter nbconvert
    jupyter nbconvert --to script code-llm_based_description_enrichment.ipynb
    python code-llm_based_description_enrichment.py

### Skipping Heavy Processing Steps

The pipeline includes **automatic checkpoint and cache detection** at every heavy stage.
To skip a stage, place the corresponding pre-computed file in `outputs/` before running.

| Stage | Skip Condition | File Required |
|---|---|---|
| Cell 4 — SmolVLM2 descriptions | File exists | `outputs/stage2_smolvlm_checkpoint.csv` |
| Cell 5 — LLM enrichment | File exists | `outputs/stage3_llm_checkpoint.csv` |
| Cell 6 — Text embeddings | All .npy files exist | `outputs/embeddings/{source}_{model}_embeddings.npy` |
| Cell 8 — Clustering stats | Both CSVs exist | `outputs/07_stats_summary_QWEN.csv` + `07_stats_summary_OPENAI.csv` |
| Cell 10 — K-Sensitivity | All 4 files exist | `09_k_sensitivity_QWEN.csv`, `09_k_sensitivity_OPENAI.csv` + both PNGs |

> 📁 All pre-computed files from the paper's experimental run are available for download — see [Pre-computed Files](#pre-computed-files) below.

---

## Usage

### Quick test run (reduced dataset)

To validate the pipeline quickly without processing all videos,
set `test_mode = True` in `ExperimentConfig` before running:

    # Cell 1 — ExperimentConfig
    config = ExperimentConfig(
        test_mode=True,
        test_sample_size=50,   # Use only 50 videos
        n_runs=3               # Reduce to 3 clustering runs
    )

### Full experimental run

Use the default configuration:

    config = ExperimentConfig(
        n_runs=15,
        n_clusters=20,
        test_mode=False
    )

---

## Configuration

All hyperparameters are controlled via the `ExperimentConfig` dataclass in Cell 1:

| Parameter | Default | Description |
|---|---|---|
| `n_runs` | `15` | Number of independent KMeans runs for statistical analysis |
| `n_clusters` | `20` | Number of clusters K (matches MSR-VTT ground-truth categories) |
| `test_mode` | `False` | If True, uses only `test_sample_size` videos |
| `test_sample_size` | `50` | Number of videos used in test mode |
| `smolvlm_path` | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` | VLM model from HuggingFace |
| `llm_models` | gemini / claude / llama | LLM enrichment models via OpenRouter |
| `drive_workspace` | `/content/drive/MyDrive/MSRVTT_Workspace` | Google Drive path |
| `random_seeds` | `[42, 43, ..., 56]` | Seeds for reproducibility across runs |

---

## Generated Outputs

After a complete run, the `outputs/` directory contains:

| File | Description |
|---|---|
| `MSRVTT_base.csv` | Base dataset with video IDs, categories, human captions |
| `MSRVTT_dados_multillm.csv` | Full dataset with SmolVLM2 + all LLM enriched descriptions |
| `embeddings/{source}_{model}_embeddings.npy` | Text embedding arrays (8 files total) |
| `00_llm_predicted_categories_validation.csv` | LLM category prediction validity rates |
| `00_llm_categories_vs_groundtruth.csv` | LLM category hit rate vs. ground truth |
| `06_summary_metrics_consolidated.csv` | NMI/ARI statistics for all configurations |
| `07_stats_summary_QWEN.csv` | Per-source stats for Qwen embeddings |
| `07_stats_summary_OPENAI.csv` | Per-source stats for OpenAI embeddings |
| `07_hypothesis_testing.csv` | Wilcoxon p-values and Cohen's d results |
| `08_hypothesis_testing_qwen.png` | Statistical validation figure — Qwen |
| `08_hypothesis_testing_openai.png` | Statistical validation figure — OpenAI |
| `09_k_sensitivity_sidebyside_2embeddings.png` | K-Sensitivity side-by-side plot |
| `09_k_sensitivity_overlay_2embeddings.png` | K-Sensitivity overlay plot |
| `embeddings_metadata.json` | Metadata for reproducibility |
| `FINAL_REPORT.txt` | Full experiment summary with all tables |

---

## Pre-computed Files

To skip the heavy processing steps, all pre-computed checkpoints, embeddings and outputs
from the paper's experimental run are available for download:

📁 **[Archives-LLMenrichment — Google Drive (read-only)](https://drive.google.com/drive/folders/SEU_LINK_AQUI)**

After downloading, place the files in `MSRVTT_Workspace/outputs/` before running:

    MSRVTT_Workspace/
    └── outputs/
        ├── MSRVTT_base.csv
        ├── MSRVTT_dados_multillm.csv
        ├── stage2_smolvlm_checkpoint.csv
        ├── stage3_llm_checkpoint.csv
        ├── 06_summary_metrics_consolidated.csv
        ├── 07_hypothesis_testing.csv
        ├── 07_stats_summary_QWEN.csv
        ├── 07_stats_summary_OPENAI.csv
        ├── 09_k_sensitivity_QWEN.csv
        ├── 09_k_sensitivity_OPENAI.csv
        ├── embeddings_metadata.json
        ├── FINAL_REPORT.txt
        └── embeddings/
            ├── smolvlm_qwen_embeddings.npy
            ├── smolvlm_openai_embeddings.npy
            ├── gemini_qwen_embeddings.npy
            ├── gemini_openai_embeddings.npy
            ├── claude_qwen_embeddings.npy
            ├── claude_openai_embeddings.npy
            ├── llama_qwen_embeddings.npy
            └── llama_openai_embeddings.npy

| File | Skips Stage |
|---|---|
| `stage2_smolvlm_checkpoint.csv` | Cell 4 — SmolVLM2 descriptions |
| `stage3_llm_checkpoint.csv` | Cell 5 — LLM enrichment |
| `embeddings/*.npy` | Cell 6 — Text embeddings |
| `07_stats_summary_QWEN/OPENAI.csv` | Cell 8 — Clustering stats |
| `09_k_sensitivity_*.csv` + PNGs | Cell 10 — K-Sensitivity |

---

## Dataset

This work uses the **MSR-VTT** dataset:

- **Videos:** 10,000 short video clips across 20 semantic categories
- **Annotations:** Human captions + category labels
- **Access:** [MSR-VTT at Microsoft Research](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)

**MSR-VTT categories (20):**

    music, people, gaming, sports/actions, news/events/politics, education,
    tv shows, movie/comedy, animation, vehicles/autos, howto, travel,
    science/technology, animals/pets, kids/family, documentary,
    food/drink, cooking, beauty/fashion, advertisement

---

## Citation

If you use this code or findings in your research, please cite:

    @inproceedings{yugoshi2026llm,
      title        = {LLM-based Description Enrichment for Short Video Clustering},
      author       = {Yugoshi, Juliano and Marcacini, Ricardo},
      booktitle    = {Proceedings of the Symposium on Knowledge Discovery,
                      Mining and Learning (KDMiLe)},
      year         = {2026},
      address      = {Brazil},
      institution  = {Institute of Mathematics and Computer Science,
                      University of Sao Paulo (ICMC-USP)}
    }

---

## Acknowledgements

This work was carried out at:

- **ICMC-USP** — Institute of Mathematics and Computer Science, University of Sao Paulo
- **UFMS-CPTL** — Federal University of Mato Grosso do Sul

Models and infrastructure:

- [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) — Hugging Face
- [OpenRouter](https://openrouter.ai/) — Unified LLM API Gateway

---

## License

This project is licensed under the MIT License.

    MIT License

    Copyright (c) 2026 Juliano Yugoshi, Ricardo Marcacini
    Institute of Mathematics and Computer Science
    University of Sao Paulo (ICMC-USP)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

<p align="center">
  Made with care at <strong>ICMC-USP</strong> — Sao Carlos, Brazil
</p>
````
