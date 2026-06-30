# LLM-based Description Enrichment for Short Video Clustering

> **KDMiLe 2025** — Symposium on Knowledge Discovery, Mining and Learning
> Juliano Yugoshi · Ricardo Marcacini
> Institute of Mathematics and Computer Science, University of São Paulo (ICMC-USP), Brazil
> `{juliano.yugoshi, ricardo.marcacini}@usp.br`

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: MSR-VTT](https://img.shields.io/badge/Dataset-MSR--VTT-orange.svg)](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/)
[![KDMiLe 2025](https://img.shields.io/badge/Venue-KDMiLe%202025-red.svg)]()

---

<p align="center">
  <img src="results/figures/figura1.png" alt="5-Step Video Analysis Pipeline" width="820"/>
  <br/>
  <em>Figure 1 — Proposed 5-step pipeline: from raw short-video collection to unsupervised semantic clustering via compact VLM captioning, LLM enrichment, and dense text embedding.</em>
</p>

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Method](#method)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)

---

## Overview

The rapid growth of short-video platforms (TikTok, YouTube Shorts, Instagram Reels) has
generated massive unlabeled video collections that are expensive to organize manually.
**Video clustering** offers a scalable, unsupervised alternative, but its quality is limited
by the richness of the video representations used.

Compact Vision-Language Models (VLMs) produce short, generic captions that capture only
immediately visible elements, creating a *semantic gap* between low-level visual signals and
high-level meaning. This work addresses that gap by introducing an **LLM-based semantic
enrichment stage** between caption generation and embedding projection.

> **Core Question:** *Can LLM-based semantic enrichment of compact visual descriptions
> improve unsupervised short-video clustering quality?*

**Answer:** Yes — improvements of up to **13.18% in NMI** over the baseline are observed,
statistically validated via paired Wilcoxon tests (p < 10⁻³) and Cohen's d effect size analysis.

---

## Key Contributions

| # | Contribution |
|---|---|
| (i) | A **semantic enrichment method** for video clustering that improves unsupervised performance without fine-tuning or annotation |
| (ii) | Analysis of how **heterogeneous LLMs** contribute complementary semantic views of the same visual content |
| (iii) | Systematic empirical comparison of baseline vs. enriched representations across LLM × embedding configurations |
| (iv) | Statistical validation via **paired Wilcoxon tests** and **Cohen's d effect size** |

---

## Method

### Problem Definition

Let $\mathcal{V} = \{v_1, v_2, \ldots, v_n\}$ be a collection of unlabeled short videos. The
goal is to learn a clustering assignment $g : \mathcal{V} \rightarrow \{1, 2, \ldots, K\}$
**without** class labels during grouping. Clustering quality depends on the representation
function $f : \mathcal{V} \rightarrow \mathbb{R}^d$.

### Pipeline

The full representation is defined as:

$f_{m,e}(v_i) = \psi_e\!\left(T_m\!\left(\varphi(v_i),\, q\right)\right)$

where $\varphi$ is the compact VLM captioner (SmolVLM2), $T_m$ is the LLM enrichment function
for model $m$, and $\psi_e$ is the text embedding model $e$.

The **baseline** skips enrichment: $f_{0,e}(v_i) = \psi_e(\varphi(v_i))$.

Before clustering, vectors are standardized: $z = (x - \mu)/\sigma$.

Clustering objective (K-Means):

$\min_{\{\eta_k\}_{k=1}^K} \sum_{i=1}^n \min_{k \in \{1,\ldots,K\}} \left\| z_i^{(m,e)} - \eta_k \right\|_2^2$

### Enrichment Prompt

The following zero-shot prompt is applied uniformly to all LLMs:

