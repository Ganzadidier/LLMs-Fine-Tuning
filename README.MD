# 🏥 MedAssist — Domain-Specific Medical Education Assistant

> Fine-tuned `Qwen2.5-1.5B-Instruct` with LoRA on 2,000 medical Q&A pairs, deployed via an interactive Gradio chatbot.

Main_notebook : https://colab.research.google.com/drive/1SV_iQ8G1mj3RC6WjhWPqsu2zKAOzw5HG?usp=drive_link

Qucik_Demo : https://colab.research.google.com/drive/1yiq7booGodwyr0j4z4RA_8M9FXkbcn4Q?usp=sharing

HuggingFace Model : https://huggingface.co/dganza/medassist-qwen25-lora

---

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Model & Training Details](#model--training-details)
- [Experiments](#experiments)
- [Results](#results)
- [Quick Demo — Try it in 5 minutes](#quick-demo--try-it-in-5-minutes)
- [Running the Full Training Notebook](#running-the-full-training-notebook)
- [Gradio UI](#gradio-ui)
- [Requirements](#requirements)

---

## Overview

MedAssist is a **parameter-efficient fine-tuned language model** for medical education. It uses LoRA (Low-Rank Adaptation) to fine-tune `Qwen2.5-1.5B-Instruct` on 2,000 curated medical question-answer pairs, then deploys the best-performing adapter as an interactive chatbot via Gradio.

The project systematically compares **4 LoRA configurations** across key metrics (eval loss, BLEU, ROUGE, Token-F1, keyphrase accuracy) to identify the optimal training setup for medical domain adaptation.

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Method** | LoRA fine-tuning with 4-bit quantization (bitsandbytes) |
| **Dataset** | `medalpaca/medical_meadow_medical_flashcards` — 2,000 pairs |
| **Domains** | Pharmacology · Pathology · Anatomy · Physiology · Diagnosis · General Medicine |
| **Best config** | Exp4 — LoRA r=64, LR=2e-4, 2 epochs |
| **Model weights** | [HuggingFace 🤗](https://huggingface.co/dganza/medassist-qwen25-lora) |

> ⚠️ **Educational use only.** MedAssist is not a substitute for professional medical advice, diagnosis, or treatment.

---

## Project Structure

```
📦 medassist/
├── 📓 Medassist_qwen25_lora_finetune__1__final.ipynb   ← Full training pipeline
├── 📓 MedAssist_Quick_Demo.ipynb                        ← Quick inference demo
└── 📄 README.md
```

The fine-tuned LoRA adapter (228MB) is hosted on Hugging Face — no large files in this repo.

---

## Model & Training Details

### Base Model
`Qwen/Qwen2.5-1.5B-Instruct` — a 1.5 billion parameter instruction-tuned model from Alibaba Cloud, loaded with 4-bit NF4 quantization to fit on a single T4 GPU.

### Dataset
`medalpaca/medical_meadow_medical_flashcards` loaded directly from HuggingFace. After quality filtering (removing too-short, too-long, or duplicate pairs), 2,000 examples are sampled and split 90/10 into train and validation sets.

### LoRA Configuration (shared across experiments)
| Parameter | Value |
|---|---|
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| LoRA alpha | 2× rank |
| Dropout | 0.05 |
| Batch size | 1 × gradient accumulation 16 (effective batch = 16) |
| Optimizer | AdamW (paged) |
| Scheduler | Cosine |
| Max sequence length | 256 tokens |

---

## Experiments

Four LoRA configurations were trained and evaluated:

| Experiment | LoRA r | Learning Rate | Epochs | Description |
|---|---|---|---|---|
| **Exp1** | 8 | 2e-4 | 1 | Conservative baseline — fewest parameters, single pass |
| **Exp2** | 16 | 1e-4 | 2 | Balanced — doubled rank, halved LR, more convergence time |
| **Exp3** | 32 | 5e-5 | 2 | Conservative LR — tests careful fine-tuning with higher rank |
| **Exp4** ⭐ | 64 | 2e-4 | 2 | High rank + aggressive LR — maximum expressiveness |

---

## Results

**Best experiment: Exp4 (r=64, LR=2e-4, 2 epochs)** — achieved the lowest eval loss (0.5986).

### Metrics evaluated
- **Eval Loss / Perplexity** — primary training objective
- **BLEU** — n-gram overlap with reference answers
- **ROUGE-1 / ROUGE-2 / ROUGE-L** — recall-oriented overlap
- **Token-F1** — token-level precision/recall
- **Exact Match** — strict answer matching
- **Keyphrase Accuracy** — presence of domain-specific medical terms

### Visualizations (in Section 9 of the main notebook)
- 📊 Bar charts across all 6 metrics per experiment
- 🕸️ Radar chart for multi-metric comparison
- 📉 Training & eval loss curves for each experiment
- 🔬 Keyphrase accuracy by medical category (grouped bar)
- 🌡️ Metric heatmap across all experiments
- ⚡ Training efficiency scatter (time vs Token-F1)

### Base Model vs Fine-Tuned (Section 10)
Qualitative and quantitative comparison on 30 held-out samples shows consistent improvement on in-domain medical questions across all metrics after fine-tuning.

---

## Quick Demo — Try it in 5 minutes

> **No training required.** The model is downloaded directly from HuggingFace.

### Option 1 — Google Colab (recommended)

1. Click the badge below to open the demo notebook in Colab:

   [![Open Quick Demo In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo/blob/main/MedAssist_Quick_Demo.ipynb)

2. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU → Save**

3. Click **Runtime → Run all**

4. Wait for a public Gradio link to appear (looks like `https://xxxx.gradio.live`) — click it to open the chatbot

### Option 2 — Local machine

Requirements: Python 3.9+, NVIDIA GPU with 4GB+ VRAM (CPU works but is very slow)

```bash
# 1. Clone the repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

# 2. Install dependencies
pip install transformers peft accelerate bitsandbytes gradio huggingface_hub torch

# 3. Download model from HuggingFace
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='dganza/medassist-qwen25-lora', local_dir='./medassist_best_model')
print('Downloaded!')
"

# 4. Open and run MedAssist_Quick_Demo.ipynb in Jupyter
jupyter notebook MedAssist_Quick_Demo.ipynb
```

### What the demo notebook does

| Step | What happens |
|---|---|
| Step 1 | Installs all required packages |
| Step 2 | Downloads the LoRA adapter from HuggingFace (~228MB, skipped if already present) |
| Step 3 | Loads base model + LoRA adapter with 4-bit quantization |
| Step 4 | Defines the `chat()` inference function |
| Step 5 | Launches the Gradio web interface |

---

## Running the Full Training Notebook

> ⏱️ **Estimated time: 2–3 hours** on a Colab T4 GPU (4 experiments × ~30 min each)

1. Open the main notebook in Colab:

   [![Open Main Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo/blob/main/Medassist_qwen25_lora_finetune__1__final.ipynb)

2. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU → Save**

3. Run cells **in order, top to bottom** — each section builds on the previous

### Notebook structure

| Section | Description | Est. Time |
|---|---|---|
| **0** | Install packages & imports | ~2 min |
| **1** | Load dataset from HuggingFace + EDA | ~1 min |
| **2** | Prompt template & preprocessing | ~1 min |
| **3** | Tokenizer loading + tokenization | ~1 min |
| **4** | Shared metrics & utility functions | ~1 min |
| **5** | Experiment 1 — r=8, LR=2e-4, 1 epoch | ~20 min |
| **6** | Experiment 2 — r=16, LR=1e-4, 2 epochs | ~35 min |
| **7** | Experiment 3 — r=32, LR=5e-5, 2 epochs | ~40 min |
| **8** | Experiment 4 — r=64, LR=2e-4, 2 epochs | ~45 min |
| **9** | Cross-experiment comparison (tables + 6 charts) | ~2 min |
| **9i** | Save best model to `./medassist_best_model/` | ~1 min |
| **10** | Base model vs fine-tuned comparison | ~5 min |
| **11** | Gradio chatbot deployment | instant |

> 💡 **Tip:** If your Colab session disconnects mid-training, you can resume from any experiment section independently — each one saves its adapter to `./adapters/ExpN_xxx/` before the next begins.

---

## Gradio UI

The chatbot interface features:
- 🌿 Dark theme with emerald green accents
- ⚡ Quick-start prompt chips for instant exploration
- 📝 Input box at the top, conversation history below
- ⚕️ Automatic educational disclaimer on every response
- 🗑️ Clear button to reset the conversation

**Covered medical domains:**
`Pharmacology` · `Pathology` · `Anatomy` · `Physiology` · `Clinical Diagnosis` · `General Medicine`

---

## Requirements

```
torch>=2.0
transformers>=4.40
peft>=0.10
accelerate>=0.27
bitsandbytes>=0.46.1
gradio==5.50.0
huggingface_hub
evaluate
rouge_score
sacrebleu
pandas
matplotlib
seaborn
numpy
```

**Hardware minimum:** NVIDIA GPU with 4GB VRAM (T4 on Colab free tier works perfectly)

---

## Citation / Acknowledgements

- Base model: [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) by Alibaba Cloud
- Dataset: [medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)
- Fine-tuning method: [LoRA — Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- Quantization: [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) by Tim Dettmers


