# Fine-Tuning Mistral 7B with LoRA

Fine-tune the Mistral-7B model on a custom dataset using QLoRA (4-bit quantization + LoRA adapters), tracked with Weights & Biases.

## Overview

This notebook fine-tunes `mistralai/Mistral-7B-v0.1` on the `mlabonne/guanaco-llama2-1k` dataset using:

- **QLoRA** — 4-bit quantization via BitsAndBytes to reduce memory usage
- **LoRA** — Low-Rank Adaptation applied to attention layers
- **SFTTrainer** — Supervised fine-tuning via the TRL library
- **W&B** — Training metrics logged to Weights & Biases

## Requirements

```bash
pip install transformers accelerate bitsandbytes peft trl
```

You'll also need accounts and API keys for:
- [Hugging Face](https://huggingface.co/settings/tokens) — to download the Mistral model
- [Weights & Biases](https://wandb.ai) — for experiment tracking

## Setup

Store your API keys as secrets in Google Colab:
- `HUGGING_FACE` — your HF token
- `wandb` — your W&B API key

## Usage

Run all cells in `Fine_tuning.ipynb` in order:

1. Install dependencies
2. Authenticate with HF and W&B
3. Load dataset and base model (4-bit quantized)
4. Configure LoRA and training arguments
5. Train with `SFTTrainer`
6. Save the fine-tuned adapter weights

## Key Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `mistralai/Mistral-7B-v0.1` |
| Dataset | `mlabonne/guanaco-llama2-1k` |
| Quantization | 4-bit NF4 |
| LoRA rank (r) | 64 |
| LoRA alpha | 16 |
| Learning rate | 2e-4 |
| Epochs | 1 |
| Batch size | 4 |

## Output

<img width="284" height="406" alt="Screenshot 2026-03-13 185858" src="https://github.com/user-attachments/assets/538db8ce-7fa9-4183-935d-2579c702628b" />


The fine-tuned LoRA adapter is saved to `./mistral_7b_guanaco`.
