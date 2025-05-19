
# IB-Tune for Mean-Field LLMs

This repository provides an implementation of **IB-Tune**, a lightweight fine-tuning framework designed for **Mean-Field LLMs** (MF-LLMs) in large-scale agent-based simulations. Built on top of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), IB-Tune enables LoRA-based adaptation for both **mean-field modeling** and **policy optimization**, offering a modular and efficient training pipeline.

---

## 🚀 Quick Start

### Installation

Please follow the [OpenRLHF installation instructions](https://github.com/OpenRLHF/OpenRLHF#installation) to set up the environment and dependencies.

---

## 🧠 Step 1: Fine-Tune the Mean-Field Model

1. **Configure the initial LLM** in
   `openrlhf/trainer/mean_field_utils/mean_field_loss.py`:

   ```python
   MODEL_PATH = "XXX"  # Path to base LLM
   MODEL_NAME = "Qwen2-1.5B-Instruct"  # e.g., Qwen2-1.5B-Instruct
   model_name = MODEL_PATH + MODEL_NAME
   
   initial_model = AutoModelForCausalLM.from_pretrained(
       model_name,
       torch_dtype=torch.float16
   )
   ```

2. **Specify dataset, pre-trained model, and output paths** in
   `examples/scripts/train_mf_sft_qwen_lora.sh`:

   ```bash
   openrlhf.cli.train_mf \
       --alg mf \
       --dataset /your/path/to/data/rumdect/Weibo/train \
       --pretrain /your/path/to/Qwen2-1.5B-Instruct \
       --save_path /your/path/to/output/Qwen2-1.5B-mf \
       --ckpt_path /your/path/to/checkpoints/
       # (Other hyperparameters omitted)
   ```
   
3. **Launch mean-field fine-tuning** via:

   ```bash
   bash examples/scripts/train_mf_sft_qwen_lora.sh
   ```

---

## 🤖 Step 2: Fine-Tune the Policy Model with IB-Tune

1. **Load the trained mean-field model** in
   `openrlhf/trainer/mf_sft_trainer.py`:

   ```python
   elif self.args.have_mf_model and self.args.alg == "policy_on_mf":
       MODEL_PATH = "XXX"  # Path to fine-tuned MF model
       MODEL_NAME = "Qwen2-1.5B-mf"
       model_name = MODEL_PATH + MODEL_NAME
   
       self.mf_model = AutoModelForCausalLM.from_pretrained(
           model_name,
           torch_dtype=torch.float16
       )
       print(f"mf_model: {MODEL_NAME}")
   ```

2. **Set training paths** in
   `examples/scripts/train_policy_sft_qwen_lora.sh` as you did in step 1.

3. **Start IB-Tune policy fine-tuning**:

   ```bash
   bash examples/scripts/train_policy_sft_qwen_lora.sh
   ```

---

## 🔁 Baseline: Standard SFT (No Mean Field)

To reproduce a supervised fine-tuning baseline without the mean-field module:

```bash
bash examples/scripts/train_state_sft_qwen_lora.sh
```

---



## 🧩 Code Structure

```text
IB-Tune/openrlhf/
├── cli/
│   └── train_mf.py                       # CLI entry point for mean-field training
│
├── trainer/
│   ├── mf_sft_trainer.py                # IB-Tune trainer: policy model on mean field
│   ├── mean_field_utils/
│   │   └── mean_field_loss.py           # Mean-field loss function & model setup
│   └── __init__.py
│
├── utils/
│   └── mean_field_utils.py              # Utilities and dataset interface for mean-field training
│
examples/
└── scripts/
    ├── train_mf_sft_qwen_lora.sh        # Script to fine-tune the mean-field model
    ├── train_policy_sft_qwen_lora.sh    # Script to fine-tune the policy model with IB-Tune
    └── train_state_sft_qwen_lora.sh     # Script for standard supervised fine-tuning (baseline)
IB-Tune for Mean-Field LLMs.md                                # Project documentation (you are here)
```



---

## 📌 Notes

* IB-Tune uses [LoRA](https://arxiv.org/abs/2106.09685) for memory- and compute-efficient parameter adaptation.
* Compatible with any HuggingFace `AutoModelForCausalLM` model (e.g., Qwen, LLaMA, GPT).
* Training data should follow the multi-agent interaction format defined in `mean_field_utils.py`.

---

