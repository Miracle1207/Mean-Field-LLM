# MF-LLM: Simulating Population Decision Dynamics via a Mean-Field Large Language Model Framework

This repository implements **MF‑LLM**, a simulation framework for modeling large-scale collective behavior using **Mean-Field Large Language Models**. It supports trajectory generation for heterogeneous agents under diverse scenarios, enabling the study of population decision dynamics and policy interventions.

The framework also supports **GPT-based evaluation** of generated behaviors and integrates with our fine-tuning toolkit **[IB-Tune](IB-Tune/IB-Tune for Mean-Field LLMs.md)** for efficient LoRA-based model adaptation.

---

## 🛠️ Installation

To install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚦 Running Simulations with MF‑LLM

### Step 1: Specify Model Names and Algorithm

In `scripts/run_mf_batch.py`, configure model names according to the chosen algorithm (`--alg`):

```python
if ALG == 'mf':
    a_MODEL_NAME = "your-policy-model-name"
    mf_MODEL_NAME = "your-mean-field-model-name"
elif ALG == 'state':
    a_MODEL_NAME = "Qwen2-1.5B-Instruct"
    mf_MODEL_NAME = "Qwen2-1.5B-Instruct"
# Other variants omitted for brevity
```

Set `args.a_MODEL_NAME` and `args.mf_MODEL_NAME` accordingly.

---

### Step 2: Launch a Simulation

Run the MF‑LLM simulation and generate agent-level decision sequences:

```bash
cd mf_llm
python scripts/run_mf_batch.py \
  --alg "mf" \
  --comment_n 300 \
  --simulation_start 50 \
  --file_name "retirement" \
  --model "1.5B" \
  --task ""
```

| Argument             | Description                                                          |
| -------------------- | -------------------------------------------------------------------- |
| `--alg`              | Algorithm type (see table below)                                     |
| `--comment_n`        | Number of agents to simulate                                         |
| `--simulation_start` | Warm-up horizon $t_{\text{warmup}}$, i.e., when simulation begins    |
| `--file_name`        | Scenario/event name (e.g., `"retirement"`)                           |
| `--model`            | Backbone model: `"1.5B"`, `"7B"`, `"gpt-4o-mini"`, `"DeepSeek"` etc. |
| `--task`             | Optional: path to save generated sequences                           |

---

### Step 3: Batch Simulation (Optional)

To simulate multiple scenarios or models in batch:

```bash
bash scripts/run_simulation.sh
```

---





## 📊 GPT-Based Evaluation (Optional)

MF‑LLM supports automatic evaluation of agent trajectories via GPT models.

### How to Evaluate

1. Set your OpenAI API key:

   ```bash
   export OPENAI_API_KEY=your-key-here
   ```

2. Ensure simulation results are registered in:

   ```python
   eval_dir_path = "../scripts/save_data/main/saved_file_paths.json"
   ```

   Files already evaluated (ending in `_eval.csv`) will be automatically skipped.

3. Run the evaluation script:

   ```bash
   cd mf_llm
   python evaluate/evaluate_gpt_batch.py
   ```

---

## 🧪 Supported Algorithms

### ✅ Baselines

| Algorithm   | Description                                                                          |
| ----------- | ------------------------------------------------------------------------------------ |
| `state`     | **State-only:** Uses user profile and event topic only (cf. *ElectionSim*)           |
| `pre`       | **Recent:** Includes most recent $k$ actions (cf. *TrendSim*, *AgentSociety*)        |
| `hot`       | **Popular:** Includes top-$k$ actions with highest popularity (cf. *OASIS*, *HiSim*) |
| `state_sft` | **Supervised Fine-Tuning (SFT):** Uses state–action pairs, no mean field             |

### 🔍 MF-LLM Variants and Ablations

| Algorithm        | Description                                                               |
| ---------------- | ------------------------------------------------------------------------- |
| `mf`             | **MF-LLM (Ours):** Full model with both modules fine-tuned via IB-Tune    |
| `mf_wo_sft_mf`   | **w/o IB-Tune MF:** Pretrained mean-field model, fine-tuned policy module |
| `mf_wo_sfta`     | **w/o IB-Tune Policy:** Fine-tuned mean-field model, pretrained policy    |
| `mf_wo_sft_mf_a` | **w/o IB-Tune (Full Pretrained):** No fine-tuning for either component    |

For fine-tuning details and training scripts, refer to:
👉 [IB-Tune for MF-LLMs →](IB-Tune/IB-Tune for Mean-Field LLMs.md)

---

## 📁 Code Structure

```text
.
├── data/
│   └── rumdect/                        # Weibo-based simulation dataset
├── evaluate/
│   └── evaluate_gpt_batch.py          # GPT-based evaluation for decision trajectories
├── mean_field_utils/
│   ├── loss.py                         # Mean-field loss formulation
│   └── update_prompt.py                # Prompt generation and update logic
├── scripts/
│   ├── run_mf_batch.py                # Core script to simulate MF-LLM
│   └── run_simulation.sh             # Batch run utility across configs
├── readme.md                         # Main documentation (you are here)
```

---

## 🧩 See Also

* 🔧 [IB-Tune: Fine-Tuning Framework for MF-LLMs →](IB-Tune/IB-Tune for Mean-Field LLMs.md)
  A modular, LoRA-based fine-tuning framework for both the mean-field and policy modules in MF-LLM.

---

