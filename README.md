# Iterative DPO Training Pipeline (Data redacted)

This project implements a self-improving training loop to align the Llama-3 model with specific personas (Gandhi/Churchill).



### Python installations
```
pip install -U torch transformers datasets peft trl bitsandbytes accelerate
pip install -U google-generativeai
```

### Phase 1: Cold Start

Before the loop begins, the model must learn the concept of the persona.

* **Data Collection:** Gather a diverse set of Q&A/Chat data for each persona. *Note: `data/raw_data/backup.csv` is redacted and contains only sample data points.*
* **Data Creation:** Use `data/psychometry/processed_data` scripts


* **Base SFT:** Train `Llama-3-Base` on the formatted data.
* *Command:* `python -m sft.gandhi_sft` and `python -m sft.churchill_sft`


* **Result:** `SFT_Model_v0 for Gandhi and Churchill`.

### Loop 1:

* **Generate:** Use `SFT_Model_v0` to answer 500 questions (temperature 0.8), creating 4 candidate answers per question.
* *Reference Script:* `self_improve_models.gemini_reward_loop`


* **Judge:** Gemini selects the Best () and Worst ().
* **Construct Dataset:** Create `dpo_dataset_1.json`.
* **DPO Training:** Train `SFT_Model_v0` using `dpo_dataset_1`.
* *Reference Script:* `dpo.gandhi_dpo and dpo.churchill_dpo`


* **Result:** `DPO_Model_v1`.

### Loop 2:

We discard the old data to force the model to beat its current self.

* **Generate:** Use `DPO_Model_v1` (the new model) to answer the same (or new) questions.

* **Judge:** Gemini picks the Best () and Worst () from this new batch.
* **Construct Dataset:** Create `dpo_dataset_2.json`.
* **DPO Training:** Train `DPO_Model_v1` using `dpo_dataset_2`.
* *Reference Script:* `dpo.gandhi_dpo and dpo.churchill_dpo`


* **Result:** `DPO_Model_v2`.

### Loop 3... N: Convergence

* **Repeat:** I stopped after 2 loops, and plan to automate further loops.
