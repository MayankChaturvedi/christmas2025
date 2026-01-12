import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
import json
import os

# Hyperparameters
LORA_R = 16          # DPO usually works well with lower R than SFT, but 32/64 is fine too
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

BATCH_SIZE = 2       # DPO uses more memory (2 forward passes), so lower batch size
GRAD_ACCUMULATION = 4
LEARNING_RATE = 5e-6 # DPO needs very low LR (5e-6 to 1e-5)
NUM_EPOCHS = 3
BETA = 0.1           # The KL-divergence coefficient (Standard is 0.1)

# Sequence lengths
MAX_SEQ_LENGTH = 2048
MAX_PROMPT_LENGTH = 1024


def load_dpo_data(file_path):
    """
    Loads JSON data format: 
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Check if data is a list of dicts, if not wrap it
    if isinstance(data, dict): 
        data = [data]
        
    return Dataset.from_list(data)


def train(*, sft_model, dpo_training_data, output_dir):
    torch.cuda.empty_cache()

    print(f"Loading model from {sft_model}...")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(sft_model)
    tokenizer.pad_token = tokenizer.eos_token
    # Llama 3 specific: ensure padding side is left for generation/DPO stability
    tokenizer.padding_side = "right" 

    # 2. Load Model (Full Precision bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        sft_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 3. Setup LoRA
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. Load Dataset
    dataset = load_dpo_data(dpo_training_data)
    print(f"Loaded {len(dataset)} preference pairs.")

    # 5. Initialize DPO Trainer
    
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=1,
        save_strategy="epoch",
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        
        # DPO Specifics
        beta=BETA,
        max_length=MAX_SEQ_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
        dataset_num_proc=4,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None, # Implicitly handled by PEFT
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 6. Train
    print("Starting DPO training...")
    trainer.train()

    # 7. Save
    print(f"Saving DPO adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("✓ DPO Training Complete.")
