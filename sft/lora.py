import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer, SFTConfig
import json

# LoRA Hyperparameters
LORA_R = 64
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# Training Hyperparameters
BATCH_SIZE = 4  # Increased since we have full precision
GRAD_ACCUMULATION = 2  # Decreased since batch size is higher
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 1024

# Response template for Llama chat format
RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>"


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Custom data collator that masks prompt tokens and only trains on completions.
    Finds the response template in each sequence and masks everything before it.
    """
    def __init__(self, response_template, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template
        # Encode the response template to get its token IDs
        self.response_template_ids = tokenizer.encode(
            response_template, 
            add_special_tokens=False
        )
        
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        
        # For each example in the batch, find the response template and mask before it
        for i in range(len(batch["labels"])):
            response_token_ids_start_idx = None
            
            # Find where the response template starts
            for idx in range(len(batch["labels"][i]) - len(self.response_template_ids) + 1):
                # Check if the template matches at this position
                if batch["labels"][i][idx:idx + len(self.response_template_ids)].tolist() == self.response_template_ids:
                    response_token_ids_start_idx = idx
                    break
            
            if response_token_ids_start_idx is not None:
                # Mask everything before AND including the response template
                # Training starts right after the template
                batch["labels"][i, :response_token_ids_start_idx + len(self.response_template_ids)] = -100
            else:
                # If template not found, mask the entire sequence (safety measure)
                print(f"Warning: Response template not found in example {i}")
                batch["labels"][i, :] = -100
                
        return batch


def load_data(file_path):
    """
    Loads your local JSON list and converts it to a HF Dataset.
    Expects format: [{"prompt": "...", "completion": "..."}]
    Returns dataset with combined 'text' field.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Combine prompt and completion into a single 'text' field
    formatted_data = []
    for item in data:
        formatted_data.append({
            'text': f"{item['prompt']}{item['completion']}"
        })
    
    dataset = Dataset.from_list(formatted_data)
    return dataset


def train(*, base_model, training_data, output_dir):
    torch.cuda.empty_cache()

    # 1. Load Tokenizer & Model (NO QUANTIZATION)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model in full bfloat16 precision - much cleaner!
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Auto-distribute across your H100s
    )

    # 2. Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    
    # 3. Setup LoRA
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Load and tokenize data
    raw_dataset = load_data(training_data)
    
    def tokenize_function(examples):
        """Tokenize the texts."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,  # Will pad in collator
        )
    
    train_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing dataset"
    )

    # 5. Create Custom Data Collator for Completion-Only Training
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
        mlm=False
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=tokenizer,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUMULATION,
            learning_rate=LEARNING_RATE,
            logging_steps=10,
            num_train_epochs=NUM_EPOCHS,
            save_strategy="epoch",
            optim="adamw_torch",  # Standard optimizer for full precision
            bf16=True,  # Use bfloat16 for training
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="text",
        ),
        peft_config=peft_config,
    )

    # 7. Verify masking is working correctly
    print("\n" + "="*60)
    print("VERIFYING COMPLETION-ONLY MASKING")
    print("="*60)
    
    # Test on first example (already tokenized)
    test_batch = collator([train_dataset[0]])
    
    masked_tokens = (test_batch['labels'][0] == -100).sum().item()
    training_tokens = (test_batch['labels'][0] != -100).sum().item()
    total_tokens = len(test_batch['labels'][0])
    
    print(f"Response template: '{RESPONSE_TEMPLATE}'")
    print(f"Total tokens: {total_tokens}")
    print(f"Masked tokens (prompt): {masked_tokens} ({masked_tokens/total_tokens*100:.1f}%)")
    print(f"Training tokens (completion): {training_tokens} ({training_tokens/total_tokens*100:.1f}%)")
    
    if training_tokens > 0:
        print("✓ Masking is working correctly!")
    else:
        print("⚠ WARNING: No training tokens found! Check your response template.")
    
    print("="*60 + "\n")

    # 8. Train
    print("Starting training...")
    trainer.train()

    # 9. Save
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
