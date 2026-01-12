import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sft.constants import GANDHI_CONSTANTS
from sft.constants import CHURCHILL_CONSTANTS
from sft.constants import BASE_MODEL_ID

def merge_model(adapter_path, merged_model_path, base_model_id):
    print("Loading base model in bfloat16 (CPU/GPU offload)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_model_path}...")
    model.save_pretrained(merged_model_path)
    
    # Save the tokenizer as well so the folder is self-contained
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.save_pretrained(merged_model_path)
    print("✓ Merge complete.")

if __name__ == "__main__":
    gandhi_adapter_path = GANDHI_CONSTANTS["adapter_path"]
    gandhi_merged_path = GANDHI_CONSTANTS["merged_model_path"]
    churchill_adapter_path = CHURCHILL_CONSTANTS["adapter_path"]
    churchill_merged_path = CHURCHILL_CONSTANTS["merged_model_path"]
    merge_model(gandhi_adapter_path, gandhi_merged_path, base_model_id=BASE_MODEL_ID)
    merge_model(churchill_adapter_path, churchill_merged_path, base_model_id=BASE_MODEL_ID)