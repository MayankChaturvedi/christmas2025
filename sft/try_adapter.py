from numpy import dtype
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from data.psychometry.processed_data.constants import GANDHI_CONSTANTS
from data.psychometry.processed_data.constants import CHURCHILL_CONSTANTS
from data.psychometry.processed_data.create_sft_data import create_llama3_text
from sft.constants import GANDHI_CONSTANTS as SFT_GANDHI_CONSTANTS
from sft.constants import CHURCHILL_CONSTANTS as SFT_CHURCHILL_CONSTANTS
from sft.constants import BASE_MODEL_ID


def load_base_model(model_id):
    """Load the base model without LoRA"""
    print(f"Loading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer

def load_finetuned_model(base_model_id, lora_dir):
    """Load the base model with LoRA adapter"""
    print(f"Loading fine-tuned model from: {lora_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model = model.merge_and_unload()  # Optional: merge for faster inference
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=200):
    """Generate a response from the model"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return full_response


def try_adapter(lora_checkpoint_dir, prompt_preamble, chat_name, base_model_id):
    # Test prompts - customize these based on your training data
    test_prompts = [
        "What are your thoughts on non-violence?",
        "How should one approach conflict resolution?",
        "What is the role of truth in leadership?",
    ]

    for i in range(len(test_prompts)):
        test_prompts[i] = create_llama3_text(
            system=prompt_preamble,
            instruction=f"Question: {test_prompts[i]}\n\nAnswer by {chat_name}:",
            response=""
        )['prompt']

    print("="*80)
    print("LOADING MODELS")
    print("="*80)
    
    # Load both models
    base_model, base_tokenizer = load_base_model(base_model_id)
    finetuned_model, ft_tokenizer = load_finetuned_model(base_model_id, lora_checkpoint_dir)
    
    print("\n" + "="*80)
    print("COMPARING RESPONSES")
    print("="*80)
    
    # Compare responses for each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_prompts)}")
        print(f"{'='*80}")
        print(f"\n📝 PROMPT:\n{prompt}\n")
        
        print("-" * 80)
        print("🤖 BASE MODEL RESPONSE:")
        print("-" * 80)
        base_response = generate_response(base_model, base_tokenizer, prompt)
        print(base_response)
        
        print("\n" + "-" * 80)
        print("✨ FINE-TUNED MODEL RESPONSE:")
        print("-" * 80)
        ft_response = generate_response(finetuned_model, ft_tokenizer, prompt)
        print(ft_response)
        print()

if __name__ == "__main__":
    gandhi_preamble = GANDHI_CONSTANTS['prompt_preamble']
    gandhi_chat_name = GANDHI_CONSTANTS['chat_name']
    gandhi_adapter = SFT_GANDHI_CONSTANTS['adapter_path']
    try_adapter(gandhi_adapter, gandhi_preamble, gandhi_chat_name, BASE_MODEL_ID)
    churchill_preamble = CHURCHILL_CONSTANTS['prompt_preamble']
    churchill_chat_name = CHURCHILL_CONSTANTS['chat_name']
    churchill_adapter = SFT_CHURCHILL_CONSTANTS['adapter_path']
    try_adapter(churchill_adapter, churchill_preamble, churchill_chat_name, BASE_MODEL_ID)