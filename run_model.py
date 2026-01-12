from sft.try_adapter import load_base_model, generate_response
from data.psychometry.processed_data.create_sft_data import create_llama3_text

def run(checkpoint, test_prompts):
    print("="*80)
    print("LOADING MODELS")
    print("="*80)

    base_model, base_tokenizer = load_base_model(checkpoint)

    print("="*80)
    
    # Response for each prompt
    for i, prompt in enumerate(test_prompts, 1):
        print("-" * 80)
        print("🤖 MODEL RESPONSE:")
        print("-" * 80)
        base_response = generate_response(base_model, base_tokenizer, prompt)
        print(base_response)


def run_sample_questions(checkpoint, prompt_preamble, chat_name):
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

    run(checkpoint, test_prompts)