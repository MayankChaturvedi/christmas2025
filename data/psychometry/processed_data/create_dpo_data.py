import json

def create_llama3_dpo_json(*, system, instruction, chosen, rejected):
    """
    Constructs the Llama 3.2 standard chat format.
    """
    # 1. System Message
    text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    
    # 2. User Message
    text += f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # 3. Assistant Message (Target)
    chosen = f"{chosen}<|eot_id|>"
    rejected = f"{rejected}<|eot_id|>"

    return {"prompt": text, "chosen": chosen, "rejected": rejected}

def create_dpo_data_llama(*, raw_data, full_name, chat_name, debator_name, debator_full_name, prompt_preamble, dump_file_name):
    processed_data = []
    for data in raw_data:
        question = data['question']
        chosen = data[full_name]
        rejected = data[debator_full_name]
        processed_data.append(
            create_llama3_dpo_json(
                system=prompt_preamble,
                instruction=f"Question: {question}\n\nAnswer by {chat_name}:",
                chosen=chosen,
                rejected=rejected
            )
        )
        processed_data.append(
            create_llama3_dpo_json(
                system=prompt_preamble,
                instruction=f"Continue the chat:\n{debator_name}: {question}\n{chat_name}:",
                chosen=chosen,
                rejected=rejected
            )
        )

    print(f"Total processed entries: {len(processed_data)}")
    with open(dump_file_name, 'w') as f:
        json.dump(processed_data, f)

def create_dpo_data_gemma(*, raw_data, full_name, chat_name, debator_name, debator_full_name, prompt_preamble, dump_file_name):
    processed_data = []
    for data in raw_data:
        question = data['question']
        chosen = data[full_name]
        rejected = data[debator_full_name]
        processed_data.append(
            {
                "prompt": f"<start_of_turn>user\n{prompt_preamble}\n\nQuestion: {question}\n\nAnswer by {chat_name}:<end_of_turn>\n<start_of_turn>model\n",
                "chosen": f"{chosen}<end_of_turn>",
                "rejected": f"{rejected}<end_of_turn>"
            }
        )
        processed_data.append(
            {
                "prompt": f"<start_of_turn>user\n{prompt_preamble}\n\nChat so far:\n{debator_name}: {question}\n{chat_name}:<end_of_turn>\n<start_of_turn>model\n",
                "chosen": f"{chosen}<end_of_turn>",
                "rejected": f"{rejected}<end_of_turn>"
            }
        )

    print(f"Total processed entries: {len(processed_data)}")
    with open(dump_file_name, 'w') as f:
        json.dump(processed_data, f)