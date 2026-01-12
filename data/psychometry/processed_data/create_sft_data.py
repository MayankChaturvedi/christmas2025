import json


def create_llama3_text(*, system, instruction, response):
    """
    Constructs the Llama 3.2 standard chat format.
    """
    # 1. System Message
    text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    
    # 2. User Message
    text += f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # 3. Assistant Message (Target)
    answer = f"{response}<|eot_id|>"

    return {"prompt": text, "completion": answer}


def create_sft_data_llama(*, raw_data, full_name, chat_name, debator_name, prompt_preamble, dump_file_name):
    processed_data = []
    for data in raw_data:
        question = data['question']
        answer = data[full_name]
        processed_data.append(
            create_llama3_text(
                system=prompt_preamble,
                instruction=f"Question: {question}\n\nAnswer by {chat_name}:",
                response=answer
            )
        )
        processed_data.append(
            create_llama3_text(
                system=prompt_preamble,
                instruction=f"Continue the chat:\n{debator_name}: {question}\n{chat_name}:",
                response=answer
            )
        )
        chat_script = ""
        for chat in data['chat']:
            speaker = chat['speaker']
            comment = chat['comment']
            if speaker == chat_name:
                processed_data.append(
                    create_llama3_text(
                        system=prompt_preamble,
                        instruction=f"Continue the chat:\n{chat_script}\n{chat_name}:",
                        response=answer
                    )
                )
            chat_script += f"{speaker}: {comment}\n"


    print(f"Total processed entries: {len(processed_data)}")
    with open(dump_file_name, 'w') as f:
        json.dump(processed_data, f)


def create_sft_data_gemma(*, raw_data, full_name, chat_name, debator_name, prompt_preamble, dump_file_name):
    processed_data = []
    for data in raw_data:
        question = data['question']
        answer = data[full_name]
        processed_data.append(
            {
                "prompt": f"<start_of_turn>user\n{prompt_preamble}\n\nQuestion: {question}\n\nAnswer by {chat_name}:<end_of_turn>\n<start_of_turn>model\n",
                "completion": f"{answer}<end_of_turn>"
            }
        )
        processed_data.append(
            {
                "prompt": f"<start_of_turn>user\n{prompt_preamble}\n\nChat so far:\n{debator_name}: {question}\n{chat_name}:<end_of_turn>\n<start_of_turn>model\n",
                "completion": f"{answer}<end_of_turn>"
            }
        )
        chat_script = ""
        for chat in data['chat']:
            speaker = chat['speaker']
            comment = chat['comment']
            if speaker == chat_name:
                processed_data.append(
                    {
                        "prompt": f"<start_of_turn>user\n{prompt_preamble}\n\nChat so far:\n{chat_script}\n{chat_name}:<end_of_turn>\n<start_of_turn>model\n",
                        "completion": f"{comment}<end_of_turn>"
                    }
                )
            chat_script += f"{speaker}: {comment}\n"


    print(f"Total processed entries: {len(processed_data)}")
    with open(dump_file_name, 'w') as f:
        json.dump(processed_data, f)