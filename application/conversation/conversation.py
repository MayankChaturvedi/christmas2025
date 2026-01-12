import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dpo.constants import GANDHI_CONSTANTS as GANDHI_DPO_CONSTANTS
from dpo.constants import CHURCHILL_CONSTANTS as CHURCHILL_DPO_CONSTANTS
from data.psychometry.processed_data.constants import GANDHI_CONSTANTS, CHURCHILL_CONSTANTS
from data.psychometry.processed_data.create_sft_data import create_llama3_text

MAX_NEW_TOKENS = 100

class PersonaBot:
    def __init__(self, data_consts, model_path):
        self.data_consts = data_consts
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    
    def answer(self, question):
        # 1. Create the prompt text
        prompt_data = create_llama3_text(
            system=self.data_consts['prompt_preamble'],
            instruction=self.format_question(question),
            response=""
        )
        prompt = prompt_data["prompt"]
        # 2. Tokenize inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1] # <--- Save the length of the prompt

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.2,
                top_p=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 3. Slice the output tensor to keep ONLY the new tokens
        #    outputs[0] is the sequence. [input_length:] takes everything after the prompt.
        generated_tokens = outputs[0][input_length:]

        # 4. Decode only the generated part
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def generate_chat_prompt(self, chat):
        prompt_data = create_llama3_text(
            system=self.data_consts['prompt_preamble'],
            instruction=self.format_chat(chat),
            response=""
        )
        prompt = prompt_data["prompt"]
        return prompt

    def speak(self, chat, temperature=0.2, top_p=0.3):
        # 1. Create the prompt text
        prompt = self.generate_chat_prompt(chat)

        # 2. Tokenize inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1] # <--- Save the length of the prompt

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 3. Slice the output tensor to keep ONLY the new tokens
        #    outputs[0] is the sequence. [input_length:] takes everything after the prompt.
        generated_tokens = outputs[0][input_length:]

        # 4. Decode only the generated part
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def format_chat(self, chat):
        speaker_name = self.data_consts['chat_name']
        formatted = "Continue the chat:\n"
        for turn in chat:
            formatted += f"{turn['speaker']}: {turn['comment']}\n"
        formatted += f"{speaker_name}:"
        return formatted

    def format_question(self, question):
        speaker_name = self.data_consts['chat_name']
        formatted = f"Question: {question}\n\nAnswer by {speaker_name}:"
        return formatted

def run_debate(first_speaker, first_speaker_comment, max_turns):
    gandhi = PersonaBot(GANDHI_CONSTANTS, GANDHI_DPO_CONSTANTS["merged_model_path"])
    churchill = PersonaBot(CHURCHILL_CONSTANTS, CHURCHILL_DPO_CONSTANTS["merged_model_path"])
    chat = [
        {
            "speaker": first_speaker,
            "comment": first_speaker_comment
        }
    ]
    print(f"\n{first_speaker}: {first_speaker_comment}\n")
    while(len(chat) < max_turns*2):
        if chat[-1]['speaker'] == GANDHI_CONSTANTS['chat_name']:
            words = churchill.speak(chat)
            print(f"\n{CHURCHILL_CONSTANTS['chat_name']}: {words}\n")
            chat.append({
                "speaker": CHURCHILL_CONSTANTS['chat_name'],
                "comment": words
            })
        else:
            words = gandhi.speak(chat)
            print(f"\n{GANDHI_CONSTANTS['chat_name']}: {words}\n")
            chat.append({
                "speaker": GANDHI_CONSTANTS['chat_name'],
                "comment": words
            })
    return chat

if __name__ == "__main__":
    run_debate("Gandhi", "India in 2024 has risen above Britain in terms of GDP, Mr Churchill. What comment do you have?", max_turns=5)