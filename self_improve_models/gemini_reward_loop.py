from application.conversation.conversation import PersonaBot
from data.psychometry.processed_data.constants import GANDHI_CONSTANTS, CHURCHILL_CONSTANTS
from dpo.constants import GANDHI_CONSTANTS as DPO_GANDHI_CONSTANTS
from dpo.constants import CHURCHILL_CONSTANTS as DPO_CHURCHILL_CONSTANTS
from data.psychometry.raw_data.load_backup import restore_data
import random
import json
import logging
from google import genai
from google.genai import types

DATA_QUANTITY = 5 # Expect to generate ANSWERS_TO_GENERATExDATA_QUANTITY samples
ANSWERS_TO_GENERATE = 2
self_improve_gandhi_dpo_filepath = "self_improve_models/self_improve_gandhi_dpo.json"
self_improve_churchill_dpo_filepath = "self_improve_models/self_improve_churchill_dpo.json"
# The client gets the API key from the environment variable `GEMINI_API_KEY`.
gemini_client = genai.Client()
JUDGE_MODEL_ID = "gemini-2.5-flash"

def generate_n_answers(bot, chat, n=3):
    answers = []
    for _ in range(n):
        answer = bot.speak(chat, temperature=0.7, top_p=0.9)
        answers.append(answer)
    return answers

def select_best_answer(chat, answers, persona_name):
    prompt = f"""
    You are an expert dialogue judge.
    Context: A debate involving {persona_name}.
    Chat History: {json.dumps(chat, indent=2)}
    
    Candidate Options for next response:
    {json.dumps(answers, indent=2)}
    
    Task: Select the response that is most character-accurate, engaging, and logically consistent.
    Return the index (0-based) of the best option.
    """

    try:
        response = gemini_client.models.generate_content(
            model=JUDGE_MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "best_answer_index": {"type": "integer"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["best_answer_index"]
                }
            )
        )
        
        result = json.loads(response.text)
        best_idx = result.get("best_answer_index", 0)
        
        # Safety check to ensure index is within bounds
        if 0 <= best_idx < len(answers):
            return best_idx
        return 0 # Fallback
        
    except Exception as e:
        logging.error(f"Error in select_best_answer: {e}")
        return 0 # Fallback to first answer on error

def get_list_of_questions():
    filename = GANDHI_CONSTANTS['backup_filename']
    raw_data = restore_data(filename)
    return [data_point['question'] for data_point in raw_data]

def create_dpo_pairs(*, bot, chat, answers, best_answer_index):
    dpo_pairs = []
    prompt = bot.generate_chat_prompt(chat)
    right_answer = f'{answers[best_answer_index]}<|eot_id|>'
    wrong_answers = [f'{answers[i]}<|eot_id|>' for i in range(len(answers)) if i != best_answer_index]
    for wrong_answer in wrong_answers:
        dpo_pairs.append({
            "prompt": prompt,
            "chosen": right_answer,
            "rejected": wrong_answer
        })
    return dpo_pairs

if __name__ == "__main__":
    generated_data = [[], []]
    bots = [
        PersonaBot(GANDHI_CONSTANTS, DPO_GANDHI_CONSTANTS["merged_model_path"]), 
        PersonaBot(CHURCHILL_CONSTANTS, DPO_CHURCHILL_CONSTANTS["merged_model_path"])
    ]

    list_of_questions = get_list_of_questions()

    for i in range(DATA_QUANTITY):
        random_question = random.choice(list_of_questions)
        starter_index = random.randint(0, 1)

        chat = [
            {
                "speaker": bots[starter_index].data_consts['chat_name'],
                "comment": random_question
            }
        ]
        debator_index = starter_index

        for j in range(5):
            # Switch speaker for next turn
            debator_index = (debator_index + 1) % 2
            debator_bot = bots[debator_index]
            persona_name = bots[debator_index].data_consts['chat_name']
            persona_full_name = debator_bot.data_consts['full_name']

            answers = generate_n_answers(debator_bot, chat, n=ANSWERS_TO_GENERATE)

            best_answer_index = select_best_answer(chat, answers, persona_full_name)

            generated_data[debator_index].extend(create_dpo_pairs(
                bot=debator_bot,
                chat=chat,
                answers=answers,
                best_answer_index=best_answer_index
            ))
            chat.append(
                {
                    "speaker": persona_name,
                    "comment": random_question if j == 0 else bots[debator_index].speak(chat, temperature=0.7, top_p=0.9)
                }
            )
    print(generated_data)
    with open(self_improve_gandhi_dpo_filepath, 'w') as f:
        json.dump(generated_data[0], f)
    with open(self_improve_churchill_dpo_filepath, 'w') as f:
        json.dump(generated_data[1], f)

