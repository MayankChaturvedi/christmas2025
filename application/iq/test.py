from application.conversation.conversation import PersonaBot
from dpo.constants import GANDHI_CONSTANTS as GANDHI_DPO_CONSTANTS
from dpo.constants import CHURCHILL_CONSTANTS as CHURCHILL_DPO_CONSTANTS
from data.psychometry.processed_data.constants import GANDHI_CONSTANTS, CHURCHILL_CONSTANTS
from sft.constants import BASE_MODEL_ID

questions = [
    "You are running a race and you pass the person in second place. What place are you in now?",
    "How many \"r\"s are in the word \"strawberry\"?",
    "If it takes 1 hour to dry a towel outside in the sun, how long does it take to dry 10 towels?",
    "When I was 6 years old, my sister was half my age. I am now 70. How old is my sister?",
    "Which is heavier: one pound of feathers or one kilogram of steel?",
    "A farmer needs to cross a river with a wolf, a goat, and a cabbage. His boat is massive and can fit the farmer, the wolf, the goat, and the cabbage all at once. How many trips does he need to take?",
    "Kevin currently has 10 apples. He ate 2 apples yesterday. How many apples does Kevin have now?",
    "Which number is larger: 9.11 or 9.9?",
    "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?",
    "David's father has three sons: Snap, Crackle, and _____?"
]

if __name__ == "__main__":
    gandhi = PersonaBot(GANDHI_CONSTANTS, GANDHI_DPO_CONSTANTS["merged_model_path"])
    churchill = PersonaBot(CHURCHILL_CONSTANTS, CHURCHILL_DPO_CONSTANTS["merged_model_path"])
    llama = PersonaBot({
            "full_name": "Einstein",
            "chat_name": "Einstein",
            "prompt_preamble": "You are Albert Einstein. You speak with scientific authority, curiosity, and a touch of humor. You value intellectual exploration, creativity, and the pursuit of knowledge. You often use thought experiments and analogies to explain complex ideas.",
            }, BASE_MODEL_ID)

    print("\n" + "="*80)
    print("GANDHI ANSWERS")
    print("="*80)
    for question in questions:
        answer = gandhi.answer(question)
        print(f"\nQ: {question}\nA: {answer}\n")

    print("\n" + "="*80)
    print("CHURCHILL ANSWERS")
    print("="*80)
    for question in questions:
        answer = churchill.answer(question)
        print(f"\nQ: {question}\nA: {answer}\n")
    
    print("\n" + "="*80)
    print("LLAMA ANSWERS")
    print("="*80)
    for question in questions:
        answer = llama.answer(question)
        print(f"\nQ: {question}\nA: {answer}\n")