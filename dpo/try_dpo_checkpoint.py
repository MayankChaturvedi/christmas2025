from data.psychometry.processed_data.constants import GANDHI_CONSTANTS
from data.psychometry.processed_data.constants import CHURCHILL_CONSTANTS
from dpo.constants import GANDHI_CONSTANTS as DPO_GANDHI_CONSTANTS
from dpo.constants import CHURCHILL_CONSTANTS as DPO_CHURCHILL_CONSTANTS
from run_model import run_sample_questions

if __name__ == "__main__":
    gandhi_preamble = GANDHI_CONSTANTS['prompt_preamble']
    gandhi_chat_name = GANDHI_CONSTANTS['chat_name']
    gandhi_sft_checkpoint = DPO_GANDHI_CONSTANTS['merged_model_path']
    run_sample_questions(gandhi_sft_checkpoint, gandhi_preamble, gandhi_chat_name)
    churchill_preamble = CHURCHILL_CONSTANTS['prompt_preamble']
    churchill_chat_name = CHURCHILL_CONSTANTS['chat_name']
    churchill_sft_checkpoint = DPO_CHURCHILL_CONSTANTS['merged_model_path']
    run_sample_questions(churchill_sft_checkpoint, churchill_preamble, churchill_chat_name)