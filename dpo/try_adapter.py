from data.psychometry.processed_data.constants import GANDHI_CONSTANTS
from dpo.constants import GANDHI_CONSTANTS as DPO_GANDHI_CONSTANTS
from sft.constants import GANDHI_CONSTANTS as SFT_GANDHI_CONSTANTS
from data.psychometry.processed_data.constants import CHURCHILL_CONSTANTS
from dpo.constants import CHURCHILL_CONSTANTS as DPO_CHURCHILL_CONSTANTS
from sft.constants import CHURCHILL_CONSTANTS as SFT_CHURCHILL_CONSTANTS
from sft.try_adapter import try_adapter


if __name__ == "__main__":
    gandhi_preamble = GANDHI_CONSTANTS['prompt_preamble']
    gandhi_chat_name = GANDHI_CONSTANTS['chat_name']
    gandhi_adapter = DPO_GANDHI_CONSTANTS['adapter_path']
    gandhi_base_model = SFT_GANDHI_CONSTANTS['merged_model_path']
    try_adapter(gandhi_adapter, gandhi_preamble, gandhi_chat_name, gandhi_base_model)
    churchill_preamble = CHURCHILL_CONSTANTS['prompt_preamble']
    churchill_chat_name = CHURCHILL_CONSTANTS['chat_name']
    churchill_adapter = DPO_CHURCHILL_CONSTANTS['adapter_path']
    churchill_base_model = SFT_CHURCHILL_CONSTANTS['merged_model_path']
    try_adapter(churchill_adapter, churchill_preamble, churchill_chat_name, churchill_base_model)