from dpo.constants import GANDHI_CONSTANTS as DPO_GANDHI_CONSTANTS
from sft.constants import GANDHI_CONSTANTS as SFT_GANDHI_CONSTANTS
from dpo.constants import CHURCHILL_CONSTANTS as DPO_CHURCHILL_CONSTANTS
from sft.constants import CHURCHILL_CONSTANTS as SFT_CHURCHILL_CONSTANTS
from sft.merge_adapter_and_checkpoint import merge_model

if __name__ == "__main__":
    gandhi_adapter_path = DPO_GANDHI_CONSTANTS["adapter_path"]
    gandhi_merged_path = DPO_GANDHI_CONSTANTS["merged_model_path"]
    gandhi_base_model = SFT_GANDHI_CONSTANTS["merged_model_path"]
    churchill_adapter_path = DPO_CHURCHILL_CONSTANTS["adapter_path"]
    churchill_merged_path = DPO_CHURCHILL_CONSTANTS["merged_model_path"]
    churchill_base_model = SFT_CHURCHILL_CONSTANTS["merged_model_path"]
    merge_model(gandhi_adapter_path, gandhi_merged_path, base_model_id=gandhi_base_model)
    merge_model(churchill_adapter_path, churchill_merged_path, base_model_id=churchill_base_model)