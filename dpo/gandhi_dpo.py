from dpo.dpo import train
from dpo.constants import GANDHI_CONSTANTS as DPO_GANDHI_CONSTANTS
from sft.constants import GANDHI_CONSTANTS as SFT_GANDHI_CONSTANTS
from data.psychometry.processed_data.constants import GANDHI_CONSTANTS

train(
    sft_model=SFT_GANDHI_CONSTANTS['merged_model_path'],
    dpo_training_data=GANDHI_CONSTANTS['dpo_data_writename'],
    output_dir=DPO_GANDHI_CONSTANTS['adapter_path']
)