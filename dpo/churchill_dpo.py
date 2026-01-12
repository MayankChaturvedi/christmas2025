from dpo.dpo import train
from dpo.constants import CHURCHILL_CONSTANTS as DPO_CHURCHILL_CONSTANTS
from sft.constants import CHURCHILL_CONSTANTS as SFT_CHURCHILL_CONSTANTS
from data.psychometry.processed_data.constants import CHURCHILL_CONSTANTS

train(
    sft_model=SFT_CHURCHILL_CONSTANTS['merged_model_path'],
    dpo_training_data=CHURCHILL_CONSTANTS['dpo_data_writename'],
    output_dir=DPO_CHURCHILL_CONSTANTS['adapter_path']
)