from sft.lora import train
from data.psychometry.processed_data.constants import CHURCHILL_CONSTANTS
from sft.constants import CHURCHILL_CONSTANTS as SFT_CHURCHILL_CONSTANTS
from sft.constants import BASE_MODEL_ID

input_data = CHURCHILL_CONSTANTS['sft_data_writename']
output_dir = SFT_CHURCHILL_CONSTANTS['adapter_path']

train(
    base_model=BASE_MODEL_ID,
    training_data=input_data,
    output_dir=output_dir
)