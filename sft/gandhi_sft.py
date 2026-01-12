from email.mime import base
from sft.lora import train
from data.psychometry.processed_data.constants import GANDHI_CONSTANTS
from sft.constants import GANDHI_CONSTANTS as SFT_GANDHI_CONSTANTS
from sft.constants import BASE_MODEL_ID

input_data = GANDHI_CONSTANTS['sft_data_writename']
output_dir = SFT_GANDHI_CONSTANTS['adapter_path']

train(
    base_model=BASE_MODEL_ID,
    training_data=input_data,
    output_dir=output_dir
)