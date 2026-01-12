from data.psychometry.raw_data.load_backup import restore_data
from data.psychometry.processed_data.create_sft_data import create_sft_data_llama as create_sft_data
from data.psychometry.processed_data.create_dpo_data import create_dpo_data_llama as create_dpo_data
from data.psychometry.processed_data.constants import GANDHI_CONSTANTS
from data.psychometry.processed_data.constants import CHURCHILL_CONSTANTS

def build_training_data(constants):
    filename = constants['backup_filename']
    sft_writename = constants['sft_data_writename']
    dpo_writename = constants['dpo_data_writename']
    full_name = constants['full_name']
    chat_name = constants['chat_name']
    debator_name = constants['debator_name']
    debator_full_name = constants['debator_full_name']
    prompt_preamble = constants['prompt_preamble']

    raw_data = restore_data(filename)

    create_sft_data(
        raw_data=raw_data,
        full_name=full_name,
        chat_name=chat_name,
        debator_name=debator_name,
        prompt_preamble=prompt_preamble,
        dump_file_name=sft_writename)

    create_dpo_data(
        raw_data=raw_data,
        full_name=full_name,
        chat_name=chat_name,
        debator_full_name=debator_full_name,
        debator_name=debator_name,
        prompt_preamble=prompt_preamble,
        dump_file_name=dpo_writename)


if __name__ == "__main__":
    build_training_data(GANDHI_CONSTANTS)
    build_training_data(CHURCHILL_CONSTANTS)