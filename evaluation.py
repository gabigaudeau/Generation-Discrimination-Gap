from datasets import load_dataset
import random
from RiddleDataset import RiddleSenseDataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import time
import torch


RANDOM_SEED = 42
BATCH_SIZE = 8
K = 5
DO_SAMPLE = K != 1
MAX_SEQUENCE_LENGTH = 160


if __name__ == '__main__':
    # Setting the random seed.
    random.seed(RANDOM_SEED)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("---- PREPARING DATASET ----\n")
    # TODO. Do I need to do any pre-processing of the dataset?
    original_dataset = load_dataset('riddle_sense')

    print("\n---- GENERATION ----\n")
    SIZE = '70m'
    MODEL_NAME = f'EleutherAI/pythia-{SIZE}-deduped'
    REVISION = 'step3000'
    CACHE_DIR = f'./pythia-{SIZE}-deduped/step3000'

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=REVISION,
        cache_dir=CACHE_DIR,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = GPTNeoXForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=REVISION,
        cache_dir=CACHE_DIR
    )
    model.to(device)

    valid_set = RiddleSenseDataset(original_dataset['validation'], tokenizer, MAX_SEQUENCE_LENGTH, is_generation=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)

    input_ids, answers, labels = next(iter(valid_loader))
    input_ids = input_ids.squeeze(1).to(device)

    st = time.time()
    tokens = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, num_return_sequences=K,
                            do_sample=DO_SAMPLE, max_new_tokens=MAX_SEQUENCE_LENGTH + 20)
    et = time.time()
    print(f"Generate with K sampling took {et - st}s.")
    print(tokenizer.batch_decode(tokens, skip_special_tokens=True))
