from datasets import load_dataset
import random
from RiddleDataset import RiddleSenseDataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader


RANDOM_SEED = 42
BATCH_SIZE = 64
K = 5
DO_SAMPLE = K != 1
MAX_SEQUENCE_LENGTH = 160


if __name__ == '__main__':
    # Setting the random seed.
    random.seed(RANDOM_SEED)

    print("---- PREPARING DATASET ----\n")
    # TODO. Do I need to do any pre-processing of the dataset?
    original_dataset = load_dataset('riddle_sense')

    print("---- GENERATION ----\n")
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

    model = GPTNeoXForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=REVISION,
        cache_dir=CACHE_DIR
    )

    valid_set = RiddleSenseDataset(original_dataset['validation'], tokenizer, MAX_SEQUENCE_LENGTH, is_generation=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)

    input_ids, answers, labels = next(iter(valid_loader))
    print(input_ids, answers, labels)
    tokens = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, num_return_sequences=K,
                            do_sample=DO_SAMPLE, max_length=MAX_SEQUENCE_LENGTH)
    print(tokenizer.batch_decode(tokens))


