import re
from datasets import load_dataset
import random
import torch
import openai
import sys
from RiddleDataset import RiddleSenseDataset
from utils import write_result_to_file, get_first_new_token

# Fields
RANDOM_SEED = 42
DO_SAMPLE = True
MAX_SEQUENCE_LENGTH = 160
K = 2
BATCH_SIZE = 10


def get_exact_match_for_generation(sample, model):
    sample_set = RiddleSenseDataset(sample, MAX_SEQUENCE_LENGTH,
                                    is_generation=True, is_exact_match=True)

    dataset = []
    for i in range(0, len(sample_set), 5):
        _, answer, label, prompt = sample_set[i]

        count = 1
        while not label:
            _, answer, label, prompt = sample_set[i + count]
            count += 1
        dataset.append((prompt, answer))

    total_score = 0
    processed = 0
    for entry in dataset:
        processed += 1
        if processed % 50 == 0:
            print(f'Processed {processed}/{len(dataset)}.')
        # Generate a response
        completion = openai.Completion.create(
            engine=model,
            prompt=entry[0],
            max_tokens=MAX_SEQUENCE_LENGTH + 10,
            n=K,
            temperature=0.5,
            stop=None,
        )

        number_of_matches = 0
        for choice in completion.choices:
            output = choice.text
            token, token_index = get_first_new_token(output)
            print(f"PROMPT: {entry[0]}")
            print(f"ANSWER: {entry[1]}")

            if token is not None:
                token = re.sub(r'[^\w\s]', '', token)
                print(f"TOKEN: {token}")
                answer_split = entry[1].split(" ")

                idx = 0
                while idx < len(answer_split):
                    if token.lower() == answer_split[idx]:
                        idx += 1
                        token_index += 1
                        if token_index < len(output):
                            token = output[token_index]
                            token = re.sub(r'[^\w\s]', '', token)
                        else:
                            break
                    else:
                        break

                if idx == len(answer_split):
                    number_of_matches += 1
                    print(f"MATCH!!!")

        total_score += number_of_matches / len(completion.choices)

    result = total_score / len(dataset)
    print(f"Exact match accuracy for generation: {result}")
    return result


def get_exact_match_for_discrimination(sample, model):
    sample_set = RiddleSenseDataset(sample, MAX_SEQUENCE_LENGTH,
                                    is_generation=False, is_exact_match=is_exact_match)
    total_score = 0
    processed = 0
    for _, answer, label, prompt in sample_set:
        processed += 1
        if processed % 50 == 0:
            print(f'Processed {processed}/{len(sample_set)}.')

        # Generate a response
        completion = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=MAX_SEQUENCE_LENGTH + 10,
            n=K,
            temperature=0.5,
            stop=None,
        )

        number_of_matches = 0
        for choice in completion.choices:
            output = choice.text
            token, token_index = get_first_new_token(output)
            print(f"PROMPT: {prompt}")
            print(f"ANSWER: {answer}")
            print(f"LABEL: {label}")
            print(f"OUTPUT: {output}")

            if token is not None:
                print(f"TOKEN: {token}")
                token = re.sub(r'[^\w\s]', '', token)
                if (token.lower() == 'yes' and label) or \
                        (token.lower() == 'no' and not label):
                    number_of_matches += 1
                    print(f"MATCH!!!")

        total_score += number_of_matches / len(completion.choices)

    result = total_score / len(sample_set)
    print(f"Exact match accuracy for discrimination: {total_score / len(sample_set)}")
    return result


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using: {device}\n")

    # Setting the random seed.
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("\n---- FETCHING MODEL ----")
    # Define OpenAI API key
    openai.api_key = sys.argv[1]

    # Define model.
    MODEL = "text-davinci-003"

    print("\n---- PREPARING DATASET ----")
    original_dataset = load_dataset('riddle_sense')
    data = original_dataset["validation"]

    # Define setting.
    is_exact_match = True
    is_generation = True

    print("\n---- GENERATION ----")
    em_gen = get_exact_match_for_generation(data, MODEL)
    write_result_to_file(MODEL, is_generation, is_exact_match, K, em_gen)

    print("\n---- DISCRIMINATION ----")
    em_dis = get_exact_match_for_discrimination(data, MODEL)
    write_result_to_file(MODEL, not is_generation, is_exact_match, K, em_dis)
