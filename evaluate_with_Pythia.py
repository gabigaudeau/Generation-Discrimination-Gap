from datasets import load_dataset
import random
from transformers import GPTNeoXForCausalLM, AutoTokenizer, set_seed
from torch.utils.data import DataLoader
import torch
import sys
from RiddleDataset import RiddleSenseDataset
from utils import *


def get_exact_match_for_generation_with_batching(loader):
    processed_batches = 0
    total_score = 0
    for input_ids, answers, labels, prompts in iter(loader):
        processed_batches += 1
        if processed_batches % 50 == 0:
            print(f'Processed {processed_batches}/{number_of_batches} batches.')

        input_ids = input_ids.squeeze(1).to(device)

        with torch.inference_mode():
            outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, num_return_sequences=K,
                                     do_sample=DO_SAMPLE, max_new_tokens=MAX_SEQUENCE_LENGTH + 10,
                                     use_cache=True, temperature=0.5)
        tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        index = 0
        for sample in range(len(answers)):
            sample_score = 0
            for k in range(K):
                token, token_index = get_first_new_token(tokens[index], prompts[sample])
                if token is not None:
                    token = re.sub(r"[^\w\s]", '', token)
                    answer_split = answers[sample].split(" ")

                    idx = 0
                    while idx < len(answer_split):
                        if token.lower() == answer_split[idx]:
                            idx += 1
                            token_index += 1
                            if token_index < len(tokens[k]):
                                token = tokens[k][token_index]
                                token = re.sub(r'[^\w\s]', '', token)
                            else:
                                break
                        else:
                            break

                    if idx == len(answer_split):
                        sample_score += 1
                index += 1
            total_score += sample_score / K

    result = total_score / total_entries
    print(f"Exact match accuracy: {result}")
    return result


def get_exact_match_for_discrimination_with_batching(loader):
    processed_batches = 0
    total_score = 0
    for input_ids, answers, labels, prompts in iter(loader):
        processed_batches += 1
        if processed_batches % 50 == 0:
            print(f'Processed {processed_batches}/{number_of_batches} batches.')

        input_ids = input_ids.squeeze(1).to(device)

        with torch.inference_mode():
            outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, num_return_sequences=K,
                                     do_sample=DO_SAMPLE, max_new_tokens=MAX_SEQUENCE_LENGTH + 10,
                                     use_cache=True, temperature=0.5)
        tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        index = 0
        for sample in range(len(answers)):
            sample_score = 0
            for k in range(K):
                token, _ = get_first_new_token(tokens[index], prompts[sample])
                if token is not None:
                    token = re.sub(r'[^\w\s]', '', token)
                    if (token.lower() == 'yes' and labels[sample]) or \
                            (token.lower() == 'no' and not labels[sample]):
                        sample_score += 1
                index += 1
            total_score += sample_score / K

    result = total_score / total_entries
    print(f"Exact match accuracy: {result}")
    return result


def get_logprob_for_generation_with_batching(loader):
    processed_batches = 0
    sum_log_probabilities = 0
    for input_ids, answers, labels, prompts in iter(loader):
        processed_batches += 1
        if processed_batches % 50 == 0:
            print(f'Processed {processed_batches}/{number_of_batches} batches.')

        input_ids = input_ids.squeeze(1).to(device)
        with torch.inference_mode():
            outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, num_return_sequences=K,
                                     do_sample=DO_SAMPLE, max_new_tokens=MAX_SEQUENCE_LENGTH + 10,
                                     use_cache=True, return_dict_in_generate=True, output_scores=True,
                                     temperature=0.5)
        gen_sequences = outputs.sequences[:, input_ids.shape[-1]:]
        scores = torch.stack(outputs.scores, dim=1)
        gen_probs = torch.gather(scores, 2, gen_sequences[:, :, None]).squeeze(-1)

        probabilities = []
        index = 0
        for sample in range(len(answers)):
            # Deal with multi-word/long tokens.
            # Assumption: sum the probabilities of the parts.
            # https://stackoverflow.com/questions/59435020/get-probability-of-multi-token-word-in-mask-position
            answer = answers[sample]
            log_prob = 0
            token_index = len(input_ids[sample]) - 1
            token = tokenizer.decode(input_ids[sample][token_index])
            for k in range(K):
                while token in answer:
                    log_prob += gen_probs[index][token_index].item()
                    # Won't turn into an infinite loop since we put the answer at the end ourselves.
                    token_index -= 1
                    token = tokenizer.decode(input_ids[sample][token_index])
                index += 1
            probabilities.append(log_prob / K)

        for i in range(0, len(answers), 5):
            normalised_probabilities = softmax(probabilities[i:i + 5])

            for j in range(5):
                if labels[i + j]:
                    sum_log_probabilities += normalised_probabilities[j]

    result = 5 * sum_log_probabilities / total_entries
    print(f"Log match accuracy: {result}")
    return result


def get_logprob_for_discrimination_with_batching(loader):
    processed_batches = 0
    sum_log_probabilities = 0
    for input_ids, answers, labels, prompts in iter(loader):
        processed_batches += 1
        if processed_batches % 50 == 0:
            print(f'Processed {processed_batches}/{number_of_batches} batches.')

        input_ids = input_ids.squeeze(1).to(device)

        with torch.inference_mode():
            outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, num_return_sequences=K,
                                     do_sample=DO_SAMPLE, max_new_tokens=MAX_SEQUENCE_LENGTH + 10,
                                     temperature=0.5, return_dict_in_generate=True, output_scores=True,
                                     use_cache=True, )
        gen_sequences = outputs.sequences[:, input_ids.shape[-1]:]
        scores = torch.stack(outputs.scores, dim=1)
        gen_probs = torch.gather(scores, 2, gen_sequences[:, :, None]).squeeze(-1)

        probabilities = []
        index = 0
        for sample in range(len(answers)):
            log_prob = 0
            # The answer yes/no is always the last token and won't be split.
            token_index = len(input_ids[sample]) - 1
            for k in range(K):
                log_prob += gen_probs[index][token_index].item()
                index += 1
            probabilities.append(log_prob / K)

        for i in range(0, len(answers), 2):
            normalised_probabilities = softmax(probabilities[i:i + 2])
            if labels[i]:
                sum_log_probabilities += normalised_probabilities[0]
            else:
                sum_log_probabilities += normalised_probabilities[1]

    result = 2 * sum_log_probabilities / total_entries
    print(f"Log match accuracy: {result}")
    return result


def get_exact_match_for_generation_without_batching(sample_set):
    dataset = []
    for i in range(0, len(sample_set), 5):
        input_ids, answer, is_correct, prompt = sample_set[i]

        count = 1
        while not is_correct:
            input_ids, answer, is_correct, prompt = sample_set[i + count]
            count += 1
        dataset.append((input_ids, answer, prompt))

    total_score = 0
    processed = 0
    for entry in dataset:
        processed += 1
        if processed % 50 == 0:
            print(f'Processed {processed}/{len(dataset)}.')
        # Generate a response

        input_ids = entry[0].squeeze(1).to(device)

        with torch.inference_mode():
            outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, num_return_sequences=K,
                                     do_sample=DO_SAMPLE, max_new_tokens=MAX_SEQUENCE_LENGTH + 10,
                                     use_cache=True, temperature=0.5)
        tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        number_of_matches = 0
        for k in range(K):
            token, token_index = get_first_new_token(tokens[k], entry[2])

            if token is not None:
                token = re.sub(r"[^\w\s]", '', token)
                answer_split = entry[1].split(" ")

                idx = 0
                while idx < len(answer_split):
                    if token.lower() == answer_split[idx]:
                        idx += 1
                        token_index += 1
                        if token_index < len(tokens[k]):
                            token = tokens[k][token_index]
                            token = re.sub(r'[^\w\s]', '', token)
                        else:
                            break
                    else:
                        break

                if idx == len(answer_split):
                    number_of_matches += 1
                    print(f"MATCH!!!")

        total_score += number_of_matches / K

    result = total_score / len(dataset)
    print(f"Exact match accuracy for generation: {result}")
    return result


def get_exact_match_for_discrimination_without_batching(sample_set):
    total_score = 0
    processed = 0
    for input_ids, answer, is_correct, prompt in iter(sample_set):
        processed += 1
        if processed % 50 == 0:
            print(f'Processed {processed}/{len(sample_set)}.')

        input_ids = input_ids.squeeze(1).to(device)

        with torch.inference_mode():
            outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, num_return_sequences=K,
                                     do_sample=DO_SAMPLE, max_new_tokens=MAX_SEQUENCE_LENGTH + 10,
                                     use_cache=True, temperature=0.5)
        tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        number_of_matches = 0
        for k in range(K):
            token, token_index = get_first_new_token(tokens[k], prompt)

            if token is not None:
                token = re.sub(r"[^\w\s]", '', token)
                if (token.lower() == 'yes' and is_correct) or \
                        (token.lower() == 'no' and not is_correct):
                    number_of_matches += 1
                    print(f"MATCH!!!")

        total_score += number_of_matches / K

    result = total_score / len(sample_set)
    print(f"Exact match accuracy for discrimination: {result}")
    return result


def get_logprob_for_generation_without_batching(sample_set):
    processed = 0
    sum_log_probabilities = 0
    for i in range(0, len(sample_set), 5):
        processed += 5
        if processed % 50 == 0:
            print(f'Processed {processed}/{len(sample_set)}.')

        input_ids_1, answer_1, is_correct_1, _ = sample_set[i]
        input_ids_2, answer_2, is_correct_2, _ = sample_set[i+1]
        input_ids_3, answer_3, is_correct_3, _ = sample_set[i+2]
        input_ids_4, answer_4, is_correct_4, _ = sample_set[i+3]
        input_ids_5, answer_5, is_correct_5, _ = sample_set[i+4]

        input_ids = torch.stack((input_ids_1, input_ids_2, input_ids_3, input_ids_4, input_ids_5))
        answers = [answer_1, answer_2, answer_3, answer_4, answer_5]
        labels = [is_correct_1, is_correct_2, is_correct_3, is_correct_4, is_correct_5]

        input_ids = input_ids.squeeze(1).to(device)
        with torch.inference_mode():
            outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, num_return_sequences=K,
                                     do_sample=DO_SAMPLE, max_new_tokens=MAX_SEQUENCE_LENGTH + 10,
                                     use_cache=True, return_dict_in_generate=True, output_scores=True,
                                     temperature=0.5)
        gen_sequences = outputs.sequences[:, input_ids.shape[-1]:]
        scores = torch.stack(outputs.scores, dim=1)
        gen_probs = torch.gather(scores, 2, gen_sequences[:, :, None]).squeeze(-1)

        probabilities = []
        index = 0
        for j in range(5):
            # Deal with multi-word/long tokens.
            # Assumption: sum the probabilities of the parts.
            # https://stackoverflow.com/questions/59435020/get-probability-of-multi-token-word-in-mask-position
            answer = answers[j]
            prob = 0
            token_index = len(input_ids[j]) - 1
            token = tokenizer.decode(input_ids[j][token_index])
            for k in range(K):
                while token in answer:
                    prob += gen_probs[index][token_index].item()
                    # Won't turn into an infinite loop since we put the answer at the end ourselves.
                    token_index -= 1
                    token = tokenizer.decode(input_ids[j][token_index])
                index += 1
            probabilities.append(prob / K)

        normalised_probabilities = softmax(probabilities)

        for j in range(5):
            if labels[j]:
                sum_log_probabilities += normalised_probabilities[j]

    result = 5 * sum_log_probabilities / len(sample_set)
    print(f"Log match accuracy: {result}")
    return result


def get_logprob_for_discrimination_without_batching(sample_set):
    processed = 0
    sum_log_probabilities = 0

    for i in range(0, len(sample_set), 2):
        processed += 2
        if processed % 50 == 0:
            print(f'Processed {processed}/{len(sample_set)}.')

        # for "yes"
        input_ids_yes, _, label, _ = sample_set[i]
        # for "no"
        input_ids_no, _, _, _ = sample_set[i + 1]

        input_ids = torch.stack((input_ids_yes, input_ids_no))
        input_ids = input_ids.squeeze(1).to(device)

        with torch.inference_mode():
            outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, num_return_sequences=K,
                                     do_sample=DO_SAMPLE, max_new_tokens=MAX_SEQUENCE_LENGTH + 10,
                                     temperature=0.5, return_dict_in_generate=True, output_scores=True,
                                     use_cache=True, )
        gen_sequences = outputs.sequences[:, input_ids.shape[-1]:]
        scores = torch.stack(outputs.scores, dim=1)
        gen_probs = torch.gather(scores, 2, gen_sequences[:, :, None]).squeeze(-1)

        probabilities = []
        for j in range(2):
            prob = 0
            # The answer yes/no is always the last token and won't be split.
            token_index = len(input_ids[j]) - 1

            for k in range(K):
                prob += gen_probs[k+j][token_index].item()
            probabilities.append(prob / K)

        normalised_probabilities = softmax(probabilities)
        if label:
            sum_log_probabilities += normalised_probabilities[0]
        else:
            sum_log_probabilities += normalised_probabilities[1]

    result = 2 * sum_log_probabilities / len(sample_set)
    print(f"Log match accuracy: {result}")
    return result


def evaluate(sample_set, loader):
    if is_generation:
        print(f"\n---- GENERATION for {SIZE}----\n")

        if is_exact_match:
            if loader is None:
                return get_exact_match_for_generation_without_batching(sample_set)
            else:
                return get_exact_match_for_generation_with_batching(loader)
        else:
            if loader is None:
                return get_logprob_for_generation_without_batching(sample_set)
            else:
                return get_logprob_for_generation_with_batching(loader)

    else:
        print(f"\n---- DISCRIMINATION for {SIZE}----\n")

        if is_exact_match:
            if loader is None:
                return get_exact_match_for_discrimination_without_batching(sample_set)
            else:
                return get_exact_match_for_discrimination_with_batching(loader)

        else:
            if loader is None:
                return get_logprob_for_discrimination_without_batching(sample_set)
            else:
                return get_logprob_for_discrimination_with_batching(loader)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using: {device}\n")

    is_generation = sys.argv[1] == '0'
    print(f"IS GENERATION: {is_generation}")
    is_exact_match = sys.argv[2] == '0'
    print(f"IS EXACT MATCH: {is_exact_match}")
    SIZE = sys.argv[3]  # 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, and 12B.
    print(f"MODEL SIZE: {SIZE}")
    BATCH_SIZE = int(sys.argv[4])  # Need to be multiple of 2 (yes/no) * 5 (possible answers).
    print(f"BATCH SIZE: {BATCH_SIZE}")
    K = int(sys.argv[5])
    print(f"K: {K}")

    RANDOM_SEED = 42
    DO_SAMPLE = True
    MAX_SEQUENCE_LENGTH = 160
    REVISION = "step143000"
    MODEL_NAME = f'EleutherAI/pythia-{SIZE}-deduped'
    CACHE_DIR = f'{os.path.dirname(os.path.abspath(__file__))}/pythia-{SIZE}-deduped/{REVISION}'

    # Setting the random seed.
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    set_seed(RANDOM_SEED)

    print("\n---- PREPARING DATASET ----")
    original_dataset = load_dataset('riddle_sense')

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # A few pre-processing steps.
    eval_set = RiddleSenseDataset(original_dataset["validation"], MAX_SEQUENCE_LENGTH, tokenizer=tokenizer,
                                  is_generation=is_generation, is_exact_match=is_exact_match)

    if BATCH_SIZE != 1:
        eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE)
        number_of_batches = len(eval_loader)
    else:
        eval_loader = None

    total_entries = len(eval_set.data)

    print("\n---- DOWNLOADING MODEL ----")
    model = GPTNeoXForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    accuracy = evaluate(eval_set, eval_loader)
    write_result_to_file(model, is_generation, is_exact_match, K, accuracy, batch_size=BATCH_SIZE)
