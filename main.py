from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from torch.nn import functional as F


def get_log_discrimination_accuracy(model, tokenizer, dataset):

    for question in dataset['validation'].keys():
        for tuple in question:
            input_ids = tokenizer(tuple[0], return_tensors="pt").input_ids
            output = model.generate(input_ids, max_length=160, pad_token_id=tokenizer.eos_token_id,
                                    return_dict_in_generate=True, output_scores=True)
            tokens = output["sequences"][:, input_ids.shape[-1]:]
            sequences = tokenizer.decode(tokens[0], skip_special_tokens=True)
            probs = torch.stack(output["scores"], dim=1).softmax(-1)
            gen_probs = torch.gather(probs, 2, tokens[:, :, None]).squeeze(-1)


            print(gen_probs)

            # if dataset['validation'][question] in output.lower():
            #     correct_outputs += 1

    return None


def get_generation_accuracy(model, tokenizer, dataset):
    correct_outputs = 0
    total_outputs = 0

    for question in dataset['validation'].keys():
        total_outputs += 1

        if total_outputs % 100 == 0:
            print(f'Processed {total_outputs} data entries.')

        inputs = tokenizer(question, return_tensors="pt")
        tokens = model.generate(**inputs, max_length=160, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(tokens[0])

        if dataset['validation'][question] in output.lower():
            correct_outputs += 1

    return correct_outputs / total_outputs


def get_discrimination_accuracy(model, tokenizer, dataset):
    correct_outputs = 0
    total_outputs = 0

    for context in dataset['validation'].keys():
        total_outputs += 1

        if total_outputs % 100 == 0:
            print(f'Processed {total_outputs} data entries.')

        inputs = tokenizer(context, return_tensors="pt")
        tokens = model.generate(**inputs, max_length=160, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(tokens[0])

        if 'yes' in output.lower() and dataset['validation'][context]:
            correct_outputs += 1
        elif 'no' in output.lower() and not dataset['validation'][context]:
            correct_outputs += 1

    return correct_outputs/total_outputs


def prepare_riddle_dataset_for_log_discrimination():
    original_dataset = load_dataset('riddle_sense')

    new_dataset = {
        'train': {},
        'validation': {},
        'test': {}}

    for split in ['train', 'validation', 'test']:
        for entry in original_dataset[split]:
            question = entry['question']
            new_dataset[split][question] = []
            for i in range(len(entry['choices']['text'])):
                # encoding = tokenizer(question + ' [SEP]', choice, return_tensors='pt')
                answer = entry['choices']['text'][i]
                context = f'Q: {question}\nA: {answer.capitalize()}\nIs this a correct answer to the riddle?'
                label = entry['choices']['label'][i] == entry['answerKey']
                new_dataset[split][question].append((context, label))
    return new_dataset


def prepare_riddle_dataset_for_generation():
    original_dataset = load_dataset('riddle_sense')

    new_dataset = {
        'train': {},
        'validation': {},
        'test': {}}

    for split in ['train', 'validation', 'test']:
        for entry in original_dataset[split]:
            question = entry['question']
            for i in range(len(entry['choices']['text'])):
                if entry['choices']['label'][i] == entry['answerKey']:
                    answer = entry['choices']['text'][i]
                    new_dataset[split][question] = answer

    print(new_dataset['train'].popitem())
    return new_dataset


def prepare_riddle_dataset_for_discrimination():
    original_dataset = load_dataset('riddle_sense')

    new_dataset = {
        'train': {},
        'validation': {},
        'test': {}}

    for split in ['train', 'validation', 'test']:
        for entry in original_dataset[split]:
            question = entry['question']
            for i in range(len(entry['choices']['text'])):
                # encoding = tokenizer(question + ' [SEP]', choice, return_tensors='pt')
                answer = entry['choices']['text'][i]
                context = f'Q: {question} \nA: {answer.capitalize()} \nIs this a correct answer to the riddle?'
                label = entry['choices']['label'][i] == entry['answerKey']
                new_dataset[split][context] = label
    return new_dataset


if __name__ == '__main__':
    # Settings
    MODEL_NAME = 'EleutherAI/pythia-70m-deduped'
    REVISION = 'step3000'
    CACHE_DIR = './pythia-70m-deduped/step3000'

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=REVISION,
        cache_dir=CACHE_DIR,
    )

    # tokenizer.add_tokens(['[SEP]'], special_tokens=True)

    # Vary model sizes:
    # 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, and 12B.
    model = GPTNeoXForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=REVISION,
        cache_dir=CACHE_DIR,
    )

    # TODO. Pre-process dataset?
    # Format:
    # 'answerKey': 'E',
    # 'question': 'A man is incarcerated in prison, and as his punishment he has to carry a one tonne bag of sand
    # backwards and forwards across a field the size of a football pitch.  What is the one thing he can put in it to
    # make it lighter?',
    # 'choices':
    #           'label': ['A', 'B', 'C', 'D', 'E'],
    #           'text': ['throw', 'bit', 'gallon', 'mouse', 'hole']

    # Discrimination
    # dataset = prepare_riddle_dataset_for_discrimination()
    # d_acc = get_discrimination_accuracy(model, tokenizer, dataset)
    # print(d_acc)

    # Generation
    # dataset = prepare_riddle_dataset_for_generation()
    # g_acc = get_generation_accuracy(model, tokenizer, dataset)
    # print(g_acc)

    dataset = prepare_riddle_dataset_for_log_discrimination()

    sentence = "Q: Something very helpful if you want to go gently down a stream.\nA: [MASK]\n" \
               "Is this a correct answer to the riddle?"
    targets = ["raft", "roll down hill", "rowboat", "water", "roll over"]
    target_log_P = {t: None for t in targets}
    for target in target_log_P:
        input = tokenizer(sentence.replace("[MASK]", target), return_tensors="pt", )
        output = model(**input)
        target_log_P[target] = sum([
            torch.log(F.softmax(output.logits[0][i], dim=-1)[idx])
            for i, idx in enumerate(input.input_ids[0])
        ]).item()

    print(target_log_P)

    # {'raft': -254.1611785888672, 'roll down hill': -275.2216796875, 'rowboat': -259.2183837890625,
     # 'water': -251.62864685058594, 'roll over': -262.9842529296875}

    # l_acc = get_log_discrimination_accuracy(model, tokenizer, dataset)

