from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset


def get_generation_accuracy(model, tokenizer, dataset):
    correct_outputs = 0
    total_outputs = 0

    for context in dataset['validation'].keys():
        total_outputs += 1

        if total_outputs % 100 == 0:
            print(f'Processed {total_outputs} data entries.')

        inputs = tokenizer(context, return_tensors="pt")
        tokens = model.generate(**inputs, max_length=128, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(tokens[0])

        if 'yes' in output.lower() and dataset['validation'][context]:
            correct_outputs += 1
        elif 'no' in output.lower() and not dataset['validation'][context]:
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
        tokens = model.generate(**inputs, max_length=128, pad_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(tokens[0])

        if 'yes' in output.lower() and dataset['validation'][context]:
            correct_outputs += 1
        elif 'no' in output.lower() and not dataset['validation'][context]:
            correct_outputs += 1

    return correct_outputs/total_outputs


def prepare_for_generation():
    # TODO. Pre-process dataset?
    # Format:
    # 'answerKey': 'E',
    # 'question': 'A man is incarcerated in prison, and as his punishment he has to carry a one tonne bag of sand
    # backwards and forwards across a field the size of a football pitch.  What is the one thing he can put in it to
    # make it lighter?',
    # 'choices':
    #           'label': ['A', 'B', 'C', 'D', 'E'],
    #           'text': ['throw', 'bit', 'gallon', 'mouse', 'hole']
    original_dataset = load_dataset('riddle_sense')

    dataset = {
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
                dataset[split][context] = label
    return dataset


def prepare_riddle_sense_dataset():
    # TODO. Pre-process dataset?
    # Format:
    # 'answerKey': 'E',
    # 'question': 'A man is incarcerated in prison, and as his punishment he has to carry a one tonne bag of sand
    # backwards and forwards across a field the size of a football pitch.  What is the one thing he can put in it to
    # make it lighter?',
    # 'choices':
    #           'label': ['A', 'B', 'C', 'D', 'E'],
    #           'text': ['throw', 'bit', 'gallon', 'mouse', 'hole']
    original_dataset = load_dataset('riddle_sense')

    dataset = {
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
                dataset[split][context] = label
    return dataset


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

    dataset = prepare_riddle_sense_dataset()


    # Generation
    d_acc = get_discrimination_accuracy(model, tokenizer, dataset)
    print(d_acc)

