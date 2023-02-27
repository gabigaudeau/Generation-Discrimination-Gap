from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset

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

    tokenizer.add_tokens(['[SEP]'], special_tokens=True)

    # Prepare dataset for discrimination.
    # Format:
    # 'answerKey': 'E',
    # 'question': 'A man is incarcerated in prison, and as his punishment he has to carry a one tonne bag of sand
    # backwards and forwards across a field the size of a football pitch.  What is the one thing he can put in it to
    # make it lighter?',
    # 'choices':
    #           'label': ['A', 'B', 'C', 'D', 'E'],
    #           'text': ['throw', 'bit', 'gallon', 'mouse', 'hole']
    dataset = load_dataset('riddle_sense')

    encodings = {}  # questions and answers
    labels = {}  # labels
    for split in ['train', 'validation', 'test']:
        encodings[split] = []
        labels[split] = []
        for entry in dataset[split]:
            question = entry['question']
            for choice in entry['choices']['text']:
                encoding = tokenizer(question + ' [SEP]', choice, return_tensors='pt')
                encodings[split].append(encoding)
            for label in entry['choices']['label']:
                if label == entry['answerKey']:
                    labels[split].append(True)
                else:
                    labels[split].append(False)

    print(encodings['train'][0])

    # Vary model sizes:
    # 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, and 12B.
    model = GPTNeoXForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=REVISION,
        cache_dir=CACHE_DIR,
    )
    
    # Generation
    # inputs = tokenizer("Hello, I am", return_tensors="pt")
    # tokens = model.generate(**inputs)
    # print(tokenizer.decode(tokens[0]))
