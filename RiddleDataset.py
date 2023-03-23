from torch.utils.data import Dataset
import torch


class RiddleSenseDataset(Dataset):
    DISCRIMINATION_PROMPT = "Q: [QUESTION]\nA: [ANSWER]\nIs this a correct answer to the riddle?\n[OUTPUT]"
    GENERATION_PROMPT = "Q: [QUESTION]\nThe answer to the riddle is: [ANSWER]"

    def __init__(self, data, tokenizer, max_len, is_generation, is_exact_match):
        self.data = []
        for entry in data:
            question = entry['question']
            for i in range(len(entry['choices']['text'])):
                label = entry['choices']['label'][i]
                answer = entry['choices']['text'][i]
                if not is_generation and not is_exact_match:
                    self.data.append(DataEntry(question, answer, label == entry['answerKey'], 'yes'))
                    self.data.append(DataEntry(question, answer, label == entry['answerKey'], 'no'))
                else:
                    self.data.append(DataEntry(question, answer, label == entry['answerKey']))

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_generation = is_generation
        self.is_exact_match = is_exact_match

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        if self.is_generation:
            prompt = self.GENERATION_PROMPT
            if self.is_exact_match:
                prompt = prompt.replace('[ANSWER]', "")
        else:
            prompt = self.DISCRIMINATION_PROMPT
            prompt = prompt.replace('[ANSWER]', entry.answer)

        prompt = prompt.replace('[QUESTION]', entry.question)

        # Will only do so when an output is provided.
        self.prompt = prompt.replace('[OUTPUT]', entry.output)

        input_ids = self.tokenizer(self.prompt, return_tensors="pt", truncation=True, padding='max_length',
                                   max_length=self.max_len).input_ids

        if entry.output == "":
            return input_ids, entry.answer, entry.is_correct, self.prompt
        else:
            return input_ids, entry.output, entry.is_correct, self.prompt


class DataEntry:
    """
    Input entry in the format, for e.g.,
    'answerKey': 'E',
    'question': 'A man is incarcerated in prison, and as his punishment he has to carry a one tonne bag of sand
    backwards and forwards across a field the size of a football pitch. What is the one thing he can put in it to
    make it lighter?',
    'choices':
            'label': ['A', 'B', 'C', 'D', 'E'],
            'text': ['throw', 'bit', 'gallon', 'mouse', 'hole']
    """

    def __init__(self, question, answer, is_correct, output=""):
        self.question = question
        self.answer = answer
        self.is_correct = is_correct
        self.output = output


class GenerationDataset(Dataset):
    def __init__(self, dataset, split, tokenizer, max_len):
        self.data = []
        self.labels = []

        for entry in dataset[split]:
            question = entry['question']

            if split != 'test':
                for i in range(len(entry['choices']['text'])):
                    label = entry['choices']['label'][i]
                    answer = entry['choices']['text'][i]

                    if label == entry['answerKey']:
                        self.data.append(question + " " + answer)
            else:
                self.data.append(question)

        self.max_len = max_len
        self.encodings = tokenizer(self.data, return_tensors="pt", truncation=True, padding='max_length',
                                   max_length=self.max_len)
        
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings)


class DiscriminationDataset(Dataset):
    def __init__(self, dataset, split, tokenizer, max_len):
        self.data = []
        self.labels = []

        for entry in dataset[split]:
            question = entry['question']

            if split != 'test':
                for i in range(len(entry['choices']['text'])):
                    label = entry['choices']['label'][i]
                    answer = entry['choices']['text'][i]

                    if label == entry['answerKey']:
                        self.data.append(question + " " + answer)
            else:
                self.data.append(question)

        self.max_len = max_len
        self.encodings = tokenizer(self.data, return_tensors="pt", truncation=True, padding='max_length',
                                   max_length=self.max_len)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings)