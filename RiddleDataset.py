from torch.utils.data import Dataset


class RiddleSenseDataset(Dataset):
    DISCRIMINATION_PROMPT = "Q: [QUESTION]\nA: [ANSWER]\nIs this a correct answer to the riddle?\n[OUTPUT]"
    GENERATION_PROMPT = "Q: [QUESTION]\nThe answer to the riddle is: [ANSWER]"

    def __init__(self, data, tokenizer, max_len, is_generation):
        self.data = []
        for entry in data:
            question = entry['question']
            for i in range(len(entry['choices']['text'])):
                label = entry['choices']['label'][i]
                answer = entry['choices']['text'][i]
                self.data.append(DataEntry(question, answer, label == entry['answerKey']))

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_generation = is_generation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        entry = self.data[index]

        if self.is_generation:
            self.prompt = self.GENERATION_PROMPT \
                .replace('[QUESTION]', entry.question) \
                .replace('[ANSWER]', "")
        else:
            self.prompt = self.DISCRIMINATION_PROMPT \
                .replace('[QUESTION]', entry.question) \
                .replace('[ANSWER]', entry.answer) \
                .replace('[OUTPUT]', "")

        input_ids = self.tokenizer(self.prompt, return_tensors="pt", truncation=True, padding='max_length',
                                   max_length=self.max_len).input_ids

        return input_ids, entry.answer, entry.is_correct, self.prompt


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

    def __init__(self, question, answer, is_correct):
        self.question = question
        self.answer = answer
        self.is_correct = is_correct
