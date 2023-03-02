from datasets import load_dataset


class RiddleSenseDataset:
    DATASET_NAME = 'riddle_sense'

    def __init__(self):
        original_dataset = load_dataset(self.DATASET_NAME)

        self.dataset = {}
        for split in ['train', 'validation', 'test']:
            self.dataset[split] = []
            for entry in original_dataset[split]:
                self.dataset[split].append(DataEntry(entry))


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
    def __init__(self, entry):
        # For the test set, the correct answer is not provided.
        if entry['answerKey'] == "":
            self.answer_key = None
        else:
            self.answer_key = entry['answerKey']

        self.question = entry['question']

        # Define a dictionary of answer labels mapping to string answers.
        self.choices = {}
        for i in range(len(entry['choices']['text'])):
            label = entry['choices']['label'][i]
            answer = entry['choices']['text'][i]
            self.choices[label] = answer
