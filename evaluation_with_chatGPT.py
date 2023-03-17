from datasets import load_dataset
import random
from torch.utils.data import Dataset
import torch
from itertools import islice
import openai
import sys


class RiddleSenseDataset(Dataset):
    DISCRIMINATION_PROMPT = "[QUESTION]\nIs [ANSWER] a correct answer to the riddle?\n[OUTPUT]"
    GENERATION_PROMPT = "[QUESTION]\nThe answer to the riddle is: [ANSWER]"

    def __init__(self, data, max_len, is_generation, is_exact_match):
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
                prompt = prompt.replace('[ANSWER]', entry.answer)
        else:
            prompt = self.DISCRIMINATION_PROMPT
            prompt = prompt.replace('[ANSWER]', entry.answer)

        prompt = prompt.replace('[QUESTION]', entry.question)

        # Will only do so when an output is provided.
        self.prompt = prompt.replace('[OUTPUT]', entry.output)

        if entry.output == "":
            return self.prompt, entry.answer, entry.is_correct
        else:
            return self.prompt, entry.output, entry.is_correct


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
        self.question = question.strip()
        self.answer = answer.strip()
        self.is_correct = is_correct
        self.output = output


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using: {device}\n")

    RANDOM_SEED = 42
    DO_SAMPLE = True
    MAX_SEQUENCE_LENGTH = 160
    K = 2
    BATCH_SIZE = 10

    # Setting the random seed.
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("\n---- PREPARING DATASET ----")
    original_dataset = load_dataset('riddle_sense')

    # Split validation set into two.
    sample = list(islice(original_dataset["validation"], 100))
    sample_set = RiddleSenseDataset(sample, MAX_SEQUENCE_LENGTH,
                                    is_generation=True, is_exact_match=True)

    dataset = []
    for i in range(0, len(sample_set), 5):
        prompt, answer, label = sample_set[i]

        count = 1
        while not label:
            _, answer, label = sample_set[i + count]
            count += 1
        dataset.append((prompt, answer))

    print("\n---- FETCHING MODEL ----")
    # Define OpenAI API key
    openai.api_key = "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"

    # Set up the model and prompt
    model_engine = "gpt-3.5-turbo"

    # Generate a response
    # completion = openai.Completion.create(
    #     engine=model_engine,
    #     prompt=prompt,
    #     max_tokens=170,
    #     n=1,
    #     stop=None,
    #     temperature=0.5,
    #     best_of=K,
    # )
    #
    # response = completion.choices[0].text
    # print(response)

