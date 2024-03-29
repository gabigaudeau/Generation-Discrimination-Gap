import os
import re
import numpy as np


def write_result_to_file(model_name, is_generation, is_exact_match, k, result, batch_size=1):
    """Print result to separate individual identifiable file.
    For e.g. 70m_generation_em_b40_K5.txt for the exact match accuracy on
    a generation task with the 70m deduplicated Pythia model, a batch-size
     of 40, over 5 sampled sequences."""

    if is_generation:
        task = "generation"
    else:
        task = "discrimination"

    if is_exact_match:
        accuracy = "em"
    else:
        accuracy = "lp"

    # Write results
    file = open(f"{os.path.dirname(os.path.abspath(__file__))}/evaluation_results/"
                f"{model_name}_{task}_{accuracy}_b{batch_size}_K{k}.txt", "w")
    file.write(f"{accuracy} for {task} model {model_name}")
    file.write("\n----------------------\n")
    file.write(f"{result}  | ")
    file.write("\n----------------------\n\n")

    file.close()


def get_first_new_token(output, prompt=None):
    """Return the first nonempty token generated by the model.
    One that is different from the prompt in the case of Pythia outputs."""
    split_output = output.split()

    if prompt is not None:
        split_prompt = prompt.split()
        index = len(split_prompt)
    else:
        index = 0
    while index < len(split_output):
        token = split_output[index]
        token = re.sub(r"[^\w\s]", '', token)
        if token != "" and token is not None and token != " " and token.lower() != "a"\
                and token.lower() != 'the' and token.lower() != "an":
            return token, index
        index += 1

    return None, None


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / sum(e_x) .sum(axis=0)  # only difference
