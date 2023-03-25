import os


def write_result_to_file(model_name, is_generation, is_exact_match, k, result, batch_size=1):
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