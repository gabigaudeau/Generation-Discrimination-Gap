import torch
import random
from datasets import load_dataset
import os
from RiddleDataset import DiscriminationDataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import numpy as np
import evaluate


metric = evaluate.load("accuracy")


if __name__ == '__main__':
    RANDOM_SEED = 42
    BATCH_SIZE = 20  # Need to be multiple of 2 * K.
    K = 5
    DO_SAMPLE = K != 1
    MAX_SEQUENCE_LENGTH = 160
    NUM_EPOCHS = 3
    LR = 4e-5
    OUTPUT_DIR = "./results/final"

    model_sizes = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
    # 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, and 12B.

    # Setting the random seed.
    random.seed(RANDOM_SEED)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    print("---- PREPARING DATASET ----\n")
    original_dataset = load_dataset('riddle_sense')

    SIZE = "70m"
    MODEL_NAME = f'EleutherAI/pythia-{SIZE}-deduped'
    REVISION = 'step3000'
    CACHE_DIR = f'{os.path.dirname(os.path.abspath(__file__))}/pythia-{SIZE}-deduped/step3000'

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=REVISION,
        cache_dir=CACHE_DIR,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.mask_token = "[MASK]"
    tokenizer.padding_side = 'left'

    model = GPTNeoXForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=REVISION,
        cache_dir=CACHE_DIR
    )
    model.to(device)

    # Generation == question answering
    train_set = GenerationDataset(original_dataset, 'train', tokenizer, MAX_SEQUENCE_LENGTH)
    val_set = GenerationDataset(original_dataset, 'validation', tokenizer, MAX_SEQUENCE_LENGTH)
    test_set = GenerationDataset(original_dataset, 'test', tokenizer, MAX_SEQUENCE_LENGTH)

    print(len(train_set))
    print(len(val_set))

    # Specify the arguments for the trainer
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        logging_dir='./logs',
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        seed=RANDOM_SEED,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # Call the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    trainer.evaluate()

    model_to_save = trainer.model.module if hasattr(trainer.model,
                                                    'module') else trainer.model
    model_to_save.save_pretrained(OUTPUT_DIR)

    # loading the model you previously trained
    model = GPTNeoXForCausalLM.from_pretrained(OUTPUT_DIR)

    inputs = tokenizer("My life can be measured in hours. I serve by being devoured. Thin, I am quick. "
                       "Fat, I am slow. Wind is my foe. What am I?", padding=True, return_tensors="pt")

    tokens = model.generate(**inputs, max_length=160, pad_token_id=tokenizer.eos_token_id,
                            num_return_sequences=K, do_sample=DO_SAMPLE)

    out = tokenizer.batch_decode(tokens)
    print(out)





