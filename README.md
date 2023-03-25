# Riddle-style question answering: the effects of model size on the Generation-Discrimination gap.


This repository is for a [CarperAI](https://carper.ai) work-in-progress project which involves evaluating and fine-tuning Large Language 
Models ([Pythia Scaling Suite](https://huggingface.co/EleutherAI/pythia-70m-deduped), 
[GPT-3 text](https://platform.openai.com/docs/models/gpt-3), etc.) of varying capabilities on the task of riddle-style 
question answering.


## Abstract

-----

Leveraging Large Language Models (LLMs) to supervise the development of other models can alleviate the need for or work 
of human evaluators in Human-in-the-loop (HITL) approaches. To identify which applications might  be most receptive to 
this technique, [Saunders et al. (2022)](https://arxiv.org/pdf/2206.05802.pdf) proposes a way of quantifying the gap 
between the ability of a model to discriminate the quality of its own output, and its generation capacities. This study
turns to the task of riddle-style question answering as a likely candidate. Our findings suggest that for this 
application, there is a positive gap between discrimination and generation. Further, we find that with scale, both 
generation and discrimination improve, but the gap remains highly positive.


## Contents

-----


- `evaluation_with_Pythia.py` contains the main workflow and model accuracy metrics (exact match, log probability) to 
work with Pythia models.
- Similarly, see `evaluation_with_GPT-3.py` for working with GPT-3 models and exact match accuracies.
- See `RiddleDataset.py` for [RiddleSense](https://huggingface.co/datasets/riddle_sense) dataset processing, and 
`utils.py` for helper functions.
- `tuning_discriminator_with_Pythia.py` (and its generator equivalent) are first attempts at fine-tuning the Pythia 
suite on the RiddleSense dataset. These are both incomplete and potentially incorrect.
- See `paper.pdf` for first results.
