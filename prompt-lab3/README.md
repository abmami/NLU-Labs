# Prompt Engineering with GPT-J using a Twitter Disaster Dataset for Classification

This repository contains code and instructions for using quantized GPT-J to perform prompt engineering and classify whether a tweet is related to a disaster or not.


## About GPT-J 

[GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b/) is a state-of-the-art language model developed by EleutherAI, which has 6 billion parameters and is trained on a large corpus of text data. It has achieved impressive results on a wide range of natural language processing tasks, including language generation, question-answering, and text classification.

GPT-J is very big and requires over 22 GB of memory for one type of parameter alone, making it difficult to use on most single-GPU setups. However, [Quantized GPT-J](https://huggingface.co/hivemind/gpt-j-6B-8bit) is an 8-bit quantized version of GPT-J that is much smaller and faster, and can be fine-tuned on a single GPU with about 11 GB memory using dynamic 8-bit quantization of large weight tensors, gradient checkpoints, and scalable fine-tuning. 

## About the Dataset

The dataset used in this repository is a collection of tweets from the [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started) Kaggle competition. The dataset is split into a training set (~7500 tweets) and a test set (~3200 tweets). The training set is further split : 10% for training and 90% for validation. The dataset is available in the `data` folder.

The dataset contains the following columns:

- `id`: a unique identifier for each tweet
- `keyword`: a particular keyword from the tweet (may be blank)
- `location`: the location the tweet was sent from (may be blank)
- `text`: the text of the tweet
- `target`: this denotes whether a tweet is about a real disaster (1) or not (0)

## About the Task

The task is to classify whether a tweet is related to a disaster or not using prompt engineering with GPT-J. There are several approaches to this task:

* Zero-Shot classification : this approach involves using the pre-trained GPT-J model to classify tweets as disaster-related or not without any additional training or fine-tuning. This is achieved by using a prompt that includes the relevant keywords and context to guide the model towards the correct classification.
* Few-Shot Classification : this approach involves providing the model with a small number of labeled examples (e.g. 10-20 tweets) related to disasters and non-disasters to guide the model on the specific task at hand. This can improve the accuracy of the model compared to zero-shot classification by providing some task-specific information to the model.
* Prompt Tuning : it involves experimenting with different prompts or input formats to find the one that works best for the target task. 
* Another approach is to use a combination of the above approaches to achieve the best results.
* ...

## System Requirements
The code in this repository was tested on the following system:
* Ubuntu 21.04
* GTX 1650 Ti 4GB
* RAM 24 GB
* Python 3.10.6
* CUDA 11.7

## Usage 
To reproduce the results of the prompt engineering experiments locally, follow the steps below:
1. Clone the prompt-lab3 branch.
2. Install the requirements:

``` 
cd prompt-lab3
pip install -r requirements.txt
```

3. Run the notebook `notebook.ipynb` to perform prompt engineering with GPT-J and classify tweets as disaster-related or not.


## Results

The results of the prompt engineering experiments are summarized in the table below:

| Approach | Accuracy | F1 Score |
| --- | --- | --- |
| Zero-Shot | X | X |
| Few-Shot | X | X |
| Few-Shot + Prompt Tuning | X | X |

