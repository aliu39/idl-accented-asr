Credit to Sanchit Gandhi for starter code, README, and loading/fine-tuning with commonvoice.

POSSIBLE TODOS for team, after grading deadline:
- add AccentDB classifier notebook (Mohamed) and integrate into a fluid pipeline (same notebook as finetune model)
- Speech Accent Archive loading and inference code (Laura)
- tune hyperparameters
- support work with more collaborator credits and references

# Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers

CMU 11-785 Intro to Deep Learning Course Project

Democratizing Speech: Automatic Speech Recognition for Second Language Teaching

## Introduction

This notebook presents a detailed guide on how to fine-tune OpenAI's Whisper model for any multilingual Automatic Speech Recognition (ASR) dataset using Hugging Face's Transformers. This content serves as a hands-on companion to the accompanying blog post, offering a practical approach to the fine-tuning process.

## Whisper Model Overview

Whisper, released by OpenAI in September 2022, is a groundbreaking pre-trained model for ASR. Developed by Alec Radford et al., it is unique in its training approach, utilizing a massive quantity of labelled audio-transcription data (680,000 hours). This is significantly more than the unlabelled data used in previous models like Wav2Vec 2.0. Notably, 117,000 hours of this pre-training data is multilingual, enabling the model to handle over 96 languages, including many low-resource ones.

The extensive training allows Whisper to generalize effectively across various datasets and domains. It achieves competitive results in benchmarks like LibriSpeech ASR and sets new standards in others like TED-LIUM. The pre-trained checkpoints can be further fine-tuned for specific datasets and languages, enhancing their performance for low-resource languages.

## How to run the Code

### Dependencies and Installation

We'll employ several popular Python packages to fine-tune the Whisper model.
We'll use `datasets` to download and prepare our training data and
`transformers` to load and train our Whisper model. We'll also require
the `soundfile` package to pre-process audio files, `evaluate` and `jiwer` to
assess the performance of our model. Finally, we'll
use `gradio` to build a flashy demo of our fine-tuned model.

```bash
!pip install datasets>=2.6.1
!pip install git+https://github.com/huggingface/transformers
!pip install kaleido
!pip install cohere
!pip install openai
!pip install tiktoken
!pip install soundfile
!pip install tiktoken
!pip install librosa
!pip install evaluate>=0.30
!pip install jiwer
!pip install gradio
!pip install accelerate -U
```

### Hugging Face Hub

You can upload model checkpoints directly the [Hugging Face Hub](https://huggingface.co/)
whilst training. 

Find your Hub authentication token [here](https://huggingface.co/settings/tokens):

### Load dataset and Preprocessing

Ensure you have accepted the terms of use on the Hugging Face Hub: [mozilla-foundation/common_voice_11_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0). Once you have accepted the terms, you will have full access to the dataset and be able to download the data locally. Just load the dataset with `load_dataset` function. 

Data is loaded from [AccentDB](https://github.com/AccentDB/code).

Follow the code cells in the notebook to preprocess data and load pre-trained models.

### Training and Evaluation

We used the [🤗 Hugging Face  Trainer](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer) to train our data

- Evaluation metrics: during evaluation, we want to evaluate the model using the [word error rate (WER)](https://huggingface.co/metrics/wer) metric. We need to define a `compute_metrics` function that handles this computation.

- Load a pre-trained checkpoint: we need to load a pre-trained checkpoint and configure it correctly for training.

- Define the training configuration: this will be used by the 🤗 Trainer to define the training schedule.

To launch training, simply execute:

```
trainer.train()
```

Training will take approximately 5-10 hours depending on your GPU or the one
allocated to this Google Colab. If using this Google Colab directly to
fine-tune a Whisper model, you should make sure that training isn't
interrupted due to inactivity. A simple workaround to prevent this is
to paste the following code into the console of this tab (_right mouse click_
-> _inspect_ -> _Console tab_ -> _insert code_).

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.
