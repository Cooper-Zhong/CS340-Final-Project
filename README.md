# Bias in Toxicity Classification

## Intro

This is the final project of SUSTech 2024 Spring CS340 Computational Ethics. The project is to investigate the bias in toxicity classification. The dataset used in this project is the [Civil Comments dataset](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) from Kaggle. In short, we trained a BERT model and acheived a `0.935` final score under a special metric (please refer to the [Evaluation](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation) section of the competition).

## Note
- Use `conda` to manage environment and packages, e.g. `conda install xxx`. Make sure Pytorch, tensorflow, keras, cudann ... are cmopatible.
- GPU is all you need. Training is suggested to be done on kaggle, where you have 30h gpu quota per week, and **9 hours** per kernel session. Make sure your kernel finishes in **9h**. Inference can be done locally, for refernce, it takes around 15GB memory(maximum) on a Quadro RTX 6000 for a batch size of **300** comments using detoxify models. The `pytorch-bert-inference.ipynb` uses less than 2GB memory for a batch size of 32.


## Structure

Some of the folders or files are missing due to the size.

- `Data` (missing): Due to data size, we didn't upload the data here. You can download the data from the [Kaggle competition page](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data) and put them in this folder which includes `sample_submission.csv`, `test_private_expanded.csv`, `test_public_expanded.csv`, `test.csv`, `train.csv`.
- `Embeddings` (missing): The `glove.6B` embeddings are stored here (to train the benchmark CNN). Download them from [standfordnlp](https://nlp.stanford.edu/projects/glove/) and put them in this folder.
- `models` (missing): To save the trained benchmark CNN model and tokenizer.
- `submissions`: The submission files (to calculate the final score).
- `bias`: The bias analysis (bias and final score metrics) results.
- `kaggle`: The demonstration notebook for training (on kaggle) and inference (locally). Inside `bert-inference` is the BERT config and model files, checkout the [kaggle kernel](https://www.kaggle.com/code/cooperkaggle/toxic-bert-plain-vanila) for the `.bin` model file.
- `benchmark.ipynb`: A CNN model training using keras and tensorflow.
- `cnn_predict.ipynb`: Prediction using CNN model.
- `alignment.ipynb`: Quick check of the CNN prediction score to align with human intent.
- `detoxify_predict.ipynb`: Prediction using model from [Detoxify](https://github.com/unitaryai/detoxify)
- `metrics.py`: Functions to calculate final score.
- `eval_bias.ipynb`: Bias evaluated by Demographic Parity and Equalized Opportunity for all models.
- `final_score.ipynb`: Calculate the final score for all models.
- `tradeoff.ipynb`: Calculate the accuracy of models.

In total, we compared 5 models: `cnn` from benchmark; `roberta-base-unbiased-small` and `roberta-base-unbiased` from Detoxify,`kaggle_bert` and `my_kaggle_bert` from kaggle.

## Final Score

- `cnn`:  0.8811
- `roberta-base-unbiased-small`:  0.9336
- `roberta-base-unbiased`:  0.9374
- `kaggle_bert`:  0.9383
- `my_kaggle_bert`:  **0.9351**

Checkout [the kaggle kernel](https://www.kaggle.com/code/cooperkaggle/toxic-bert-plain-vanila) to see the training process and obtain the `.bin` model file. It takes **8.5h** to train on **1200000** data.


## References
- https://github.com/unitaryai/detoxify
- https://www.kaggle.com/code/yuval6967/toxic-bert-plain-vanila/notebook
- https://www.kaggle.com/code/abhishek/pytorch-bert-inference/notebook
- https://www.kaggle.com/datasets/matsuik/ppbert
- https://www.kaggle.com/code/christofhenkel/loading-bert-using-pytorch-with-tokenizer-apex




