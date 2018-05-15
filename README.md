# LEAM
This repository contains source code necessary to reproduce the results presented in the paper [Joint Embedding of Words and Labels for Text Classification](https://arxiv.org/) (ACL 2018):

```
@inproceedings{wang_id_2018_ACL,
  title={Joint Embedding of Words and Labels for Text Classification},
  author={Guoyin Wang, Chunyuan Li, Wenlin Wang, Yizhe Zhang, Dinghan Shen, Xinyuan Zhang, Ricardo Henao, Lawrence Carin},
  booktitle={ACL},
  year={2018}
}
```
## Contents
There are four steps to use this codebase to reproduce the results in the paper.

1. [Dependencies](#dependencies)
2. [Prepare datasets](#prepare-datasets)
3. [Training](#training)
    1. Training on standard dataset
    2. Training on your own dataset
4. [Reproduce paper figure results](#reproduce-paper-figure-results)

## Dependencies

This code is based on Python 2.7, with the main dependencies being [TensorFlow==1.7.0](https://www.tensorflow.org/) and [Keras==2.1.5](https://keras.io/). Additional dependencies for running experiments are: `numpy`, `cPickle`, `scipy`, `math`, `gensim`. 

## Prepare datasets

We consider the following datasets: Yahoo, AGnews, DBPedia, yelp, yelp binary. For convenience, we provide pre-processed versions of all datasets. Data are prepared in pickle format. Each `.p` file has the same fields in same order: `train text`, `val text`, `test text`, `train label`, `val label`, `test label`, `dictionary` and `reverse dictionary`.

Datasets can be downloaded [here](https://drive.google.com/open?id=1QmZfoKSgZl8UMN8XenAYqHaRzbW5QA26). Put the download data in data directory. Each dataset has two files: tokenized data and corresponding pretrained Glove embedding.

To run your own dataset, please follow the code in `preprocess_yahoo.py` to tokenize and split train/dev/test datsset. To build pretained word embeedings, first dowload [Golve word embeddings](https://nlp.stanford.edu/projects/glove/) and then follow `glove_generate.py`. 

## Training
**1. Training on standard dataset**

To run the test, use the command `python -u main.py`. The default test is on Yahoo dataset. To run other default dataset, change the [`Option class`] attribute dataset to corresponding dataset name. Most the parameters are defined in the [`Option class`](./main.py) part. 

## Reproduce paper figure results
Jupyter notebooks in [`plots`] folders are used to reproduce paper figure results.

Note that without modification, we have copyed our extracted results into the notebook, and script will output figures in the paper. If you've run your own training and wish to plot results, you'll have to organize your results in the same format instead.






