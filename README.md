# GLT
This project is a PyTorch implementation of GLT,
a novel approach for stock price movement prediction 
considering historical stock prices and sparse tweets.

## Prerequisites

- Python 3.7.7
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org)
- [Scipy](https://scipy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [SentencePiece](https://github.com/google/sentencepiece)

## Usage

You can run the demo script by `bash demo.sh`, which simply runs `src/predict.py`.
It is also possible to modify directly our model package such to change
the hyperparameters such as learning rates, the dimension of hidden units, and so on.