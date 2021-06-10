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

## Dataset
We use three benchmarks to evaluate our proposed method,
CIKM21, [ACL18](https://www.aclweb.org/anthology/P18-1183.pdf),
and [CIKM18](https://dl.acm.org/doi/pdf/10.1145/3269206.3269290) in Table 1.
All datasets consist of high-trade-volume stocks in US stock markets,
and CIKM21 is a new dataset that we collect and publicly release.


Table 1:  Summary of datasets.

| Dataset | # Stocks | # Tweets | Train | Valid | Test |
|------| ---:|-----------:|:------|:----|:-----|
| CIKM21 | 50 | 272,762  | 07/2019 - | 03/2020 - | 05/2020 - 07/2020 |
| ACL18  | 87 | 106,271  | 01/2014 - | 08/2015 - | 10/2015 - 12/2015 |
| CIKM18 | 38 | 955,788  | 01/2017 - | 09/2017 - | 11/2017 - 12/2017 |

