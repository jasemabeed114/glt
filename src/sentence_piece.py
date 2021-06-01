import sentencepiece as spm
import os
import json
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pickle
import argparse
from util import *


def get_info(DATA, USER_DEFINED_SYMBOLS):
    train_date, valid_date, test_date, end_date = get_date_info(DATA)
    STOCK_LIST = os.listdir(os.path.join(GENERAL_PATH, 'data', DATA, 'tweet'))
    for stock in STOCK_LIST:
        USER_DEFINED_SYMBOLS += ',$' + stock.lower()
    return train_date, valid_date, test_date, end_date, STOCK_LIST, USER_DEFINED_SYMBOLS


def get_text(DATA_PATH, DATA):
    texts = ''
    for stock in STOCK_LIST:
        stock_path = os.path.join(DATA_PATH, DATA, 'tweet', stock)
        date_list = sorted(os.listdir(stock_path))
        # date_list = ['2017-01-01', '2017-01-02', '2017-01-03', ... ,]
        for date in date_list:
            if date >= train_date and date < valid_date:
                with open(os.path.join(stock_path, date), 'r',
                          encoding='utf8') as f:
                    for idx_tweet, line in enumerate(f):
                        tweet = json.loads(line, strict=False)
                        texts += tweet['text'] + '\n'
    return texts


def html_decode(s):
    """
    Returns the ASCII decoded version of the given HTML string. This does
    NOT remove normal HTML tags like <p>.
    """
    htmlCodes = (
        ("'", '&#39;'),
        ('"', '&quot;'),
        ('>', '&gt;'),
        ('<', '&lt;'),
        ('&', '&amp;')
    )
    for code in htmlCodes:
        s = s.replace(code[1], code[0])
    return s


def make_input(texts):
    texts = texts[:-1]
    texts = html_decode(texts)
    with open(INPUT, 'w', encoding='utf8') as f:
        f.write(texts)
    f.close()


def make_tweet_sentencepiece(data):
    if data == 'train':
        start_date = train_date
        end_date = valid_date
    elif data == 'valid':
        start_date = valid_date
        end_date = test_date

    if DATA == 'acl18':
        max_len = 180
    elif DATA == 'cikm18':
        max_len = 129
    elif DATA == 'cikm21':
        max_len = 229
    df = pd.DataFrame(
        columns={'id_list', 'text_len', 'idx_stock_position', 'up_down_label',
                 'stock_label'})
    stock_list = ['$' + stock.lower() for stock in STOCK_LIST]
    stock_id_list = sp.piece_to_id(stock_list)
    for stock in STOCK_LIST:
        stock_label = STOCK_LIST.index(stock)
        stock_path = os.path.join(DATA_PATH, DATA, 'tweet', stock)
        date_list = sorted(os.listdir(stock_path))
        # date_list = ['2017-01-01', '2017-01-02', '2017-01-03', ... ,]
        df_stock_price = pd.read_csv(
            os.path.join(DATA_PATH, DATA, 'price', stock + '.csv'))
        for date in date_list:
            if date >= start_date and date < end_date:
                up_down_label = \
                    df_stock_price[df_stock_price['date'] > date].iloc[0][
                        'label']
                with open(os.path.join(stock_path, date), 'r',
                          encoding='utf8') as f:
                    for idx_tweet, line in enumerate(f):
                        tweet = json.loads(line, strict=False)
                        ids_list = sp.encode_as_ids(tweet['text'])

                        if sp.piece_to_id('$' + stock.lower()) not in ids_list:
                            continue

                        idx_stock_position = ids_list.index(
                            sp.piece_to_id('$' + stock.lower()))
                        text_len = len(ids_list)
                        ids_list += [3] * (max_len - len(ids_list))
                        df = df.append(pd.DataFrame([[ids_list, text_len,
                                                      idx_stock_position,
                                                      up_down_label,
                                                      stock_label]],
                                                    columns=['id_list',
                                                             'text_len',
                                                             'idx_stock_position',
                                                             'up_down_label',
                                                             'stock_label']))
    with open(os.path.join(GENERAL_PATH, 'sentence_piece',
                           '{}_{}.pkl'.format(DATA, data)), 'wb') as f:
        pickle.dump((df, stock_id_list), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='data', type=str,
                        default='acl18')
    parser.add_argument('-v', '--vocab_size', help='vocab_size', type=int,
                        default=16000)
    args = parser.parse_args()
    DATA = args.data
    GENERAL_PATH = os.path.dirname(os.getcwd())
    DATA_PATH = os.path.join(GENERAL_PATH, 'data')
    INPUT = os.path.join(GENERAL_PATH, 'sentence_piece',
                         '{}_input.txt'.format(DATA))
    VOCAB_SIZE = args.vocab_size
    USER_DEFINED_SYMBOLS = '<s>,</s>,<pad>,<mask>,rt,AT_USER,URL'
    train_date, valid_date, test_date, end_date, STOCK_LIST, \
    USER_DEFINED_SYMBOLS = get_info(DATA, USER_DEFINED_SYMBOLS)
    texts = get_text(DATA_PATH, DATA)
    make_input(texts)
    # train sentencepiece model from `input.txt` and makes `m.model` and `m.vocab`
    # `m.vocab` is just a reference. not used in the segmentation.
    spm.SentencePieceTrainer.train(
        '--input={} --model_prefix={} --vocab_size={} --user_defined_symbols={}'.format(
            INPUT,
            os.path.join(GENERAL_PATH, 'sentence_piece',
                         '{}'.format(DATA)), VOCAB_SIZE, USER_DEFINED_SYMBOLS))
    # makes segmenter instance and loads the model file (m.model)
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(GENERAL_PATH, 'sentence_piece',
                         '{}.model'.format(DATA)))
    max_len = 0
    for stock in STOCK_LIST:
        stock_path = os.path.join(DATA_PATH, DATA, 'tweet', stock)
        date_list = sorted(os.listdir(stock_path))
        for date in date_list:
            if date >= train_date and date <= end_date:
                with open(os.path.join(stock_path, date), 'r',
                          encoding='utf8') as f:
                    for idx_tweet, line in enumerate(f):
                        tweet = json.loads(line, strict=False)
                        ids_list = sp.encode_as_ids(tweet['text'])
                        if len(ids_list) > max_len:
                            max_len = len(ids_list)
    make_tweet_sentencepiece('train')
    make_tweet_sentencepiece('valid')
