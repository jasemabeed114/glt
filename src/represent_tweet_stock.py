import os
import pickle
import time
import numpy as np
import pandas as pd
import argparse
from operator import add
import random
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from util import *

import warnings

warnings.simplefilter("ignore")
torch.set_num_threads(1)


def load_data(train_or_valid):
    preprocessed_data_path = \
        os.path.join('../sentence_piece/{}_{}.pkl'.format(DATA, train_or_valid))
    with open(preprocessed_data_path, 'rb') as f:
        df, stock_id_list = pickle.load(f)
    df = df.loc[(df['up_down_label'] == -1) | (df['up_down_label'] == 1)]
    df.reset_index(drop=True, inplace=True)
    x_train = df['id_list'].values
    x_train = list(x_train)
    x_train = np.array(x_train)
    x_len_train = df['text_len'].values.astype('int64')
    x_idxStock_train = df['idx_stock_position'].values.astype('int64')
    y_stock_train = df['stock_label'].values.astype('int64')
    return x_train, x_len_train, x_idxStock_train, y_stock_train, stock_id_list


class Model(nn.Module):
    def __init__(self,
                 embedding_dim=None,
                 lstm_dim=None,
                 lstm_num_layers=1,
                 dropout=0.0,
                 ):
        super(Model, self).__init__()
        self.vocab_size = 16000
        self.num_class = NUM_CLASS
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim,
                                      padding_idx=3)
        # architecture for tech feature
        # if batch_first=True,
        # lstm input tensors are provided as (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=lstm_dim,
                            num_layers=lstm_num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)
        self.fc_stock_label = nn.Linear(2 * lstm_dim, self.num_class,
                                        bias=False)
        self.fc1 = nn.Linear(2 * lstm_dim, lstm_dim)
        self.fc2 = nn.Linear(lstm_dim, 1)
        self.leakyRelu = nn.LeakyReLU()
        for layer in [self.fc_stock_label, self.fc1, self.fc2, ]:
            torch.nn.init.xavier_uniform(layer.weight)

    def forward(self, x_masked=None, x=None,
                tweet_len=None, idx_stock_position=None):
        if x_masked != None:
            embedded = self.embedding(x_masked)
            packed_embedded = pack_padded_sequence(embedded, tweet_len,
                                                   batch_first=True,
                                                   enforce_sorted=False)
            output, (h_n, c_n) = self.lstm(packed_embedded)
            output_unpacked, output_lengths = pad_packed_sequence(output,
                                                                  batch_first=True)
            out = output_unpacked[range(output_unpacked.shape[0]),
                  idx_stock_position, :]
            # torch.Size([10000, 40])
            logit = self.fc_stock_label(out)
            pred_stock_class = logit
        else:
            pred_stock_class = None

        embedded_inference = self.embedding(x)
        packed_embedded_inference = pack_padded_sequence(embedded_inference,
                                                         tweet_len,
                                                         batch_first=True,
                                                         enforce_sorted=False)
        output_inference, (h_n, c_n) = self.lstm(packed_embedded_inference)
        output_unpacked_inference, output_lengths_inference = \
            pad_packed_sequence(output_inference,
                                batch_first=True)
        output_inference_ = output_unpacked_inference[
                     range(output_unpacked_inference.shape[0]),
                     idx_stock_position, :]

        return pred_stock_class, output_inference_


def main():
    seed_num = 0
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    device = MODEL_DEVICE
    lstm_num_layers = 1
    batch_size = BATCH_SIZE

    def raw2tensor(data):
        x, x_len, x_idxStock, y_stock, stock_id_list = load_data(data)
        x = shuffle(x, seed_num)
        x_len = shuffle(x_len, seed_num)
        x_idxStock = shuffle(x_idxStock, seed_num)
        y_stock = shuffle(y_stock, seed_num)

        x_len = torch.tensor(x_len)
        y_stock = torch.tensor(y_stock).long()

        x_len = x_len.to(device)
        y_stock = y_stock.to(device)

        return x, x_len, x_idxStock, y_stock, stock_id_list

    x_train, x_len_train, x_idxStock_train, y_stock_train, stock_id_list = raw2tensor(
        'train')
    x_valid, x_len_valid, x_idxStock_valid, y_stock_valid, _ = raw2tensor(
        'valid')

    def masking(x):
        if MASK_THRESHOLD >= np.random.rand():
            words_list = x['words_list']
            words_list[x['idxStock']] = 4
            # print(sp.piece_to_id('<mask>'))
            # 4
            return words_list
        elif OTHER_THRESHOLD >= np.random.rand():
            words_list = x['words_list']
            cur_stock = words_list[x['idxStock']]
            set_stock = set(stock_id_list) - set([cur_stock])
            li_stock = list(set_stock)
            words_list[x['idxStock']] = random.choice(li_stock)
            return words_list
        else:
            return x['words_list']

    def masking_valid(x):
        words_list = x['words_list']
        words_list[x['idxStock']] = 4
        # print(sp.piece_to_id('<mask>'))
        # 4
        return words_list

    def evaluate_model(data, model, device):
        model.eval()
        if data == 'train':
            y_stock = y_stock_train
            x = x_train
            x_idxStock = x_idxStock_train
            x_len = x_len_train
        elif data == 'valid':
            y_stock = y_stock_valid
            x = x_valid
            x_idxStock = x_idxStock_valid
            x_len = x_len_valid
            total_loss = 0
        for batch in DataLoader(range(len(y_stock)),
                                batch_size=batch_size, ):
            x_batch = x[batch]
            x_idxStock_batch = x_idxStock[batch]
            df = pd.DataFrame({'words_list': x_batch.tolist(),
                               'idxStock': x_idxStock_batch})
            df['words_list'] = df.apply(lambda a: masking_valid(a), axis=1)
            x_masked_batch = np.array(df['words_list'].values.tolist())
            x_masked_batch = torch.tensor(x_masked_batch)
            x_masked_batch = x_masked_batch.to(device)
            x_batch = torch.tensor(x_batch)
            x_batch = x_batch.to(device)
            x_idxStock_batch = torch.tensor(x_idxStock_batch)
            x_idxStock_batch = x_idxStock_batch.to(device)
            pred_stock_class, _ = model(
                x_masked_batch,
                x_batch,
                x_len[batch],
                x_idxStock_batch)
            if data == 'valid':
                loss = ce_loss_function(pred_stock_class, y_stock[batch])
                total_loss += loss.item()

        if data == 'valid':
            return total_loss
        else:
            return None

    model = Model(embedding_dim=100,
                  lstm_dim=LSTM_DIM,
                  lstm_num_layers=lstm_num_layers,
                  )
    model = model.double()
    losses = []
    ce_loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,
                           eps=1e-07)
    model.to(device)
    model.train()
    s_t = time.time()
    valid_loss = 100000000
    path = os.path.join(MODEL_PATH, '{}.model'.format(DATA))
    for epoch in range(n_epoch):
        total_loss = 0
        model.train()
        for batch in DataLoader(range(len(y_stock_train)),
                                batch_size=batch_size, ):
            model.zero_grad()
            x_train_batch = x_train[batch]
            x_idxStock_train_batch = x_idxStock_train[batch]
            df = pd.DataFrame({'words_list': x_train_batch.tolist(),
                               'idxStock': x_idxStock_train_batch})
            df['words_list'] = df.apply(lambda x: masking(x), axis=1)
            x_masked_train_batch = np.array(df['words_list'].values.tolist())
            x_masked_train_batch = x_masked_train_batch.tolist()
            x_masked_train_batch = torch.tensor(x_masked_train_batch)
            x_masked_train_batch = x_masked_train_batch.to(device)
            x_train_batch = torch.tensor(x_train_batch)
            x_train_batch = x_train_batch.to(device)
            x_idxStock_train_batch = torch.tensor(x_idxStock_train_batch)
            x_idxStock_train_batch = x_idxStock_train_batch.to(device)

            pred_stock_class, _ = model(x_masked_train_batch,
                                     x_train_batch,
                                     x_len_train[batch],
                                     x_idxStock_train_batch)
            loss = ce_loss_function(pred_stock_class, y_stock_train[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('total_loss:', total_loss)
        losses.append(total_loss)
        if epoch % VERBOSE == VERBOSE - 1:
            print('Evaluate!')
            evaluate_model('train', model, device)
            cur_valid_loss = evaluate_model('valid', model, device)
            if cur_valid_loss < valid_loss:
                valid_loss = cur_valid_loss
                torch.save(model, path)
                print('tweet_model saved!')
    print(time.time() - s_t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='data', type=str,
                        default='acl18')
    parser.add_argument('-g', '--gpu', help='gpu to use', type=int,
                        default=0)
    parser.add_argument('-l', '--lstm_dim', help='lstm dim', type=int,
                        default=25)
    parser.add_argument('-m', '--mask_threshold', help='mask_threshold',
                        type=float, default=0.8)
    parser.add_argument('-o', '--other_threshold', help='other_threshold',
                        type=float, default=0.1)
    args = parser.parse_args()
    DATA = args.data
    LSTM_DIM = args.lstm_dim
    MASK_THRESHOLD = args.mask_threshold
    OTHER_THRESHOLD = MASK_THRESHOLD + args.other_threshold
    MODEL_PATH = os.path.join(os.path.dirname(os.getcwd()), 'tweet_model')

    if args.gpu == -1:
        MODEL_DEVICE = "cpu"
    elif args.gpu == 0:
        MODEL_DEVICE = "cuda:0"
    elif args.gpu == 1:
        MODEL_DEVICE = "cuda:1"
    elif args.gpu == 2:
        MODEL_DEVICE = "cuda:2"
    elif args.gpu == 3:
        MODEL_DEVICE = "cuda:3"

    VERBOSE = 1
    if DATA == 'acl18':
        NUM_CLASS = 87
    elif DATA == 'cikm18':
        NUM_CLASS = 38
    elif DATA == 'cikm21':
        NUM_CLASS = 50
    BATCH_SIZE = 1250
    n_epoch = 50
    main()
