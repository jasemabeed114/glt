import os
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import copy
import datetime
import json
import argparse
import sentencepiece as spm
from represent_tweet_stock import Model
from sklearn.metrics import accuracy_score, matthews_corrcoef
from util import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings

warnings.simplefilter("ignore")
torch.set_num_threads(1)


def make_trend_feature(model_path, train_date_idx, end_date_idx,
                       df_stock_price, stock_csv_file_list, tweet_path,
                       price_path, global_trend=1, local_trend=1):
    stock_tweet_emb_dic_g = defaultdict(dict)
    stock_tweet_emb_dic_l = defaultdict(dict)

    # max_len of tweets
    if DATA == 'acl18':
        max_len = 180
    elif DATA == 'cikm18':
        max_len = 129
    elif DATA == 'cikm21':
        max_len = 229

    classification_model = torch.load(model_path, map_location=MODEL_DEVICE)
    classification_model = classification_model.double()
    classification_model.to(MODEL_DEVICE)
    classification_model.eval()
    stock_class_emb = classification_model. \
        fc_stock_label.weight.detach().numpy()
    sp_model_path = os.path.join(GENERAL_PATH, 'sentence_piece',
                                 '{}'.format(DATA))
    sp = spm.SentencePieceProcessor()
    sp.load('{}.model'.format(sp_model_path))

    for idx in range(train_date_idx + 1, end_date_idx + 1):
        cur_date = df_stock_price.iloc[idx]['date']
        cur_date = datetime.datetime.strptime(cur_date, '%Y-%m-%d').date()
        c = 0
        date = cur_date
        df = pd.DataFrame(
            columns={'id_list', 'text_len', 'idx_stock_position', })
        while c < 1:
            date -= datetime.timedelta(days=1)
            for idx_stock_, stock_csv_file_ in enumerate(
                    stock_csv_file_list[:]):
                stock_tweet_path = os.path.join(tweet_path, stock_csv_file_[:4])
                stock_date_tweet_path = os.path.join(stock_tweet_path,
                                                     str(date))

                if os.path.exists(stock_date_tweet_path):
                    with open(stock_date_tweet_path, 'r') as f:
                        for idx_tweet, line in enumerate(f):
                            tweet = json.loads(line, strict=False)
                            ids_list = sp.encode_as_ids(tweet['text'])
                            if sp.piece_to_id(
                                    '$' + stock_csv_file_[
                                          :-4].lower()) not in ids_list:
                                continue
                            idx_stock_position = ids_list.index(
                                sp.piece_to_id(
                                    '$' + stock_csv_file_[:-4].lower()))
                            text_len = len(ids_list)
                            ids_list += [3] * (max_len - len(ids_list))
                            df = df.append(pd.DataFrame([[ids_list,
                                                          text_len,
                                                          idx_stock_position, ]],
                                                        columns=['id_list',
                                                                 'text_len',
                                                                 'idx_stock_position', ]))

                    c += idx_tweet + 1
        id_list = df['id_list'].values
        id_list = list(id_list)
        id_list = np.array(id_list)
        text_len = df['text_len'].values.astype('int64')
        idx_stock_position = df['idx_stock_position'].values.astype('int64')
        id_list = torch.tensor(id_list)
        idx_stock_position = torch.tensor(idx_stock_position)
        text_len = torch.tensor(text_len)
        id_list = id_list.to(MODEL_DEVICE)
        idx_stock_position = idx_stock_position.to(MODEL_DEVICE)
        text_len = text_len.to(MODEL_DEVICE)
        _, classification_model_output = classification_model(None, id_list,
                                                              text_len,
                                                              idx_stock_position)

        if global_trend:
            classification_model_output_mean = classification_model_output. \
                detach().numpy().mean(axis=0)
            elementwise_prod = stock_class_emb * classification_model_output_mean
            dot_prod = elementwise_prod.sum(axis=1)
            softmax = np.exp(dot_prod) / sum(np.exp(dot_prod))
            total_price_feature = np.array([])
            total_price_feature = total_price_feature.reshape(0, 11)
            for idx_stock, stock_csv_file in enumerate(stock_csv_file_list[:]):
                stock_csv_file_path = os.path.join(price_path, stock_csv_file)
                df_stock_price_ = pd.read_csv(stock_csv_file_path)
                price_feature = df_stock_price_.iloc[idx - 1, 1:12].values
                if -123321 in price_feature:
                    price_feature = np.array([0] * 11)
                price_feature = price_feature[np.newaxis, :]
                total_price_feature = np.concatenate([total_price_feature,
                                                      price_feature], axis=0)
            stock_emb = np.matmul(softmax, total_price_feature)
            for idx_stock, stock_csv_file in enumerate(stock_csv_file_list[:]):
                stock_tweet_emb_dic_g[idx][idx_stock] = stock_emb

        if local_trend:
            classification_model_output = classification_model_output.detach().numpy()
            total_price_feature = np.array([])
            total_price_feature = total_price_feature.reshape(0, 11)
            for idx_stock, stock_csv_file in enumerate(stock_csv_file_list[:]):
                stock_csv_file_path = os.path.join(price_path, stock_csv_file)
                df_stock_price_ = pd.read_csv(stock_csv_file_path)
                price_feature = df_stock_price_.iloc[idx - 1, 1:12].values
                if -123321 in price_feature:
                    price_feature = np.array([0] * 11)
                price_feature = price_feature[np.newaxis, :]
                total_price_feature = np.concatenate([total_price_feature,
                                                      price_feature], axis=0)
            for idx_stock, stock_csv_file in enumerate(stock_csv_file_list[:]):
                elementwise_prod_ = stock_class_emb[idx_stock] * \
                                    classification_model_output
                dot_prod_ = elementwise_prod_.sum(axis=1)
                softmax_ = np.exp(dot_prod_) / sum(np.exp(dot_prod_))
                target_tweet_emb = np.matmul(softmax_,
                                             classification_model_output)

                elementwise_prod = stock_class_emb * target_tweet_emb
                dot_prod = elementwise_prod.sum(axis=1)
                softmax = np.exp(dot_prod) / sum(np.exp(dot_prod))
                stock_emb = np.matmul(softmax, total_price_feature)
                stock_tweet_emb_dic_l[idx][idx_stock] = stock_emb

    if global_trend and local_trend:
        stock_tweet_emb_dic = defaultdict(dict)
        for idx in range(train_date_idx + 1, end_date_idx + 1):
            for idx_stock, stock_csv_file in enumerate(stock_csv_file_list[:]):
                stock_tweet_emb_dic[idx][idx_stock] = \
                    np.concatenate([stock_tweet_emb_dic_g[idx][idx_stock],
                                    stock_tweet_emb_dic_l[idx][idx_stock]])
        return stock_tweet_emb_dic
    elif global_trend:
        return stock_tweet_emb_dic_g
    elif local_trend:
        return stock_tweet_emb_dic_l
    else:
        raise ValueError


def load_data(data, seq_len=2, global_trend=1, local_trend=1):
    preprocessed_data_path = \
        os.path.join(os.path.dirname(os.getcwd()),
                     'temp',
                     'isGlobal_{}_isLocal_{}_data_{}_seqLen_{}.pkl' \
                     .format(global_trend, local_trend, data, seq_len))
    if os.path.exists(preprocessed_data_path):
        with open(preprocessed_data_path, 'rb') as f:
            x_tech_train, y_train, x_tech_valid, y_valid, x_tech_test, y_test \
                = pickle.load(f)
        return x_tech_train, y_train, x_tech_valid, y_valid, x_tech_test, y_test

    class_model_path = os.path.join(GENERAL_PATH, 'tweet_model',
                                    '{}.model'.format(data))

    data_path = os.path.join(GENERAL_DATA_PATH, data)
    price_path = os.path.join(data_path, 'price')
    tweet_path = os.path.join(data_path, 'tweet')

    train_date, valid_date, test_date, end_date = get_date_info(data)

    stock_csv_file_list = sorted(os.listdir(price_path))
    stock_csv_file_list = [file for file in stock_csv_file_list if 'csv' in
                           file]
    stock_csv_file_list = stock_csv_file_list[:]
    stock_csv_file_path = os.path.join(price_path, stock_csv_file_list[0])
    df_stock_price = pd.read_csv(stock_csv_file_path)

    train_date_idx, valid_date_idx, test_date_idx, end_date_idx = \
        get_date_index(df_stock_price, train_date,
                       valid_date, test_date, end_date,
                       30)
    dic_df_stock = {}
    for stock_csv_file in stock_csv_file_list:
        stock_csv_file_path = os.path.join(price_path, stock_csv_file)
        df_stock_price = pd.read_csv(stock_csv_file_path)
        stock = stock_csv_file[:-4]
        dic_df_stock[stock] = df_stock_price

    if global_trend and local_trend:
        n_column = 33
    else:
        n_column = 22

    stock_tweet_emb_dic = make_trend_feature(class_model_path,
                                             train_date_idx,
                                             end_date_idx,
                                             df_stock_price,
                                             stock_csv_file_list,
                                             tweet_path,
                                             price_path,
                                             global_trend,
                                             local_trend)
    x_tech_train = np.array([])
    x_tech_train = x_tech_train.reshape(0, seq_len, n_column)
    x_tech_valid, x_tech_test = copy.deepcopy(x_tech_train), \
                                copy.deepcopy(x_tech_train)
    y_train = np.array([])
    y_train = y_train.reshape(0, 1)
    y_valid, y_test = copy.deepcopy(y_train), copy.deepcopy(y_train)
    for idx_stock, stock_csv_file in enumerate(stock_csv_file_list[:]):
        stock_csv_file_path = os.path.join(price_path, stock_csv_file)
        df_stock_price = pd.read_csv(stock_csv_file_path)
        for idx in range(train_date_idx + seq_len, end_date_idx + 1):
            y = df_stock_price.iloc[idx, 12]
            if y == 0:
                continue
            y = int(y > 0)
            y = np.array([[y]])
            price_feature = \
                df_stock_price.iloc[idx - seq_len:idx, 1:12].values
            if -123321 in price_feature:
                continue
            tweet_feauture = np.array([stock_tweet_emb_dic[idx][idx_stock]])
            for idx_tweet in range(1, seq_len):
                tweet_feauture = np.concatenate([tweet_feauture,
                                                 np.array([stock_tweet_emb_dic
                                                           [idx - idx_tweet]
                                                           [idx_stock]])],
                                                axis=0)
            tweet_feauture = tweet_feauture[::-1]
            tech_feature = np.concatenate([price_feature,
                                           tweet_feauture], axis=1)
            cur_date = df_stock_price.iloc[idx]['date']

            if cur_date < valid_date:
                x_tech_train = np.concatenate(
                    [x_tech_train, tech_feature[np.newaxis, :, :]], axis=0)
                y_train = np.concatenate([y_train, y], axis=0)
            elif cur_date < test_date:
                x_tech_valid = np.concatenate(
                    [x_tech_valid, tech_feature[np.newaxis, :, :]], axis=0)
                y_valid = np.concatenate([y_valid, y], axis=0)
            elif cur_date <= end_date:
                x_tech_test = np.concatenate(
                    [x_tech_test, tech_feature[np.newaxis, :, :]], axis=0)
                y_test = np.concatenate([y_test, y], axis=0)

    with open(preprocessed_data_path, 'wb') as f:
        pickle.dump([x_tech_train, y_train, x_tech_valid, y_valid,
                     x_tech_test, y_test, ], f)

    return x_tech_train, y_train, x_tech_valid, y_valid, x_tech_test, y_test


class MainModel(nn.Module):
    def __init__(self,
                 hidden_dim=None,
                 tech_fea_dim=None,
                 lstm_num_layers=1,
                 dropout=0.0,
                 l2_norm_=None,
                 ):
        super(MainModel, self).__init__()
        self.tech_att_fea_dim = hidden_dim
        self.tech_fc_fea_dim = hidden_dim
        self.l2_norm_ = l2_norm_

        # architecture for tech feature
        # if batch_first=True,
        # lstm input tensors are provided as (batch, seq, feature)
        self.tech_fc = nn.Linear(tech_fea_dim, tech_fea_dim)
        self.tech_lstm = nn.LSTM(input_size=tech_fea_dim,
                                 hidden_size=hidden_dim,
                                 num_layers=lstm_num_layers,
                                 batch_first=True,
                                 dropout=dropout)

        self.tech_attn = nn.Linear(hidden_dim, self.tech_att_fea_dim)
        self.tech_attn2 = nn.Linear(self.tech_att_fea_dim, 1, bias=False)
        self.tech_fc_no_attention_input = nn.Linear(hidden_dim, 1)
        self.tech_fc_attention_input = nn.Linear(2 * hidden_dim, 1)
        torch.nn.init.zeros_(self.tech_attn.bias)
        torch.nn.init.zeros_(list(self.tech_lstm.parameters())[2])
        torch.nn.init.zeros_(list(self.tech_lstm.parameters())[3])
        for layer in [self.tech_fc, self.tech_attn, self.tech_attn2,
                      self.tech_fc_attention_input, ]:
            torch.nn.init.xavier_uniform(layer.weight)
        self.softmax = nn.Softmax(dim=1)

    def l2_norm(self):
        l2_norm = 0
        weight_list = [self.tech_fc_attention_input.weight,
                       self.tech_fc_attention_input.bias]
        for weight in weight_list:
            l2_norm += torch.sum(weight * weight)
        return l2_norm / 2

    def forward(self, tech_feature=None, y=None):
        '''

        :param tech_feature: batch_size x step_size x feature_dim
        :param tweet_feature: batch_size x step_size (tweet) x feature_dim (tweet)
        :return: batch_size x 1
        '''
        input_tech_lstm = self.tech_fc(tech_feature)
        input_tech_lstm = torch.tanh(input_tech_lstm)
        tech_hid, (hn, cn) = self.tech_lstm(input_tech_lstm)
        tech_hid_last = tech_hid[:, -1, :]
        tech_out_att = self.tech_attn(tech_hid)
        tech_out_att = torch.tanh(tech_out_att)
        tech_out_att = self.tech_attn2(tech_out_att)
        tech_out_att = self.softmax(tech_out_att)
        tech_out_att = torch.transpose(tech_out_att, 1, 2)
        tech_att_value = torch.bmm(tech_out_att, tech_hid)
        tech_att_value = torch.squeeze(tech_att_value)
        fea_con = torch.cat((tech_hid_last, tech_att_value), 1)
        tech_out = self.tech_fc_attention_input(fea_con)
        if y == None:
            return tech_out, None
        hig_loss = hinge_loss(tech_out, y, MODEL_DEVICE)
        loss = hig_loss + self.l2_norm_ * self.l2_norm()
        return tech_out, loss


def main(global_trend, local_trend, step_size, hidden_dim, l2_norm, lr,
         fix_seed):
    if fix_seed:
        seed_num = 123
    else:
        seed_num = np.random.randint(999)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    device = MODEL_DEVICE

    lstm_num_layers = 1
    batch_size = 1024

    x_tech_train, y_train, x_tech_valid, y_valid, x_tech_test, y_test = \
        load_data(DATA, step_size, global_trend, local_trend)

    x_tech_train = shuffle(x_tech_train, seed_num)
    y_train = shuffle(y_train, seed_num)

    x_tech_train = x_tech_train.astype('float64')
    x_tech_valid = x_tech_valid.astype('float64')
    x_tech_test = x_tech_test.astype('float64')
    x_tech_train = torch.tensor(x_tech_train)
    y_train = torch.tensor(y_train).double()
    x_tech_valid = torch.tensor(x_tech_valid)
    y_valid = torch.tensor(y_valid).double()
    x_tech_test = torch.tensor(x_tech_test)
    y_test = torch.tensor(y_test).double()

    x_tech_train = x_tech_train.to(device)
    y_train = y_train.to(device)
    x_tech_valid = x_tech_valid.to(device)
    y_valid = y_valid.to(device)
    x_tech_test = x_tech_test.to(device)
    y_test = y_test.to(device)

    model = MainModel(tech_fea_dim=x_tech_train.shape[2],
                      hidden_dim=hidden_dim,
                      lstm_num_layers=lstm_num_layers,
                      l2_norm_=l2_norm,
                      )
    model = model.double()
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           eps=1e-07)
    model.to(device)
    model.train()
    y_train, y_valid, y_test = (y_train * 2) - 1, (y_valid * 2) - 1, \
                               (y_test * 2) - 1
    valid_acu = 0
    for epoch in range(n_epoch):
        total_loss = 0
        model.train()
        for batch in DataLoader(range(len(y_train)), batch_size=batch_size):
            model.zero_grad()
            batch_tech = x_tech_train[batch]
            loss = model(batch_tech, y_train[batch])[1]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        model.eval()

        y_pred_valid = model(x_tech_valid)[0].flatten()
        y_pred_test = model(x_tech_test)[0].flatten()
        mcc(y_pred_valid.cpu(), y_valid.flatten().cpu())
        if binary_acc(y_pred_valid.cpu(), y_valid.flatten().cpu()) > valid_acu:
            valid_acu = binary_acc(y_pred_valid.cpu(), y_valid.flatten().cpu())
            print(str(binary_acc(y_pred_valid.cpu(), y_valid.flatten().cpu()))
                  + '\t'
                  + str(mcc(y_pred_valid.cpu(), y_valid.flatten().cpu()))
                  + '\t'
                  + str(binary_acc(y_pred_test.cpu(), y_test.flatten().cpu()))
                  + '\t'
                  + str(mcc(y_pred_test.cpu(), y_test.flatten().cpu())))
        losses.append(total_loss)
    print('Finish!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='data', type=str,
                        default='acl18')
    parser.add_argument('-g', '--global_trend', help='use global tweet trend',
                        type=int, default=1)
    parser.add_argument('-l', '--local_trend', help='use local tweet trend',
                        type=int, default=1)
    parser.add_argument('-s', '--step', help='lag size',
                        type=int, default=5)
    parser.add_argument('-c', '--hidden', help='the size of hidden state',
                        type=int, default=4)
    parser.add_argument('-a', '--l2norm', help='l2 norm',
                        type=float, default=0.01)
    parser.add_argument('-b', '--lr', help='learning rate',
                        type=float, default=0.01)
    parser.add_argument('-f', '--fix_seed', help='fix seed',
                        type=int, default=0)

    args = parser.parse_args()
    DATA = args.data
    GENERAL_PATH = os.path.dirname(os.getcwd())
    GENERAL_DATA_PATH = os.path.join(GENERAL_PATH, 'data')
    MODEL_DEVICE = "cpu"
    # MODEL_DEVICE = "cuda:1"
    n_epoch = 150
    global_trend = args.global_trend
    local_trend = args.local_trend
    step_size = args.step
    hidden_dim = args.hidden
    l2_norm = args.l2norm
    lr = args.lr
    fix_seed = args.fix_seed
    main(global_trend, local_trend, step_size, hidden_dim, l2_norm, lr, fix_seed)
