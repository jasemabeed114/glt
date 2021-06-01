from sklearn.metrics import matthews_corrcoef
import torch
import numpy as np

def get_date_info(data):
    if data == 'cikm18':
        train_date = '2017-01-03'
        valid_date = '2017-09-01'
        test_date = '2017-11-01'
        end_date = '2017-12-29'
    elif data == 'acl18':
        train_date = '2014-01-02'
        valid_date = '2015-08-03'
        test_date = '2015-10-01'
        end_date = '2015-12-31'
    elif data == 'cikm21':
        train_date = '2019-07-05'
        valid_date = '2020-03-02'
        test_date = '2020-05-01'
        end_date = '2020-07-02'
    else:
        raise ValueError

    return train_date, valid_date, test_date, end_date

def get_date_index(df_stock_price, train_date, valid_date, test_date, end_date,
                   WINDOW_SIZE_FOR_PRICE_FEATURE):
    train_date_idx = df_stock_price[df_stock_price['date'] == train_date]
    train_date_idx = int(train_date_idx.index[0])
    train_date_idx += (WINDOW_SIZE_FOR_PRICE_FEATURE - 1)

    valid_date_idx = df_stock_price[df_stock_price['date'] == valid_date]
    valid_date_idx = int(valid_date_idx.index[0])

    test_date_idx = df_stock_price[df_stock_price['date'] == test_date]
    test_date_idx = int(test_date_idx.index[0])

    end_date_idx = df_stock_price[df_stock_price['date'] == end_date]
    end_date_idx = int(end_date_idx.index[0])

    return train_date_idx, valid_date_idx, test_date_idx, end_date_idx

def shuffle(array, seed_num):
    np.random.seed(seed_num)
    np.random.shuffle(array)
    return array

def binary_acc(y_pred, y_true):
    y_pred_tag = ((y_pred.detach().numpy() > 0) * 2 - 1)
    correct_results_sum = (y_pred_tag == y_true.numpy()).sum()
    acc = correct_results_sum / y_true.shape[0]
    return acc

def mcc(y_pred, y_true):
    y_pred = y_pred.detach().numpy()
    y_pred = ((y_pred > 0) * 2 - 1)
    y_true = y_true.detach().numpy()
    return matthews_corrcoef(y_true, y_pred)

def hinge_loss(output, target, MODEL_DEVICE):
    tensor_zeros = torch.Tensor([[0]] * target.shape[0])
    tensor_zeros = tensor_zeros.to(MODEL_DEVICE)
    tensor_concat = torch.cat((tensor_zeros, 1 - target * output), dim=1)
    tensor_max = torch.max(tensor_concat, 1)
    loss = torch.mean(tensor_max.values)
    return loss