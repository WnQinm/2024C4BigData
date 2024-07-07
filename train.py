import argparse
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np
from typing import List


if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Demo')

    # basic config
    parser.add_argument('--model_id', type=str, default='v1', help='model id')
    parser.add_argument('--model', type=str, default='iTransformer', help='iTransformer')

    # data loader
    parser.add_argument('--data', type=str, default='Meteorology', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=List[str], default=["wind.npy", "temp.npy"], help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='save path of model checkpoints')
    parser.add_argument('--ckpt_path', type=str, default="./checkpoints/v1_iTransformer_Meteorology_ftM_sl168_ll2_pl24_dm1024_nh8_el16_df1024_test/checkpoint_0_best.pth",
                        help='set a path to load model or set None to create a new model')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=168, help='input sequence length')
    # 同时预测风速和温度 (wind, temp)
    parser.add_argument('--label_len', type=int, default=2, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')

    # 模型结构
    parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=16, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', type=bool, default=False, help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--eval_step', type=int, default=5, help='How often the model is evaluated and saved, and the data to the right of the progress bar is updated')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    # TODO 可以自定义学习率调度 adjust_learning_rate
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=True)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    if torch.cuda.is_available() and args.use_gpu:
        args.use_gpu = True
    else:
        args.use_gpu = False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    for ii in range(args.itr):
        # setting record of experiments
        exp = Exp_Long_Term_Forecast(args)  # set experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_ff,
            args.des, ii)

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)