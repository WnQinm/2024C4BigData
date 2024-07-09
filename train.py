from typing import List
import argparse
from datetime import datetime
import random
import numpy as np
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Demo')

    # basic config
    parser.add_argument('--model_id', type=str, default='v2', help='model id')

    # data loader
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=List[str], default=["wind.npy", "temp.npy"], help='data file')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='save path of model checkpoints')
    parser.add_argument('--ckpt_path', type=str, default="./checkpoints/070918/checkpoint_0_best.pth",
                        help='set a path to load model or set None to create a new model')

    # forecasting task 同时预测风速和温度 (wind, temp)
    parser.add_argument('--seq_len', type=int, default=168, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=2, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--feature_len', type=int, default=38, help='4*9+2 4个协变量 9个格点 2个因变量')

    # 模型结构
    ## iTransformer
    parser.add_argument('--output_attention', type=bool, default=False, help='whether to output attention in ecoder')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--itrm_e_layers', type=int, default=2, help='num of encoder layers')
    ## FAt
    parser.add_argument('--fat_e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--fat_feature_dim', type=int, default=128)
    ## Model
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--lstm_layer_num', type=int, default=2)
    parser.add_argument('--lstm_hidden_size', type=int, default=64)

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=4e-5, help='optimizer learning rate')
    parser.add_argument('--eval_step', type=int, default=10, help='How often the model is evaluated and saved, and the data to the right of the progress bar is updated')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    # TODO 自定义学习率调度
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    # parser.add_argument('--epoch_lradj', type=str, default='type1', help='adjust learning rate')
    # parser.add_argument('--step_lradj', type=str, default='cosine', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    # TODO 自定添加子目录
    parser.add_argument('--tensorboard', type=str, default="/root/tf-logs/bigdata/2", help="Whether to use tensorboard. Set None to disable")
    parser.add_argument('--autosave', type=int, default=5, help="一个epoch中每训练百分之多少iter保存一次权重")

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
        setting = '{}'.format(datetime.now().strftime("%m%d%H"))

        print('start training : {}'.format(setting))
        exp.train(setting)