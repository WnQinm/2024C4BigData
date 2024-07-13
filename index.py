import os
import numpy as np
import random
import torch
from models import iTransformer

def invoke(inputs):

    save_path = '/home/mw/project'

    fix_seed = 0
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = {
        'checkpoints': './checkpoints/',
        'seq_len': 168,
        'label_len': 1,
        'pred_len': 24,
        'enc_in': 38,
        'd_model': 64,
        'n_heads': 1,
        'e_layers': 1,
        'd_ff': 64,
        'dropout': 0.1,
        'activation': 'gelu',
        'output_attention': False
    }

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    args = Struct(**args)

    test_data_root_path = inputs

    # (N, L, S, 2)
    data = np.concatenate((np.load(os.path.join(test_data_root_path, "wind_lookback.npy")),
                           np.load(os.path.join(test_data_root_path, "temp_lookback.npy"))), axis=-1)
    N, L, S, _ = data.shape # 72, 168, 60

    cenn_era5_data = np.load(os.path.join(test_data_root_path, "cenn_data.npy"))

    repeat_era5 = np.repeat(cenn_era5_data, 3, axis=1) # (N, L, 4, 9, S)
    C1 = repeat_era5.shape[2] * repeat_era5.shape[3]
    covariate = repeat_era5.reshape(repeat_era5.shape[0], repeat_era5.shape[1], -1, repeat_era5.shape[4]) # (N, L, C1, S)
    data = data.transpose(0, 1, 3, 2) # (N, L, 2, S)
    C = C1 + 2
    data = np.concatenate([covariate, data], axis=2) # (N, L, C, S)
    data = data.transpose(0, 3, 1, 2) # (N, S, L, C)
    data = data.reshape(N * S, L, C)
    data = torch.tensor(data).float().cuda() # (N * S, L, C)

    wind_model = iTransformer.Model(args).cuda()
    wind_model.load_state_dict(torch.load("/home/mw/project/checkpoints/wind.pth"))
    temp_model = iTransformer.Model(args).cuda()
    temp_model.load_state_dict(torch.load("/home/mw/project/checkpoints/temp.pth"))

    # (N * S, P, 2)
    with torch.no_grad():
        wind_pred = wind_model(data)[:, :, -2:-1].detach().cpu().numpy()
        temp_pred = temp_model(data)[:, :, -1:].detach().cpu().numpy()

    P = args.pred_len
    wind_pred = wind_pred.reshape(N, S, P, 1) # (N, S, P, 1)
    wind_pred = wind_pred.transpose(0, 2, 1, 3) # (N, P, S, 1)
    temp_pred = temp_pred.reshape(N, S, P, 1) # (N, S, P, 1)
    temp_pred = temp_pred.transpose(0, 2, 1, 3) # (N, P, S, 1)

    np.save(os.path.join(save_path, "wind_predict.npy"), wind_pred)
    np.save(os.path.join(save_path, "temp_predict.npy"), temp_pred)
