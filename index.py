import os
import numpy as np
import random
import torch
from models import iTransformer
# from config import MockPath

def invoke(inputs):

    save_path = '/home/mw/project'

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = {
        'model_id': 'v1',
        'model': 'iTransformer',
        'data': 'Meteorology',
        'features': 'M',
        'checkpoints': './checkpoints/',
        'seq_len': 168,
        'label_len': 2,
        'pred_len': 24,
        'd_model': 1024,
        'n_heads': 8,
        'e_layers': 16,
        'd_ff': 1024,
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
    data = data.transpose(0, 1, 3, 2) # (N, L, 1, S)
    C = C1 + 2
    data = np.concatenate([covariate, data], axis=2) # (N, L, C, S)
    data = data.transpose(0, 3, 1, 2) # (N, S, L, C)
    data = data.reshape(N * S, L, C)
    data = torch.tensor(data).float().cuda() # (N * S, L, C)

    model = iTransformer.Model(args).cuda()
    model.load_state_dict(torch.load("/home/mw/project/checkpoints/weight.pth"))

    # (N * S, P, 2)
    outputs = []
    with torch.no_grad():
        for i in range(N):
            outputs.append(model(data[i:i+S, ...])[:, :, -2:].detach().cpu().numpy())
    outputs = np.concatenate(outputs, axis=0)

    P = outputs.shape[1]
    forecast = outputs.reshape(N, S, P, 2) # (N, S, P, 2)
    forecast = forecast.transpose(0, 2, 1, 3) # (N, P, S, 2)

    np.save(os.path.join(save_path, "wind_predict.npy"), forecast[..., 0:1])
    np.save(os.path.join(save_path, "temp_predict.npy"), forecast[..., 1:])
