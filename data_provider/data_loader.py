import os
import numpy as np
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

class Dataset_Meteorology(Dataset):
    def __init__(self, root_path, data_path, size=None, features='MS'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.features = features
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.stations_num = self.data.shape[1]
        self.tot_len = len(self.data) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        # (T, S, 2)
        self.data = np.concatenate((np.load(os.path.join(self.root_path, self.data_path[0])),
                                    np.load(os.path.join(self.root_path, self.data_path[1]))), axis=-1)

        era5 = np.load(os.path.join(self.root_path, 'global_data.npy'))
        repeat_era5 = np.repeat(era5, 3, axis=0)[:len(self.data), :, :, :] # (T, 4, 9, S)
        repeat_era5 = repeat_era5.reshape(repeat_era5.shape[0], -1, repeat_era5.shape[3]) # (T, 36, S)
        self.covariate = repeat_era5

    def __getitem__(self, index):
        station_id = index // self.tot_len
        input_begin = index % self.tot_len
        input_end = input_begin + self.seq_len
        target_begin = input_end - 1
        target_end = target_begin + self.pred_len

        seq_x = self.data[input_begin:input_end, station_id, :]
        seq_y = self.data[target_begin:target_end, station_id, :]

        seq_x = np.concatenate([self.covariate[input_begin:input_end, :, station_id], seq_x], axis=1)
        return seq_x, seq_y

    def __len__(self):
        # index 从0到len(dataloader) 对应于
        # station_1(时序块1, 时序块2, ..., 时序块tot_len), ..., station_n(时序块1, 时序块2, ..., 时序块tot_len)
        return self.tot_len * self.stations_num