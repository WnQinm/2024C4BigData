from data_provider.data_loader import Dataset_Meteorology
from torch.utils.data import DataLoader


def data_provider(args):
    shuffle_flag = True
    drop_last = True
    batch_size = args.batch_size

    # TODO 划分训练集测试集
    data_set = Dataset_Meteorology(
        root_path=args.root_path,
        data_path=args.data_path,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
