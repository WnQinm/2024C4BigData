from data_provider.data_loader import Dataset_Meteorology
from torch.utils.data import DataLoader, random_split


def data_provider(args):
    shuffle_flag = True
    drop_last = True
    batch_size = args.batch_size
    eval_batch_size = 8
    eval_ratio = 0.1
    # 训练集的iter数必须小于验证集的iter数
    assert (1-eval_ratio)/batch_size < eval_ratio/eval_batch_size

    data_set = Dataset_Meteorology(
        root_path=args.root_path,
        data_path=args.data_path,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features
    )

    train_size = int(len(data_set) * (1-eval_ratio))
    validate_size = int(len(data_set) * eval_ratio)
    train_dataset, eval_dataset = random_split(data_set, [train_size, validate_size])

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
        )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=eval_batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
        )
    iter(train_dataloader)
    return train_dataloader, eval_dataloader
