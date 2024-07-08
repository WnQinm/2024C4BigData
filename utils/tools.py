import math

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print('Updating learning rate to {}'.format(lr))

class Scheduler:
    args = None
    epoch_lr = None
    step_lr = None
    optimizer = None

    @classmethod
    def _init_param(cls, args, optimizer):
        cls.args = args
        cls.epoch_lr = args.learning_rate
        cls.step_lr = cls.epoch_lr
        cls.optimizer = optimizer

    @classmethod
    def epoch_scheduler(cls, *, epoch=None):
        lradj = cls.args.epoch_lradj

        if epoch is None:
            if lradj == 'type1':
                cls.epoch_lr /= 2
        else:
            if lradj == 'type1':
                cls.epoch_lr = cls.args.learning_rate * (0.5 ** ((epoch - 1) // 1))
            elif lradj == "cosine":
                cls.epoch_lr = cls.args.learning_rate /2 * (1 + math.cos(epoch / cls.args.train_epochs * math.pi))

        for param_group in cls.optimizer.param_groups:
            param_group['lr'] = cls.epoch_lr

    @classmethod
    def step_scheduler(cls, *, step=None):
        pass