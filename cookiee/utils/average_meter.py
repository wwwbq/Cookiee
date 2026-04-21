
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, windows=100,fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.windows = windows
        self.count = []
        self.reset()

    def reset(self):
        self.val = []
        self.count = []

    def update(self, val, n=1):
        self.val.append(val)
        self.count.append(n)

        while len(self.val) > self.windows:
            self.val.pop(0)
            self.count.pop(0)

        # self.val = val
        # self.sum += val * n
        # self.count += n
        # self.avg = self.sum / self.count
    
    @property
    def average(self):
        return sum([val/cnt for val, cnt in zip(self.val, self.count)]) / max(len(self.val), 1e-6)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
