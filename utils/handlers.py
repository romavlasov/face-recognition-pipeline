class AverageMeter(object):
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.moment = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.moment = self.moment * self.alpha + (1 - self.alpha) * val
        self.count += n
        self.avg = self.sum / self.count


class MetaData(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = np.inf
        self.score = 0        

    def update(self, loss, score):
        self.loss = loss
        self.score = score