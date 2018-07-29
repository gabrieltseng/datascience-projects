class CyclicalLR(object):
    """
    Cyclical learning rate, as described here: https://arxiv.org/pdf/1506.01186.pdf

    Learning rate will linearly increase between max_lr and min_lr for the duration of
    `stepsize`.
    Generally, `stepsize` should be 2 to 10 times the number of iterations in an epoch.

    Note that unlike many other learning rate schedulers, this should be 'stepped' every batch,
    not every epoch.
    """

    def __init__(self, optimizer, stepsize, min_lr=1e-5, max_lr=10):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.stepsize = stepsize

        # now, lets calculate the difference increments
        self.inc = (max_lr - min_lr) / (stepsize - 1)

        # this will help us figuring out if we should be increasing or decreasing
        self.cycle_number = 1
        self.stepcount = 0

    def get_lr(self):
        if self.cycle_number % 2 == 0:
            current_inc = - self.inc
            starting_lr = self.max_lr
        else:
            current_inc = self.inc
            starting_lr = self.min_lr
        new_lr = starting_lr + (current_inc * self.stepcount)
        if (self.stepcount > 0) and ((new_lr <= self.min_lr) or (new_lr >= self.max_lr)):
            self.cycle_number += 1
            # will go to 0 when step() is run
            self.stepcount = -1

        return new_lr

    def step(self):
        new_lr = self.get_lr()
        self.stepcount += 1
        for param_group in  self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def test(self):
        """
        Test how the learning rate will change each time `step` is called
        """

        learning_rate = []
        for i in range(4 * self.stepsize):
            learning_rate.append(self.get_lr())
            self.stepcount += 1

        # reset everything
        self.cycle_number = 1
        self.stepcount = 0

        return learning_rate
