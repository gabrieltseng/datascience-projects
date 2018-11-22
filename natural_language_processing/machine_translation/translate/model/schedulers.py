class OneCycle(object):
    """
    OneCycle, as described here: https://arxiv.org/pdf/1708.07120.pdf

    Learning rate will linearly increase between max_lr and min_lr for the duration of
    upswing * epochs.
    It will then decrease linearly for the duration of downswing * epochs

    Note that unlike many other learning rate schedulers, this should be 'stepped' every batch,
    not every epoch.
    """

    def __init__(self, optimizer, epoch_length, upswing, downswing, min_lr=1e-5, max_lr=10):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.upswing = upswing
        self.downswing = downswing
        # now, lets calculate the difference increments
        self.upsteps = epoch_length * upswing
        self.up_inc = (max_lr - min_lr) / self.upsteps

        self.downsteps = epoch_length * downswing
        self.down_inc = (min_lr - max_lr) / self.downsteps

        # this will help us figuring out if we should be increasing or decreasing
        self.stepcount = 0
        self.current_lr = min_lr

    def num_epochs(self):
        return self.upswing + self.downswing

    def get_lr(self):
        if self.stepcount > self.upsteps:
            current_inc = self.down_inc
        else:
            current_inc = self.up_inc
        self.current_lr += current_inc

    def step(self):
        self.get_lr()
        self.stepcount += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def test(self):
        """
        Test how the learning rate will change each time `step` is called
        """

        learning_rate = []
        total_steps = self.upsteps + self.downsteps
        for i in range(total_steps):
            self.get_lr()
            learning_rate.append(self.current_lr)
            self.stepcount += 1

        # reset everything
        self.stepcount = 0
        self.current_lr = self.min_lr

        return learning_rate


class TeacherForcing(object):
    """
    Teacher forcing
    """
    def __init__(self, num_epochs, epoch_length, model):
        # either linearly decrease over min(num_epochs, 11) - 1 epochs
        # (i.e. 10 epochs with forcing, at least one without)
        num_forcing_steps = epoch_length * (min(num_epochs, 11) - 1)
        self.current_forcing = 1
        self.forcing_stepsize = 1 / num_forcing_steps
        self.model = model

    def get_forcing(self):
        if self.current_forcing > 0:
            self.current_forcing = max(0, self.current_forcing - self.forcing_stepsize)

    def step(self):
        self.get_forcing()
        self.model.forcing_probability = self.current_forcing
