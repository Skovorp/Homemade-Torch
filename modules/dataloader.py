import numpy as np

class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__
        self.index_seq = None

    def __len__(self) -> int:
        """
        :return: number of batches per epoch
        """
        return int(np.ceil(self.num_samples() / self.batch_size))

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        # print("Starting iter........")
        self.batch_id = 0
        self.index_seq = np.arange(self.num_samples())
        if self.shuffle:
            np.random.shuffle(self.index_seq)
        return self

    def __next__(self):
        """
        Form and return next data batch
        :return: (x_batch, y_batch)
        """
        needed_inds = self.index_seq[self.batch_id * self.batch_size: (self.batch_id + 1) * self.batch_size]
        if len(needed_inds) > 0:
            res = self.X[needed_inds], self.y[needed_inds]
            # print(f"id {self.batch_id}, len(needed_inds): {len(needed_inds)} x: {res[0]}")
            self.batch_id += 1
            return res
        else:
            raise StopIteration
