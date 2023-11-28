import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        diff = input - target
        squares = diff ** 2
        return squares.sum() / (input.shape[0] * input.shape[1])

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        squares_grad = np.ones_like(input) / (input.shape[0] * input.shape[1])
        diff_grad = squares_grad * (2 * (input - target))
        return diff_grad


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.log_probability = None
        self.mask = None

    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        self.log_probability = self.log_softmax.compute_output(input)
        self.mask = np.zeros_like(input)
        np.put_along_axis(self.mask, target[np.newaxis].T, 1, axis=1)
        mul = self.mask * self.log_probability
        res = -1 * mul.sum() / target.shape[0]
        return res

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        mul_grad = -1 * np.ones_like(input) / target.shape[0]
        log_probability_grad = mul_grad * self.mask
        return self.log_softmax.compute_grad_input(input, log_probability_grad)
