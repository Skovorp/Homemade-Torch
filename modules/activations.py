import numpy as np
from .base import Module


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(input, np.zeros_like(input))

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        res = np.zeros_like(input)
        res[input > 0] = 1
        return res * grad_output


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return 1 / (1 + np.exp(-input))

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        t = np.exp(-input)
        return grad_output * (t / (1 + t) ** 2)


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def __init__(self, ):
        super().__init__()
        self.exps = None
        self.sums = None
        self.inv_sum = None

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # print("in compute_output")
        self.exps = np.exp(input)
        self.sums = self.exps.sum(1, keepdims=True)
        self.inv_sum = 1 / self.sums
        return self.exps * self.inv_sum

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # print("in compute_grad_input")
        num_classes = input.shape[1]

        inv_sum_grad = (grad_output * self.exps).sum(1, keepdims=True)
        sum_grad = inv_sum_grad * -1 * (self.inv_sum ** 2)
        exps_grad = grad_output * self.inv_sum + sum_grad.repeat(num_classes, axis=1)
        input_grad = exps_grad * self.exps
        return input_grad


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def __init__(self, ):
        super().__init__()
        self.exps = None
        self.sums = None
        self.log_sum = None

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        # print("in compute_output")
        self.exps = np.exp(input)
        self.sums = self.exps.sum(1, keepdims=True)
        self.log_sum = np.log(self.sums)
        return input - self.log_sum

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # print("in compute_grad_input")
        # print(f"input shape {input.shape}")
        num_classes = input.shape[1]

        log_sum_grad = -1 * grad_output.sum(axis=1, keepdims=True)
        sums_grad = log_sum_grad / self.sums
        exps_grad = sums_grad.repeat(num_classes, axis=1)
        # print(exps_grad.shape)
        input_grad = grad_output + exps_grad * self.exps
        return input_grad
