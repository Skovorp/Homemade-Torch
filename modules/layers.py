import numpy as np
from typing import List
from .base import Module


class Linear(Module):
    """
    Applies linear (affine) transformation of data: y = x W^T + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        :param in_features: input vector features
        :param out_features: output vector features
        :param bias: whether to use additive bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.random.uniform(-1, 1, (out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.random.uniform(-1, 1, out_features) / np.sqrt(in_features) if bias else None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias) if bias else None

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, in_features)
        :return: array of shape (batch_size, out_features)
        """
        if self.bias is not None:
            return input @ self.weight.T + self.bias
        else:
            return input @ self.weight.T

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        :return: array of shape (batch_size, in_features)
        """
        # print("in compute_grad_input")
        return grad_output @ self.weight

    def update_grad_parameters(self, input: np.array, grad_output: np.array):
        """
        :param input: array of shape (batch_size, in_features)
        :param grad_output: array of shape (batch_size, out_features)
        """
        # print("in update_grad_parameters")
        self.grad_weight += grad_output.T @ input
        if self.bias is not None:
            self.grad_bias += grad_output.sum(0)

    def zero_grad(self):
        self.grad_weight.fill(0)
        if self.bias is not None:
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.array]:
        if self.bias is not None:
            return [self.weight, self.bias]

        return [self.weight]

    def parameters_grad(self) -> List[np.array]:
        if self.bias is not None:
            return [self.grad_weight, self.grad_bias]

        return [self.grad_weight]

    def __repr__(self) -> str:
        out_features, in_features = self.weight.shape
        return f'Linear(in_features={in_features}, out_features={out_features}, ' \
               f'bias={not self.bias is None})'


class BatchNormalization(Module):
    """
    Applies batch normalization transformation
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        :param num_features:
        :param eps:
        :param momentum:
        :param affine: whether to use trainable affine parameters
        """
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.weight = np.ones(num_features) if affine else None
        self.bias = np.zeros(num_features) if affine else None

        self.grad_weight = np.zeros_like(self.weight) if affine else None
        self.grad_bias = np.zeros_like(self.bias) if affine else None

        # store this values during forward path and re-use during backward pass
        self.mean = None
        self.input_mean = None  # input - mean
        self.var = None
        self.sqrt_var = None
        self.inv_sqrt_var = None
        self.norm_input = None

        self.is_train_mode = True

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        # print("in compute_output")
        if self.is_train_mode:
            # print("Train")
            batch_size = input.shape[0]
            self.mean = input.sum(0) / batch_size
            self.input_mean = input - self.mean

            self.var = (self.input_mean ** 2).sum(0) / batch_size
            self.sqrt_var = (self.var + self.eps) ** 0.5
            self.inv_sqrt_var = 1 / self.sqrt_var
            self.norm_input = self.input_mean * self.inv_sqrt_var

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.var * batch_size / (batch_size - 1)

            # print(f"average mean: {self.norm_input.mean(0).mean()} average std: {self.norm_input.std(0).mean()}")
        else:
            # print("Eval")
            self.input_mean = input - self.running_mean
            self.sqrt_var = (self.running_var + self.eps) ** 0.5
            self.inv_sqrt_var = 1 / self.sqrt_var
            self.norm_input = self.input_mean * self.inv_sqrt_var

            # print(f"average mean: {self.norm_input.mean(0).mean()} average std: {self.norm_input.std(0).mean()}")

        if self.affine:
            return self.norm_input * self.weight + self.bias
        else:
            return self.norm_input

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        :return: array of shape (batch_size, num_features)
        """
        # print('in compute_grad_input')
        batch_size, num_features = input.shape[0], input.shape[1]

        if self.affine:
            grad_output_normal = grad_output * self.weight
        else:
            grad_output_normal = grad_output

        if self.is_train_mode:
            inv_sqrt_var_grad = (grad_output_normal * self.input_mean).sum(0)
            sqrt_var_grad = inv_sqrt_var_grad * (-1 * self.inv_sqrt_var ** 2)
            var_grad = sqrt_var_grad * (0.5 * self.inv_sqrt_var)
            norm_squares_grad = (1 / batch_size * var_grad).reshape(1, num_features).repeat(batch_size, axis=0)
            input_mean_grad = grad_output_normal * self.inv_sqrt_var.reshape(1, num_features).repeat(batch_size, axis=0) + \
                              norm_squares_grad * 2 * self.input_mean

            mean_grad = (-1 * input_mean_grad).sum(0)
            input_grad = input_mean_grad + (mean_grad / batch_size).reshape(1, num_features).repeat(batch_size, axis=0)

            return input_grad
        else:
            return grad_output_normal * self.inv_sqrt_var


    def update_grad_parameters(self, input: np.array, grad_output: np.array):
        """
        :param input: array of shape (batch_size, num_features)
        :param grad_output: array of shape (batch_size, num_features)
        """
        # print('in update_grad_parameters')
        if self.affine:
            self.grad_bias += grad_output.sum(0)
            self.grad_weight += (grad_output * self.norm_input).sum(0)

    def eval(self):
        self.is_train_mode = False

    def train(self):
        self.is_train_mode = True

    def zero_grad(self):
        if self.affine:
            self.grad_weight.fill(0)
            self.grad_bias.fill(0)

    def parameters(self) -> List[np.array]:
        return [self.weight, self.bias] if self.affine else []

    def parameters_grad(self) -> List[np.array]:
        return [self.grad_weight, self.grad_bias] if self.affine else []

    def __repr__(self) -> str:
        return f'BatchNormalization(num_features={len(self.running_mean)}, ' \
               f'eps={self.eps}, momentum={self.momentum}, affine={self.affine})'


class Dropout(Module):
    """
    Applies dropout transformation
    """
    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None
        self.is_train_mode = True

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        # print("in compute_output")
        # print(f'p is {self.p}')
        # print(input.mean(0))
        if self.is_train_mode:
            self.mask = np.random.binomial(1, 1 - self.p, size=input.shape)  # 1 - p is probability to get 1
            # print(self.mask)
            # print(input * self.mask / (1 - self.p))

            return input * self.mask / (1 - self.p)
        else:
            return input

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        # print("in compute_grad_input")
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        if self.is_train_mode:
            return grad_output * self.mask / (1 - self.p)
        else:
            return grad_output

    def eval(self):
        self.is_train_mode = False

    def train(self):
        self.is_train_mode = True

    def __repr__(self) -> str:
        return f'Dropout(p={self.p})'


class Sequential(Module):
    """
    Container for consecutive application of modules
    """
    def __init__(self, *args):
        super().__init__()
        self.modules = list(args)
        self.inputs = []

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size matching the input size of the first layer
        :return: array of size matching the output size of the last layer
        """
        # print('in compute_output')
        arg = input
        self.inputs = []
        for module in self.modules:
            self.inputs.append(arg)
            arg = module.compute_output(arg)

        # for el in self.inputs:
        #     print(el.mean(axis=0), '\n')
        return arg

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size matching the input size of the first layer
        :param grad_output: array of size matching the output size of the last layer
        :return: array of size matching the input size of the first layer
        """
        # print('in compute_grad_input')
        grad = grad_output
        for i in range(len(self.modules) - 1, -1, -1):
            # print(i, "mean", grad.mean(axis=0), "input", self.inputs[i].mean(axis=0))
            self.modules[i].update_grad_parameters(self.inputs[i], grad)
            grad = self.modules[i].compute_grad_input(self.inputs[i], grad)
        return grad

    def __getitem__(self, item):
        return self.modules[item]

    def train(self):
        for module in self.modules:
            module.train()

    def eval(self):
        for module in self.modules:
            module.eval()

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def parameters(self) -> List[np.array]:
        return [parameter for module in self.modules for parameter in module.parameters()]

    def parameters_grad(self) -> List[np.array]:
        return [grad for module in self.modules for grad in module.parameters_grad()]

    def __repr__(self) -> str:
        repr_str = 'Sequential(\n'
        for module in self.modules:
            repr_str += ' ' * 4 + repr(module) + '\n'
        repr_str += ')'
        return repr_str
