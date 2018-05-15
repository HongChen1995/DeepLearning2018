# -*- coding: utf-8 -*-

import math
from torch import FloatTensor, LongTensor, manual_seed
from numpy import arange

# %% Toy example - data generation
def generate_disc_set(nb, one_hot=False):
    data = FloatTensor(nb, 2).uniform_(0, 1)# uniform distribution in [0,1] x [0,1]
    target = data.pow(2).sum(1).sub(1 / (2*math.pi)).sign().sub(1).div(-2).long() # 1 if inside disk of radius sqrt(2/pi), 0 if outside
    if one_hot:
        # Useful for MSE loss: convert from scalar labels to vector labels
        target_one_hot = LongTensor(nb, 2).fill_(-1)
        # Stupid Tensor does not seem to have an efficient way to do this; use ugly loop
        for k in range(target.size(0)):
            target_one_hot[k, target[k]] = 1
        target = target_one_hot
    return data, target


def generate_data_standardized(nb, one_hot=False):
    ########## Generation of the data ##########
    train_input, train_target = generate_disc_set(nb, one_hot)
    test_input, test_target = generate_disc_set(nb, one_hot)

    ########## Calculation of mean and std of training set ##########
    mean, std = train_input.mean(), train_input.std()

    ########## Standardization ##########
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    return train_input, train_target, test_input, test_target


# %% Module class and child classes


class Module (object):

    def forward(self, * input):
        raise NotImplementedError

    def backward(self, * gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        pass


class Parameter(object):

    def __init__(self, pytorch_tensor):
        self.data = pytorch_tensor.clone()  # tensors passed by reference
        self.grad = pytorch_tensor.clone().fill_(0)  # maintains Float or Long

    def zero_grad(self):
        self.grad.fill_(0)


class Linear(Module):

    def __init__(self, input_units, output_units, bias=True, nonlinearity=None):
        super(Linear, self).__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.weights = Parameter(FloatTensor(output_units, input_units))
        self.input = None  # last input used for forward pass

        if bias:
            self.bias = Parameter(FloatTensor(output_units))
        else:
            self.bias = None

        if nonlinearity is None:
            nonlinearity = 'sigmoid'

        self.initialize_parameters(nonlinearity)

    def initialize_parameters(self, nonlinearity):
        # Variance correction associated with nonlinear activation of network
        nonlinearity = nonlinearity.lower()
        if nonlinearity == 'sigmoid':
            self.init_gain = 1.0
        elif nonlinearity == 'tanh':
            self.init_gain = 5.0 / 3.0
        elif nonlinearity == 'relu':
            self.init_gain = math.sqrt(2.0)
        else:
            raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

        # Control variance of activations (not of gradients)
        uniform_corr = math.sqrt(3)
        stdv = uniform_corr * self.init_gain / math.sqrt(self.input_units)
        self.weights.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def param(self):
        if self.bias is not None:
            return [self.weights, self.bias]
        else:
            return [self.weights]

    def zero_grad(self):
        for p in self.param():
            p.zero_grad()

    def forward(self, x):
        """
        Applies forward pass of fully connected layer to input x

        Args:
            x (FloatTensor): if 2D must have size Nb x input_units, where Nb
                is the batch size. If x is 1D, it is assumed that Nb=1.

        Returns:
            FloatTensor: Nb x output_units, always 2D.

        """
        # store current input for backward pass
        self.input = x.clone().view(-1, self.input_units)

        if self.bias is not None:
            # automatic broadcasting for bias:
            return self.input.mm(self.weights.data.t()) + self.bias.data.view(-1, self.output_units)
        else:
            return self.input.mm(self.weights.data.t())

    def backward(self, dl_dout):
        """
        Applies backward pass of fully connected layer starting from gradient
        with respect to current output.

        Args:
            dl_dout (FloatTensor): if 2D, must have size Nb x output_units,
                where Nb is the batch size. If 1D, it is assumed that Nb=1.
                Contains the derivative of the batch loss with respect to
                each output unit, for each batch sample, of the current
                backward pass.

        Returns:
            FloatTensor: Nb x input_units tensor containing the derivative of
                the batch loss with respect to each input unit, for each batch
                sample, of the current backward pass.
        """
        ndim = len(list(dl_dout.size()))
        assert ndim > 0, "dl_dout argument cannot be empty"
        Nb = 1  # case where dl_dout is 1D, only one sample
        if ndim > 1:
            Nb = dl_dout.size(0)

        # Gradient increment for weights (broadcasting for batch-processing)
        # (sum contributions of all samples in the batch)
        grad_inc = (dl_dout.view(Nb, self.output_units, 1) *
                    self.input.view(Nb, 1, self.input_units)).sum(0)
        self.weights.grad.add_(grad_inc)

        # Gradient increment for bias
        # (sum of contributions of all samples in the batch)
        if self.bias is not None:
            self.bias.grad.add_(dl_dout.view(Nb, self.output_units).sum(0))

        # Return dl_din
        return dl_dout.view(Nb, self.output_units).mm(self.weights.data)


class ReLU(Module):

    def __init__(self):
        super(ReLU, self).__init__()
        self.input = None  # last input used for forward pass

    def forward(self, x):
        # store current input for backward pass
        self.input = x.clone()

        out = x.clone()
        out[x < 0] = 0
        return out

    def backward(self, dl_dout):
        return dl_dout * self._grad(self.input)

    def _grad(self, x):
        dout_dx = x.clone().fill_(1)
        dout_dx[x < 0] = 0
        return dout_dx


class Tanh(Module):

    def __init__(self):
        super(Tanh, self).__init__()
        self.input = None  # last input used for forward pass

    def forward(self, x):
        # store current input for backward pass
        self.input = x.clone()
        return x.tanh()

    def backward(self, dl_dout):
        return dl_dout * self._grad(self.input)

    def _grad(self, x):
        return 1-x.tanh().pow(2)


class Sigmoid(Module):

    def __init__(self):
        super(Sigmoid, self).__init__()
        self.input = None

    def forward(self, x):
        # store current input for backward pass
        self.input = x.clone()
        return x.sigmoid()

    def backward(self, dl_dout):
        return dl_dout * self._grad(self.input)

    def _grad(self, x):
        return (x.exp() + x.mul(-1).exp() + 2).pow(-1)


class Sequential(Module):

    def __init__(self, moduleList):
        super(Sequential, self).__init__()
        self.moduleList = moduleList
        self.input = None

    def forward(self, x):
        self.input = x.clone()
        output = x.clone()
        for m in self.moduleList:
            output = m.forward(output)
        return output

    def backward(self, dl_dout):
        dl_din = dl_dout.clone()
        for m in self.moduleList[::-1]:
            dl_din = m.backward(dl_din)
        return dl_din

    def param(self):
        par_list = []
        for m in self.moduleList:
            par_list.extend(m.param())
        return par_list

    def zero_grad(self):
        for m in self.moduleList:
            m.zero_grad()


# %% Non-linear layers
class LogSoftMax(Module):
    """ Applies logarithm and softmax component-wise.
    """
    def __init__(self):
        super(LogSoftMax, self).__init__()
        self.input = None

    def forward(self, x):
        self.input = x.clone()
        # shift by max for numerical stability
        x_norm = x - x.max(dim=1, keepdim=True)[0]
        e_x = x_norm.exp()
        return x_norm - e_x.sum(dim=1, keepdim=True).log()

    def backward(self, dl_dout):
        # shift by max for numerical stability
        x_norm = self.input - self.input.max(1, keepdim=True)[0]
        e_x = x_norm.exp()
        softmax_x = e_x / e_x.sum(dim=1, keepdim=True)
        return (-softmax_x * dl_dout).sum(dim=1, keepdim=True) + dl_dout


# %% Loss functions
class Loss(object):
    """ Base class for Loss functions.
    """
    def loss(self, output, target):
        raise NotImplementedError

    def backward(self, output, target):
        """
        Returns derivative of loss(output, target) with respect to output.
        """
        raise NotImplementedError


class MSELoss(Loss):
    """ Mean squared error loss
    """
    def loss(self, output, target):
        """Computes the sum of the squared differences between each sample of
        the batch and sums over all batch samples.

        Args:
            output (FloatTensor): model output, of size Nb x d1 x d2 x ...
                where Nb is the batch size.
            target (FloatTensor): must have the same size as output.

        Returns:
            float: a scalar floating-point value.
        """
        return (output-target).pow(2).sum()/output.size(0)  # sum over all samples and entries

    def backward(self, output, target):
        return 2 * (output-target)/ output.size(0)


class NLLLoss(Loss):
    """ Negative log likelihood loss.
    """

    def loss(self, output, target):
        """
        """
        # Get dimension
        ndim = len(list(output.size()))
        assert ndim == 2, "output argument must have size Nb x d"
        Nb = output.size(0)
        out_dim = output.size(1)
        # sum the "on-target" activations across the batch:
        return - output.view(-1)[arange(0, Nb)*out_dim + target.long()].sum()/Nb

    def backward(self, output, target):
        dl_din = FloatTensor(output.size()).fill_(0)
        # Get dimension
        ndim = len(list(output.size()))
        assert ndim == 2, "output argument must have size Nb x d"
        Nb = output.size(0)
        for i in range(Nb):
            dl_din[i, target[i]] = -1/Nb
        return dl_din

# %% Optimizer class
class Optimizer(object):
    """ Base class for optimizers.

    Args:
        params (iterable of type Parameter): iterable yielding the parameters
            of the model to optimize.
    """
    def __init__(self, params):
        self.params = params

    def step(self, * input):
        raise NotImplementedError


class SGD(Optimizer):
    """ Stochastic gradient descend with fixed learning rate and momentum.

    Args:
        params (iterable of type Parameter): iterable yielding the parameters
            of the model to optimize.
        lr (float): strictly positive learning rate or gradient step length
        momentum (float): non-negative weight of the inertia term


    """

    def __init__(self, params, lr, momentum=0):
        super(SGD, self).__init__(params)
        assert lr > 0, "learning rate should be strictly positive"
        self.lr = lr
        assert momentum >= 0, "momentum term should be non-negative"
        self.momentum = momentum
        if self.momentum > 0:
            self.step_buf = {}
            for p in self.params:
                #self.step_buf.append(FloatTensor(p.grad.size()).zero_())
                self.step_buf[p] = FloatTensor(p.grad.size()).zero_()

    def step(self):
        for p in self.params:
            grad_step = self.lr * p.grad
            if self.momentum > 0:
                grad_step.add_(self.momentum * self.step_buf[p])
                self.step_buf[p] = grad_step.clone()
            p.data.add_(-grad_step)

# %% Train and test modules

def compute_nb_errors(model, data_input, data_target, one_hot=False, batch_size=100):
    Ndata = data_input.size(0)
    if one_hot:
        data_label = data_target.max(dim=1)[1]
    nb_errors = 0
    for b_start in range(0, data_input.size(0), batch_size):
        # account for boundary effects
        bsize_eff = batch_size - max(0, b_start + batch_size - Ndata)
        # batch output has size Nbatch x 2 if one_hot=True, Nbatch otherwise
        batch_output = model.forward(data_input.narrow(0, b_start, bsize_eff))
        if one_hot:
            pred_label = batch_output.max(dim=1)[1]  # size: Nbatch
            nb_err_batch = 0
            for k in range(bsize_eff):
                if data_label[b_start+k] != pred_label[k]:
                    nb_err_batch = nb_err_batch + 1
        else:
            nb_err_batch = (batch_output.max(1)[1] != data_target.narrow(0, b_start, bsize_eff)).long().sum()
        # SERIOUS overflow problem if conversion to Long Int not performed because treated as 1-byte short int otherwise!!
        nb_errors += nb_err_batch
    return nb_errors

# %% RUN

# Make runs reproducible
myseed = 141414
manual_seed(myseed)

Ndata = 1000
Ntest = Ndata

# Select one_hot for MSE optimization for instance
one_hot = True

train_input, train_target, test_input, test_target = generate_data_standardized(Ndata, one_hot)

# Avoid saturation
if one_hot:
    train_target = 0.9 * train_target.float()
    test_target = 0.9 * test_target.float()

# Create model
fc1 = Linear(2, 25)  #nonlinearity='relu'
fc2 = Linear(25, 25)
fc3 = Linear(25, 25)
fc_out = Linear(25, 2)


model = Sequential([fc1, ReLU(),
                    fc2, ReLU(),
                    fc3, ReLU(),
                    fc_out])  # , LogSoftMax()

# Train model
criterion = MSELoss()
if isinstance(criterion, MSELoss):
    assert one_hot is True, "Use one_hot targets with MSELoss"

Nepochs = 50
b_size = 100
lr = 0.01

optimizer = SGD(model.param(), lr, momentum=0)
for i_ep in range(Nepochs):
    model.zero_grad()
    ep_loss = 0.0
    for b_start in range(0, Ndata, b_size):
        # account for boundary effects
        bsize_eff = b_size - max(0, b_start + b_size - Ndata)

        output = model.forward(train_input.narrow(0, b_start, bsize_eff))
        batch_loss = criterion.loss(output, train_target.narrow(0, b_start, bsize_eff))
        ep_loss += batch_loss
        # derivative of loss with respect to final output
        dl_dout = criterion.backward(output, train_target.narrow(0, b_start, bsize_eff))
        # backward pass
        dl_dx = model.backward(dl_dout)
        optimizer.step()
    print("epoch {}: loss {}".format(i_ep+1, batch_loss))

nb_train_err = compute_nb_errors(model, train_input, train_target, one_hot)
nb_test_err = compute_nb_errors(model, test_input, test_target, one_hot)
print("Train error rate {}%, test error rate {}%".format(100*nb_train_err/Ndata, 100*nb_test_err/Ntest))
