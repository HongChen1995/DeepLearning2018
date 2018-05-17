# -*- coding: utf-8 -*-

from torch import FloatTensor, LongTensor, manual_seed
import nn_group14 as nn14
import toy_model
import time

# %% Train and test model

def compute_nb_errors(model, data_input, data_target, one_hot=False, batch_size=100):
    """Compute number of classification errors of a given model.

    Args:
        model (Module): Module trained for classification
        data_input (FloatTensor): must have 2D size Nb x Nin, where Nb
            is the batch size and Nin the number of input units of model.
        data_target (FloatTensor): must have 2D size Nb x Nout, where Nout
            is the number of output units of the model, which should match the
            number of classes. One-hot encoding must be used, i.e.
            data_target[i,j]=1 if data sample i belongs to class j, and
            data_target[i,j]=-1 otherwise.
        one_hot (bool): specify if one-hot encoding was used for the target.
            Default: False.
        batch_size (int): batch size which should be used for an efficient
            forward pass. Does not necessarily need to be a divider of the
            number of data samples, althgough this is often desirable for
            statistical reasons. Note that this parameter does not influence
            model training at all. Default: 100.
    """
    Ndata = data_input.size(0)
    if one_hot:
        data_label = data_target.max(dim=1)[1]
    nb_errors = 0
    for b_start in range(0, data_input.size(0), batch_size):
        # account for boundary effects:
        bsize_eff = batch_size - max(0, b_start + batch_size - Ndata)
        # batch output has size Nbatch x 2 if one_hot=True, Nbatch otherwise:
        batch_output = model.forward(data_input.narrow(0, b_start, bsize_eff))
        if one_hot:
            pred_label = batch_output.max(dim=1)[1]  # has size Nbatch
            nb_err_batch = 0
            for k in range(bsize_eff):
                if data_label[b_start+k] != pred_label[k]:
                    nb_err_batch = nb_err_batch + 1
        else:
            nb_err_batch = (batch_output.max(1)[1] !=
                            data_target.narrow(0, b_start, bsize_eff)).long().sum()
        # conversion to Long solves serious overflow problem; otherwise the above results are treated as 1-byte short ints
        nb_errors += nb_err_batch
    return nb_errors

def train_model(model, train_input, train_target, criterion, optimizer, n_epochs=50, batch_size=100, log_loss=False):
    """Train model.

    Args:
        model (Module)
        train_input (FloatTensor)
        train_target (Tensor): converted to float if needed
        criterion (Loss): loss function
        optimizer (Optimizer): optimizer
        n_epochs (int)
        batch_size (int)
        log_loss (bool): set to True to print training progress a few times
    """
    Ntrain = train_input.size(0)
    for i_ep in range(Nepochs):
        ep_loss = 0.0
        for b_start in range(0, Ntrain, batch_size):
            model.zero_grad()

            # account for boundary effects
            bsize_eff = batch_size - max(0, b_start + batch_size - Ntrain)

            # forward pass
            output = model.forward(train_input.narrow(0, b_start, bsize_eff))
            batch_loss = criterion.loss(output, train_target.narrow(0, b_start, bsize_eff))
            ep_loss += batch_loss

            # backward pass
            dl_dout = criterion.backward(output, train_target.narrow(0, b_start, bsize_eff))
            dl_dx = model.backward(dl_dout)

            # parameter update
            optimizer.step()

        # print progress
        ep_err = compute_nb_errors(model, train_input, train_target, one_hot)
        if log_loss and i_ep % round(Nepochs/5) == 0:
            print("epoch {:d}/{:d}: training loss {:4.3g} (error rate {:3.2g}%)"
                  "".format(i_ep+1,Nepochs, batch_loss, 100*ep_err/Ntrain))

# %% RUN

# Make runs reproducible
myseed = 141414
manual_seed(myseed)

# Select one_hot for MSE optimization for instance
one_hot = True

# Generate train and test standardized data
Ntrain = 1000
Ntest = Ntrain
train_input, train_target, test_input, test_target = toy_model.generate_data_standardized(Ntrain, one_hot)

# Time model creation, training and testing
Nreps = 15
batch_sizes = [1, 5, 15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 750, 1000]
time_elapsed = FloatTensor(len(batch_sizes), Nreps).zero_()
train_err = FloatTensor(len(batch_sizes), Nreps).zero_()
test_err = FloatTensor(len(batch_sizes), Nreps).zero_()

# Meta parameters
Nepochs = 30
eta = 1e-2
momentum = 0
log_loss = False

for i_b in range(len(batch_sizes)):
    b_size = batch_sizes[i_b]
    for irep in range(Nreps):
        t_start = time.time()

        # Create model
        non_lin_activ = nn14.ReLU

        if non_lin_activ is nn14.ReLU:
            nonlin = 'relu'
        elif non_lin_activ is nn14.Tanh:
            nonlin = 'tanh'
        elif non_lin_activ is nn14.Sigmoid:
            nonlin = 'sigmoid'
        else:
            raise ValueError("Please use ReLU, Tanh or Sigmoid for non-linear activation layers")

        fc1 = nn14.Linear(2, 25, nonlinearity=nonlin)
        fc2 = nn14.Linear(25, 25, nonlinearity=nonlin)
        fc3 = nn14.Linear(25, 25, nonlinearity=nonlin)
        fc_out = nn14.Linear(25, 2)
        model = nn14.Sequential([fc1, non_lin_activ(),
                                 fc2, non_lin_activ(),
                                 fc3, non_lin_activ(),
                                 fc_out])

        # Loss function
        lossfunction = nn14.MSELoss()  # MSELoss(), NLLLoss(), CrossEntropyLoss()
        if isinstance(lossfunction, nn14.MSELoss):
            assert one_hot is True, "Use one_hot targets with MSELoss"

        # Optimizer
        optimizer = nn14.SGD(model.param(), lr=eta, momentum=momentum)

        # Train model
        train_model(model, train_input, train_target, lossfunction, optimizer, Nepochs, b_size, log_loss=log_loss)

        # Evaluate and store results
        nb_train_err = compute_nb_errors(model, train_input, train_target, one_hot)
        nb_test_err = compute_nb_errors(model, test_input, test_target, one_hot)
        print("NN Gr14 (rep {:d}/{:d}) Train error rate {}%, test error rate {}%\n".format(irep+1, Nreps, 100*nb_train_err/Ntrain, 100*nb_test_err/Ntest))

        time_elapsed[i_b, irep] = time.time() - t_start
        train_err[i_b, irep] = 100*nb_train_err/Ntrain
        test_err[i_b, irep] = 100*nb_test_err/Ntest

