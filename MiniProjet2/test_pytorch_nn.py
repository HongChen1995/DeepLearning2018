# -*- coding: utf-8 -*-

import torch
from torch import Tensor, manual_seed
from torch import nn  # not allowed in project 2; just for comparison
from torch.autograd import Variable
from math import pi
import toy_model
import time

def train_model(model, train_input, train_target, criterion, optimizer, n_epochs=50, batch_size=100, log_loss=False):
    Ndata = train_input.size(0)
    for e in range(0, n_epochs):
        epoch_loss = 0
        for b_start in range(0, train_input.size(0), batch_size):
            model.zero_grad()  # seems to work well outside of batch loop, sort of inertia?

            bsize_eff = batch_size - max(0, b_start + batch_size - Ndata)   # accounts for boundary effects
            batch_output = model(train_input.narrow(0, b_start, bsize_eff))
            batch_loss = criterion(batch_output, train_target.narrow(0, b_start, bsize_eff))  # instance of Variable
            epoch_loss = epoch_loss + batch_loss.data[0]
            batch_loss.backward()
            optimizer.step()
        if log_loss and e % round(n_epochs/5) == 0:  # prints 5 times
            print("Epoch {:d}/{:d}: epoch loss {:5.4g}".format(e+1, n_epochs, epoch_loss))


def compute_nb_errors(model, data_input, data_target, one_hot=False, batch_size=100):
    nb_errors = 0
    Ndata = data_input.size(0)
    for b_start in range(0, data_input.size(0), batch_size):
        bsize_eff = batch_size - max(0, b_start + batch_size - Ndata)   # accounts for boundary effects
        batch_output = model(data_input.narrow(0, b_start, bsize_eff))  # Nbatch x 2 if one_hot=True, Nbatch otherwise
        if one_hot:
            pred_label = batch_output.max(dim=1)[1]  # size Nbatch
            data_label = data_target.narrow(0, b_start, bsize_eff).max(dim=1)[1]  # could be done outside the batch loop; size is Nbatch
            nb_err_batch = 0
            for k in range(bsize_eff): # not very efficient but safest bet
                if data_label.data[k] != pred_label.data[k]: # data extracts torch.Tensor out of Variable
                    nb_err_batch = nb_err_batch + 1
        else:
            nb_err_batch = (batch_output.max(1)[1] != data_target.narrow(0, b_start, bsize_eff)).long().sum()
        # treated as short 1-byte int otherwise!!
        nb_errors += nb_err_batch
    if isinstance(nb_errors, torch.autograd.Variable):
        nb_errors = nb_errors.data[0]
    return nb_errors


def create_miniproject2_model(nonlin_activ=nn.ReLU()):
    return nn.Sequential(nn.Linear(2, 25), nonlin_activ,
                         nn.Linear(25, 25), nonlin_activ,
                         nn.Linear(25, 25), nonlin_activ,
                         nn.Linear(25, 2))


# %% Run

# Make runs reproducible
myseed = 141414
manual_seed(myseed)

# Select one_hot for MSE optimization for instance
one_hot = True

# Generate train and test standardized data
Ntrain = 1000
Ntest = Ntrain
train_input, train_target, test_input, test_target = toy_model.generate_data_standardized(Ntrain, one_hot)
train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

if one_hot:
    train_target = train_target.float()
    test_target = test_target.float()

# Time model creation, training and testing
Nreps = 15
batch_sizes = [1, 5, 15, 25, 50, 100, 150, 200, 250, 300, 350, 400, 500, 750, 1000]
time_elapsed = torch.Tensor(len(batch_sizes), Nreps).zero_()
train_err = torch.Tensor(len(batch_sizes), Nreps).zero_()
test_err = torch.Tensor(len(batch_sizes), Nreps).zero_()

# Meta parameters
n_epochs = 30  # number of times the training samples are visited
eta = 1e-2  # learning rate
momentum = 0  # "inertia" (see handout 5, slide 22/83)
log_loss = False  # True for printing the loss during the training; False for no verbose at all

for i_b in range(len(batch_sizes)):
    batch_size = batch_sizes[i_b]
    for irep in range(Nreps):
        t_start = time.time()

        # Create model
        non_lin_activation = nn.ReLU()  # nn.Tanh(), nn.ReLU(), nn.Sigmoid(),...
        model = create_miniproject2_model(non_lin_activation)

        # Loss function
        loss_function = torch.nn.MSELoss()  #MSELoss, NLLLoss, CrossEntropyLoss

        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=momentum)

        # Train model
        train_model(model, train_input, train_target, loss_function, optimizer, n_epochs=n_epochs, batch_size=batch_size, log_loss=log_loss)

        # Evaluate and store results
        n_err_train = compute_nb_errors(model, train_input, train_target, one_hot, batch_size)
        n_err_test = compute_nb_errors(model, test_input, test_target, one_hot, batch_size)
        print("PyTorch NN (rep {:d}/{:d}): train error {:g}%, test error {:g}%\n".format(irep+1, Nreps, n_err_train*100/Ntrain, n_err_test*100/Ntest))

        time_elapsed[i_b, irep] = time.time() - t_start
        train_err[i_b, irep] = n_err_train*100/Ntrain
        test_err[i_b, irep] = n_err_test*100/Ntest