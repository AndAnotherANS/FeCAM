import copy

import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset


def mahal(samples, mean, cov):
    """
    compute square mahalanobis distance using full covariance matrix
    calculates (x - mu) @ sigma @ (x - mu).T
    function copied from base.py
    """
    x_minus_mu = F.normalize(samples, p=2, dim=-1) - F.normalize(mean, p=2, dim=-1)
    inv_covmat = torch.linalg.pinv(cov).float().to(x_minus_mu)
    left_term = torch.matmul(x_minus_mu, inv_covmat)
    mahal = torch.matmul(left_term, x_minus_mu.T)
    return torch.diag(mahal)


def mahal_chol(samples, mean, cov_chol):
    """
    compute mahalanobis distance using cholesky decomposition of covariance matrix
    calculates squared euclidean norm of L @ (x - mu) where L @ L.T is the covariance matrix
    """
    x_minus_mu = F.normalize(samples, p=2, dim=-1) - F.normalize(mean, p=2, dim=-1)
    inv_cov = torch.linalg.pinv(cov_chol).float().to(x_minus_mu)
    left_term = torch.matmul(inv_cov, x_minus_mu.T).T
    return torch.norm(left_term, 2., -1) ** 2


def normalize_chol(chol):
    """
    Normalize cholesky decomposition of covariance matrix
    outputs matrix L so that L @ L.T is normalized covariance matrix
    """
    diagonal = (chol ** 2) @ torch.ones(chol.shape[0]).to(chol)
    normalized = torch.diagflat((diagonal) ** -(0.5)) @ chol
    return normalized


def normalize_full(cov):
    """
    Normalize full covariance matrix
    function copied from base.py
    """
    sd = torch.sqrt(torch.diagonal(cov))  # standard deviations of the variables
    cov = cov / (torch.matmul(sd.unsqueeze(1), sd.unsqueeze(0)))
    return cov


class MahaModule(nn.Module):
    """
    Module for keeping parameters representing covariance and center of the multivariate gaussian
    covariance cholesky decomposition is represented as square matrix, of which lower triangular part L is only used
    """

    def __init__(self, cov, mean):
        super().__init__()
        W = torch.linalg.cholesky(cov)
        self.register_parameter("cov", nn.Parameter(W))
        self.register_parameter("mean", nn.Parameter(mean))

    def forward(self, x):
        dist = torch.distributions.MultivariateNormal(self.mean, scale_tril=normalize_chol(torch.tril(self.cov)))
        return dist.log_prob(x)

    def get_cov(self):
        return torch.tril(self.cov) @ torch.tril(self.cov).T


def get_loss(cov_module: MahaModule, x, other):
    """
    :parameter x: data from the class for which the distribution is being optimized
    :parameter other: array of negative mahalanobis distances of data points from x to all other distributions
    """
    log_prob = cov_module(x)
    maha_x = -mahal_chol(x, cov_module.mean, normalize_chol(torch.tril(cov_module.cov)))

    # temperature
    maha_x_temp = maha_x
    other_temp = other

    # compute softmax with shift by max value
    max_other = torch.max(torch.cat([other_temp, maha_x_temp.unsqueeze(-1)], -1), -1, keepdim=True).values
    softmax_nom = torch.exp(maha_x_temp - max_other.squeeze())
    softmax_denom = torch.exp(maha_x_temp - max_other.squeeze()) + torch.exp(other_temp - max_other).sum(-1)

    return log_prob, softmax_nom / softmax_denom


def optimize_covariance(data, max_cls, cls, model):
    """
    Initialize covariance to one used in FeCAM and optimize it according to the get_loss function
    """
    # get cov and mean for current class
    cov = model._cov_mat_shrink[cls].cuda().float()
    mean = model._protos[cls].unsqueeze(0).cuda().float()

    # get covs and means for all other classes up to current task
    other_covs = [normalize_full(model._cov_mat_shrink[i]).cuda() for i in range(max_cls + 1) if
                  i != cls]
    other_means = [model._protos[i].cuda().unsqueeze(0) for i in range(max_cls + 1) if i != cls]

    # initialize module and optimizer
    module = MahaModule(cov, mean).cuda()
    op = torch.optim.Adam(module.parameters(), model.args["cov_optim_lr"])

    other_dists = [(m, c) for m, c in zip(other_means, other_covs)]

    # prepare train data
    data = torch.tensor(np.vstack(data)).cuda().float()

    # prepare validation split
    n_samples = data.shape[0]
    permuted_indices = torch.randperm(n_samples)
    train_indices = permuted_indices[:int(0.9 * n_samples)]
    val_indices = permuted_indices[int(0.9 * n_samples):]

    data_val = data[val_indices]
    data_train = data[train_indices]

    # compute mahalanobis distances to other distributions
    other_cls_maha_train = torch.stack([-mahal(data_train, m, c).detach() for m, c in other_dists], 1)
    other_cls_maha_val = torch.stack([-mahal(data_val, m, c).detach() for m, c in other_dists], 1)

    # initialize dataloader
    ds_train = TensorDataset(data_train, other_cls_maha_train)
    dl_train = DataLoader(ds_train, batch_size=model.args["cov_optim_batch_size"], shuffle=True)

    # initialize lists for historic loss values
    log_probs_val = []
    log_probs_train = []
    softmax_val = []
    softmax_train = []

    max_iters = 40  # fixed, for now
    best_model_dict = None  # variable holding best model parameters in terms of loss on validation split
    best_val_loss = torch.inf
    patience = 0
    actual_iters = max_iters

    for i in range(max_iters):
        # training loop
        for x, other in dl_train:
            log_prob, softmax = get_loss(module, x, other)
            loss = (-log_prob - model.args["optimized_cov_alpha"] * softmax).mean()

            op.zero_grad()
            loss.backward()
            op.step()

        with torch.no_grad():
            # validation
            log_prob, softmax = get_loss(module, data_val, other_cls_maha_val)
            log_probs_val.append(-log_prob.detach().cpu().mean().item())
            softmax_val.append(-softmax.detach().cpu().mean().item())

            val_loss = (-softmax).mean()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_dict = module.state_dict()
                patience = 0
            else:
                patience += 1

            # save loss history
            log_prob, softmax = get_loss(module, data_train, other_cls_maha_train)
            log_probs_train.append(-log_prob.detach().cpu().mean().item())
            softmax_train.append(-softmax.detach().cpu().mean().item())


        if patience > 3:
            actual_iters = i + 1
            break


    # plot loss history and save to file
    fig = plt.figure(figsize=[10, 10])
    ax = plt.subplot(221)
    plt.plot(np.arange(actual_iters), log_probs_train)
    ax.set_title("Log probabilities on train set")

    ax = plt.subplot(222)
    plt.plot(np.arange(actual_iters), log_probs_val)
    ax.set_title("Log probabilities on val set")

    ax = plt.subplot(223)
    plt.plot(np.arange(actual_iters), softmax_train)
    ax.set_title("Softmax loss term on train set")

    ax = plt.subplot(224)
    plt.plot(np.arange(actual_iters), softmax_val)
    ax.set_title("Softmax loss term on val set")
    ax.axhline(y=best_val_loss.detach().cpu().numpy(), xmin=0, xmax=actual_iters)

    model.args["neptune"][f"loss_histories/class_{cls}"].upload(fig)
    plt.close(fig)

    # replace current covariance matrix with the optimized one
    module.load_state_dict(best_model_dict)

    cov = module.get_cov()
    model._cov_mat[cls] = cov
    model._cov_mat_shrink[cls] = model.shrink_cov(cov)
    model._norm_cov_mat = model.normalize_cov()
    model._protos[cls] = module.mean.squeeze().detach()
