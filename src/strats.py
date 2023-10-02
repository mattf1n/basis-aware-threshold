import numpy as np
import torch
from scipy.special import softmax, logsumexp


def greedy(_, logits):
    return logits.softmax(dim="vocab").max().item()


def eta(parameter, logits):
    probs = torch.softmax(logits, dim="vocab")
    logprobs = logits - torch.logsumexp(logits, dim="vocab")
    entropy = probs @ logprobs
    alpha = np.sqrt(parameter)
    return min(parameter, alpha * torch.exp(entropy).item())


def epsilon(threshold, logits):
    return threshold


def topk(k, logits):
    probs = softmax(logits)
    argsort = np.argsort(-logits)
    return probs[argsort][:k].min()


def nucleus(p, logits):
    probs = torch.softmax(logits, dim="vocab").rename(None)
    sorted_token_ids = logits.rename(None).argsort(descending=True)
    sorted_probs_cumsum = probs[sorted_token_ids].cumsum(dim=0)
    boundary_token_id = sorted_token_ids[sorted_probs_cumsum >= p][0]
    return probs[boundary_token_id].item()


strats_dict = dict(eta=eta, epsilon=epsilon, nucleus=nucleus, greedy=greedy)
