import numpy as np
import torch
import scipy.cluster.vq as vq
import cvxpy as cp
import functools, sys, json, os, operator, time, typing
import diskcache

REJECT = "\033[9m{}\033[0m"
ACCEPT = "\033[32m{}\033[0m"
FAST = "\033[34m{}\033[0m"
BLACK = "{}"
RED = "\033[31m{}\033[0m"
GREEN = ACCEPT


def get_variable_constraint_sizes(size, ratio):
    constraints = int(np.sqrt(size / ratio))
    variables = int(np.sqrt(size * ratio))
    return constraints, variables


def minpow(a, b, c):
    return pow(min(a, pow(c, 1 / b)), b)


def set_params(parameters, values):
    for param, val in zip(parameters, values):
        param.value = val


def get_params(
    model_name: str,
    variable_size: int,
    constraint_size: int,
    token_id: int,
    p_hat: torch.Tensor,
    frac: float = 0.5,
    device: typing.Union[torch.device, str, None] = None,
    eliminate_constraints: typing.Optional[typing.Callable[..., torch.Tensor]] = None,
):
    device = p_hat.device
    top_k_size = int(variable_size * frac)
    top_k = get_top_k(model_name, top_k_size, token_id, device=device)
    fused_size = variable_size - top_k_size
    label = kmeans(model_name, fused_size)
    variable_membership = (
        torch.tensor(label, device=device)
        == torch.arange(fused_size, device=device)[:, None]
    ).rename("fuse", "vocab")
    top_k_membership = torch.isin(
        torch.arange(len(label), device=device), top_k
    ).rename("vocab")
    fusion_mask = (
        variable_membership.rename(None)
        .logical_and(~top_k_membership.rename(None))
        .to(p_hat.dtype)
        .rename("fuse", "vocab")
    )
    fused_p_hat = fusion_mask @ p_hat
    embedding = eliminate_constraints(model_name, constraint_size, device=device)
    fused_embedding = (
        fusion_mask
        @ (embedding * p_hat.align_to("vocab", "constraint"))
        / (fused_p_hat + (fused_p_hat == 0)).align_to("fuse", "constraint")
    )
    kmeans_embedding = torch.cat(
        (embedding.rename(None)[top_k], fused_embedding.rename(None))
    ).rename("var", "constraint")
    kmeans_p_hat = torch.cat(
        (p_hat.rename(None)[top_k], fused_p_hat.rename(None))
    ).rename("var")
    return (
        kmeans_embedding.rename(None).cpu().numpy(),
        kmeans_p_hat.rename(None).cpu().numpy(),
    )


def get_params_constraint_elim_only(
    model_name: str,
    constraint_size: int,
    token_id: int,
    p_hat: torch.Tensor,
    frac: float = 0.5,
    device: typing.Union[torch.device, str, None] = None,
    eliminate_constraints: typing.Optional[typing.Callable[..., torch.Tensor]] = None,
):
    ...


def svd(model_name, constraint_size, device=None):
    embedding = get_embedding(model_name).to(device)
    Vh = (
        torch.linalg.svd(embedding.rename(None), full_matrices=False)
        .Vh[:constraint_size, :]
        .rename("constraint", "hidden")
    )
    return embedding @ Vh.align_to("hidden", "constraint")


@functools.cache
def clustersum(model_name, constraint_size, device=None):
    embedding = get_embedding(model_name)
    normalized = normalize(embedding, dim=0)
    _, label = vq.kmeans2(
        normalized.align_to("hidden", "vocab").rename(None).cpu(),
        constraint_size,
        minit="points",
    )
    grouping_mask = (
        torch.tensor(label)[:, None] == torch.arange(constraint_size)
    ).rename("hidden", "constraint")
    small_embedding = embedding @ grouping_mask.to(embedding.device, embedding.dtype)
    return small_embedding.to(device)


@functools.cache
def kmeans(model_name, fused_size, norm=False):
    embedding = get_embedding(model_name)
    _, label = vq.kmeans2(
        (embedding if not norm else normalize(embedding, dim=1)).rename(None).cpu(),
        fused_size,
        minit="points",
        seed=0,
    )
    return label


def get_problem(hidden_size, size):
    p_hat = cp.Parameter(size)
    embedding = cp.Parameter((size, hidden_size))
    p = cp.Variable(size - 1, nonneg=True)
    sigma = cp.Variable(nonneg=True)
    objective = cp.Minimize(sigma)
    constraints = [
        embedding[1:].T @ p == embedding.T @ p_hat,
        cp.sum(p) == cp.sum(p_hat),
        p <= p_hat[1:] * sigma,
    ]
    problem = cp.Problem(objective, constraints)
    return problem, (embedding, p_hat), (p, sigma)


@functools.cache
def get_feasibility_problem(hidden_size, vocab_size):
    p_hat = cp.Parameter(vocab_size)
    embedding = cp.Parameter((vocab_size, hidden_size))
    sigma = cp.Parameter(nonneg=True)
    p = cp.Variable(vocab_size - 1, nonneg=True)
    objective = cp.Minimize(0)
    constraints = [
        embedding[1:].T @ p == embedding.T @ p_hat,
        cp.sum(p) == cp.sum(p_hat),
        p <= p_hat[1:] * sigma,
    ]
    problem = cp.Problem(objective, constraints)
    return problem, (embedding, p_hat, sigma), (p,)


def get_token_specific_problem(embedding, hat_p, sigma, token):
    p = cp.Variable(hat_p.shape)
    constraints = [
        embedding.T @ p == embedding.T @ hat_p,
        cp.sum(p) == 1,
        p[token] == 0,
        p <= hat_p * sigma,
    ]
    problem = cp.Problem(cp.Minimize(0), constraints)
    return problem


def is_in_support_constraint_elim_only(
    model_name, constraint_size, p_hat, cutoff, token_id, **kwargs
):
    embedding = svd(model_name, constraint_size, device="cpu").rename(None).numpy()
    problem = get_token_specific_problem(
        embedding, p_hat.rename(None).cpu().numpy(), cutoff, token_id
    )
    try:
        problem.solve(**kwargs)
        return problem.status == "infeasible"
    except cp.error.SolverError:
        return False


def is_in_support(
    problem, params, cutoff_param, get_params, p_hat, cutoff, token_id, **kwargs
):
    values = get_params(
        token_id, p_hat.refine_names("vocab"), eliminate_constraints=svd
    )
    set_params(params, values)
    cutoff_param.value = cutoff
    try:
        problem.solve(**kwargs)
        return problem.status == "infeasible"
    except cp.error.SolverError:  # type: ignore
        return False


@functools.cache
def get_top_k(model_name, size, token_id, device=None):
    embeddings = get_embedding(model_name, device=device)
    similarities = embeddings @ embeddings[token_id]
    argsorted = similarities.rename(None).argsort(descending=True, dim=-1)
    token_id_mask = argsorted == token_id
    rearranged = torch.cat((argsorted[token_id_mask], argsorted[~token_id_mask]))
    assert token_id == rearranged[0]
    return rearranged[:size]


def normalize(embeddings, **kwargs):
    return embeddings / torch.linalg.norm(
        embeddings.rename(None), keepdims=True, **kwargs
    )


@functools.cache
def get_embedding(model_name, device=None):
    model, _ = load_model(model_name)
    return (
        model.get_output_embeddings().weight.data.to(device).rename("vocab", "hidden")
    )


@functools.cache
def load_model(model_name, device=None, **kwargs):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
