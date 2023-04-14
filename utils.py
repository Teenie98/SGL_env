import torch
import numpy as np
from params import args
import torch.nn.functional as F
from collections import Counter


def sp_mat_to_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()


def inner_product(x1, x2):
    return torch.sum(torch.mul(x1, x2), dim=-1)


def compute_bpr_loss(x1, x2):
    # return -torch.sum(torch.log((x1.view(-1) - x2.view(-1)).sigmoid() + 1e-8))
    return -torch.sum(F.logsigmoid(x1 - x2))


def compute_infoNCE_loss(x1, x2, temp):
    return torch.logsumexp((x2 - x1[:, None]) / temp, dim=1)


def compute_reg_loss(w1, w2, w3):
    return 0.5 * torch.sum(torch.pow(w1, 2) + torch.pow(w2, 2) + torch.pow(w3, 2))


# env add

def env_compute_bpr_loss(x1, x2, weight):
    return -torch.sum(F.logsigmoid(x1 - x2) * weight)


def env_compute_infoNCE_loss(x1, x2, temp, weight):
    return weight * torch.logsumexp((x2 - x1[:, None]) / temp, dim=1)


def env_compute_reg_loss(w1, w2, w3, weight):
    return 0.5 * torch.sum(torch.sum((torch.pow(w1, 2) + torch.pow(w2, 2) + torch.pow(w3, 2)), dim=-1) * weight)


def compute_metric(ratings, test_item):
    hit = 0
    DCG = 0.
    iDCG = 0.

    _, shoot_index = torch.topk(ratings, args.k)
    shoot_index = shoot_index.cpu().tolist()

    for i in range(len(shoot_index)):
        if shoot_index[i] in test_item:
            hit += 1
            DCG += 1 / np.log2(i + 2)
        if i < test_item.size()[0]:
            iDCG += 1 / np.log2(i + 2)

    recall = hit / test_item.size()[0]
    NDCG = DCG / iDCG

    return recall, NDCG


def env_compute_metric(ratings, test_item, all_item_env, num_envs=5):
    hit = torch.zeros(num_envs)
    DCG = torch.zeros(num_envs)
    iDCG = torch.zeros(num_envs)

    _, shoot_index = torch.topk(ratings, args.k)
    shoot_index = shoot_index.cpu().tolist()
    env_shoot_index = [[] for i in range(num_envs)]
    for s in shoot_index:
        env_shoot_index[int(all_item_env[s])].append(s)

    env_test_item = [[] for i in range(num_envs)]
    for item in test_item:
        env_test_item[int(all_item_env[item.item()])].append(item.item())
    env_num = torch.tensor([len(env_test_item[i]) for i in range(num_envs)])

    for e in range(num_envs):
        shoot = env_shoot_index[e]
        test = env_test_item[e]
        if len(shoot) == 0 or len(test) == 0:
            continue

        for i, s in enumerate(shoot):
            if s in test:
                hit[e] += 1
                DCG[e] += 1 / np.log2(i + 2)
            if i < env_num[e]:
                iDCG[e] += 1 / np.log2(i + 2)

    NDCG = DCG / iDCG
    return hit, NDCG, env_num


def compute_irm_loss(pos_scores, env_labels, num_envs=5):
    mean = torch.zeros(num_envs)
    delta = torch.zeros(num_envs)

    for env in range(num_envs):
        mean[env] = pos_scores[env_labels == env].mean()

    for env in range(num_envs):
        delta[env] = ((pos_scores[env_labels == env] - mean[env]) ** 2).sum()

    irm_loss = delta.sum()

    return irm_loss
