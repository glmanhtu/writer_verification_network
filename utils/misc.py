import random
import re
import time
from typing import Dict, List

import cv2
import numpy as np
import torch.nn.functional as F

import torch
from torch import Tensor
import pandas as pd

from utils import wi19_evaluate


class EarlyStop:

    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.losses = []
        self.best_loss = 99999999

    def should_stop(self, loss):
        self.losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
        if len(self.losses) <= self.n_epochs:
            return False
        best_loss_pos = self.losses.index(self.best_loss)
        if len(self.losses) - best_loss_pos <= self.n_epochs:
            return False
        return True


def map_location(cuda):
    if torch.cuda.is_available() and cuda:
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'


def display_terminal(iter_start_time, i_epoch, i_train_batch, num_batches, train_dict):
    t = (time.time() - iter_start_time)
    current_time = time.strftime("%H:%M", time.localtime(time.time()))
    output = "Time {}\tBatch Time {:.2f}\t Epoch [{}]([{}/{}])\t".format(current_time, t,
                                                                         i_epoch, i_train_batch, num_batches)
    for key in train_dict:
        output += '{} {:.4f}\t'.format(key, train_dict[key])
    print(output)


def display_terminal_eval(iter_start_time, i_epoch, eval_dict):
    t = (time.time() - iter_start_time)
    output = "\nEval Time {:.2f}\t Epoch [{}] \t".format(t, i_epoch)
    for key in eval_dict:
        output += '{} {:.4f}\t'.format(key, eval_dict[key])
    print(output + "\n")


def compute_distance_matrix(data: Dict[str, Tensor], distance_fn, n_times_testing=5):
    distance_map = {}
    fragments = list(data.keys())
    for i in range(len(fragments)):
        for j in range(i, len(fragments)):
            source, target = fragments[i], fragments[j]
            n_items = max(len(data[source]), len(data[target]))

            source_features = data[source][torch.randint(len(data[source]), (n_times_testing * n_items,))]
            target_features = data[target][torch.randint(len(data[target]), (n_times_testing * n_items,))]

            # Remove self comparing cases
            same_flag = torch.isclose(source_features, target_features)
            not_same_flag = torch.logical_not(same_flag)

            # source_features = F.normalize(source_features, p=2, dim=1)
            # target_features = F.normalize(target_features, p=2, dim=1)
            distance = distance_fn(source_features[not_same_flag], target_features[not_same_flag])

            mean_distance = distance.cpu().item()
            distance_map.setdefault(source, {})[target] = mean_distance
            distance_map.setdefault(target, {})[source] = mean_distance

    matrix = pd.DataFrame.from_dict(distance_map, orient='index').sort_index()
    return matrix.reindex(sorted(matrix.columns), axis=1)


def random_query_results(distance_matrix, gt_map, dataset, letter, n_queries=5, top_k=25):
    positive_pairs, _ = gt_map
    papyrus_set_indexes = list(set(distance_matrix.index))
    fragment_queries = random.sample(papyrus_set_indexes, n_queries)
    fragment_queries = fragment_queries
    result = []
    for query in fragment_queries:
        query_result = {
            'query': query,
            'query_img': dataset.get_img_by_id(letter, query),
            'results': []
        }
        distance_result = distance_matrix[query]
        for target, distance in distance_result.items():
            in_gt = False
            if target in positive_pairs[query]:
                in_gt = True
            query_result['results'].append({
                'target': target,
                'target_img': None,
                'in_gt': in_gt,
                'distance': distance,
            })
        top_k_similarities = sorted(query_result['results'], key=lambda x: x['distance'])
        query_result['results'] = top_k_similarities[:top_k]
        for item in query_result['results']:
            item['target_img'] = dataset.get_img_by_id(letter, item['target'])

        result.append(query_result)
    return result


def get_metrics(distance_matrix, triplet_def):
    positive_pairs, _ = triplet_def
    correct_retrievals = distance_matrix.copy(deep=True) * 0
    for row in distance_matrix.index:
        for col in distance_matrix.columns:
            if col in positive_pairs[row]:
                correct_retrievals[col][row] = 1
                correct_retrievals[row][col] = 1
    correct_retrievals = correct_retrievals.to_numpy() > 0
    distance_matrix = distance_matrix.to_numpy()
    precision_at, recall_at, sorted_retrievals = wi19_evaluate.get_precision_recall_matrices(
        distance_matrix, classes=None, remove_self_column=False, correct_retrievals=correct_retrievals)

    non_singleton_idx = sorted_retrievals.sum(axis=1) > 0
    mAP = wi19_evaluate.compute_map(precision_at[non_singleton_idx, :], sorted_retrievals[non_singleton_idx, :])
    top_1 = sorted_retrievals[:, 0].sum() / len(sorted_retrievals)
    pr_a_k10 = compute_pr_a_k(sorted_retrievals, 10)
    pr_a_k100 = compute_pr_a_k(sorted_retrievals, 100)
    # roc = wi19_evaluate.compute_roc(sorted_retrievals)
    return mAP, top_1, pr_a_k10, pr_a_k100


def compute_pr_a_k(sorted_retrievals, k):
    pr_a_k = sorted_retrievals[:, :k].sum(axis=1) / np.minimum(sorted_retrievals.sum(axis=1), k)
    return pr_a_k.sum() / len(pr_a_k)


def add_description(in_img, bottom_description, left_description, green_border=False):
    white = [255, 255, 255]
    height, width, depth = in_img.shape

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_scale = 1
    font_color = (0, 0, 0)
    thickness = 1
    line_type = cv2.LINE_AA
    in_img = cv2.copyMakeBorder(in_img, 0, 20, 20, 0, cv2.BORDER_CONSTANT, value=white)

    in_img = cv2.rotate(in_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.putText(in_img, left_description,
                (10, height + 15), font, font_scale, font_color, thickness, line_type)
    in_img = cv2.rotate(in_img, cv2.ROTATE_90_CLOCKWISE)

    cv2.putText(in_img, bottom_description,
                (10, height + 15), font, font_scale, font_color, thickness, line_type)

    if green_border:
        green = [0, 255, 0]
        in_img = cv2.copyMakeBorder(in_img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=green)

    return in_img
