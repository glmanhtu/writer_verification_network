import argparse
import os

import pandas as pd
from timm.utils import AverageMeter

from utils.data_utils import load_triplet_file
from utils.misc import get_metrics, get_metrics_v2

parser = argparse.ArgumentParser()

parser.add_argument('--similarity_file', type=str, help='Path to the similarity file', required=True)
parser.add_argument('--with_likely', action='store_true')

args = parser.parse_args()

# Load similarity matrix from CSV file
similarity_matrix = pd.read_csv(args.similarity_file, index_col=0)
similarity_matrix.index = similarity_matrix.index.map(str)
dir_path = os.path.dirname(os.path.realpath(__file__))

triplet_def = {
    'α': load_triplet_file(os.path.join(dir_path, 'BT120220128.triplet'), similarity_matrix.columns, args.with_likely),
    'ε': load_triplet_file(os.path.join(dir_path, 'Eps20220408.triplet'), similarity_matrix.columns, args.with_likely),
    'μ': load_triplet_file(os.path.join(dir_path, 'mtest.triplet'), similarity_matrix.columns, args.with_likely),
}

m_ap_meter = AverageMeter()
top1_meter = AverageMeter()
pr_k10_meter = AverageMeter()
pr_k100_meter = AverageMeter()
for letter in triplet_def:
    tms = []
    for tm in triplet_def[letter][0]:
        if len(triplet_def[letter][0][tm]) > 1:
            tms.append(tm)
    categories = sorted(tms)
    subset = similarity_matrix.loc[categories, categories]
    m_ap, top1, pr_a_k10, pr_a_k100 = get_metrics_v2(1 - subset, triplet_def[letter])
    m_ap_meter.update(m_ap)
    top1_meter.update(top1)
    pr_k10_meter.update(pr_a_k10)
    pr_k100_meter.update(pr_a_k100)

print(f'mAP {m_ap_meter.avg:.3f}\t' f'Top 1 {top1_meter.avg:.3f}\t' 
      f'Pr@k10 {pr_k10_meter.avg:.3f}\t' f'Pr@k100 {pr_k100_meter.avg:.3f}')
