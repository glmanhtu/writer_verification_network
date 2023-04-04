import argparse

from utils.data_utils import get_all_tms, load_triplet_file

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, help='Path to the first triplet')
parser.add_argument('--dest', type=str, help='Path to the second triplet')
args = parser.parse_args()

# Getting all TMs from both source and dest triplet files
source_tms = get_all_tms(args.source)
dest_tms = get_all_tms(args.dest)
all_tms = set(list(source_tms) + list(dest_tms))

# Collecting the group of positives and pairs of negatives
dest_positive_groups, dest_negatives = load_triplet_file(args.dest, all_tms)
dest_pos_map = {}
for group in dest_positive_groups:
    for tm in group:
        dest_pos_map[tm] = group

# Collecting the group of positives and pairs of negatives
source_positive_groups, source_negatives = load_triplet_file(args.source, all_tms)

# Compute the matching score of positive pairs from source compared to dest
matching, not_matching = 0, 0
for group in source_positive_groups:
    group = list(group)
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            tm_1, tm_2 = group[i], group[j]
            if tm_2 in dest_pos_map[tm_1]:
                matching += 1
            else:
                not_matching += 1

matching_score = matching / (matching + not_matching)
print(f'Positive matching percentage: {matching_score}')


# Compute the matching score of negative pairs from source compared to dest
matching, not_matching = 0, 0
for tm in source_negatives:
    for neg_tm in source_negatives[tm]:
        if neg_tm in dest_negatives[tm]:
            matching += 1
        else:
            not_matching += 1

matching_score = matching / (matching + not_matching)
print(f'Negative matching percentage: {matching_score}')
