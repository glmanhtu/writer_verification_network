import json

import numpy as np
from PIL import Image

letter_ascii = {
    'α': 'alpha',
    'ε': 'epsilon',
    'μ': 'm'
}


def padding_image(img, new_size, color=(0, 0, 0)):
    old_image_height, old_image_width, channels = img.shape
    new_image_width, new_image_height = new_size
    result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = img
    return result


def resize_image(image: Image.Image, scale_factor: float) -> Image.Image:
    """Resize image by scale factor."""
    if scale_factor == 1:
        return image
    return image.resize((round(image.width * scale_factor), round(image.height * scale_factor)),
                        resample=Image.BICUBIC)


def bincount_app(image_array):
    a_2d = image_array.reshape(-1, image_array.shape[-1])
    col_range = (256, 256, 256)  # generically : a2D.max(0)+1
    a_1d = np.ravel_multi_index(a_2d.T, col_range)
    return np.unravel_index(np.bincount(a_1d).argmax(), col_range)


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


def add_items_to_group(items, groups):
    """
    Add list of items to groups,
    If there are no groups that match with the items, create a new group and put those item in this new group
    If there is only one matching group, add all these items to this group
    If there is more than one matching group, add all these items to the first group, then move items from
                other matching groups to this first group
    """
    reference_group = {}
    for g_id, group in enumerate(groups):
        for fragment_id in items:
            if fragment_id in group and g_id not in reference_group:
                reference_group[g_id] = group

    if len(reference_group) > 0:
        reference_ids = list(reference_group.keys())
        for fragment_id in items:
            reference_group[reference_ids[0]].add(fragment_id)
        for g_id in reference_ids[1:]:
            for fragment_id in reference_group[g_id]:
                reference_group[reference_ids[0]].add(fragment_id)
            del groups[g_id]
    else:
        groups.append(set(items))


def get_all_tms(triplet_file):
    all_tms = []
    with open(triplet_file) as f:
        triplet_filter = json.load(f)
    for item in triplet_filter['relations']:
        current_tm = item['category']
        all_tms.append(current_tm)
        for second_item in item['relations']:
            second_tm = second_item['category']
            if current_tm == '' or second_tm == '':
                continue
            all_tms.append(second_tm)
    return set(all_tms)


def load_triplet_file(filter_file, all_tms, with_likely=False):
    all_tms = set(all_tms)
    with open(filter_file) as f:
        triplet_filter = json.load(f)
    positive_groups, negative_pairs = [], {}
    positive_pairs = {}
    mapping = {}
    for item in triplet_filter['relations']:
        current_tm = item['category']
        mapping[current_tm] = {}
        for second_item in item['relations']:
            second_tm = second_item['category']
            relationship = second_item['relationship']
            mapping[current_tm][second_tm] = relationship

    for item in triplet_filter['histories']:
        current_tm, second_tm = item['category'], item['secondary_category']
        if current_tm in all_tms and second_tm in all_tms:
            relationship = mapping[current_tm][second_tm]
            if relationship == 4:
                negative_pairs.setdefault(current_tm, set([])).add(second_tm)
                negative_pairs.setdefault(second_tm, set([])).add(current_tm)
            if with_likely and relationship == 3:
                negative_pairs.setdefault(current_tm, set([])).add(second_tm)
                negative_pairs.setdefault(second_tm, set([])).add(current_tm)
            if relationship == 1:
                add_items_to_group([current_tm, second_tm], positive_groups)
            if with_likely and relationship == 2:
                positive_pairs.setdefault(current_tm, set([])).add(second_tm)
                positive_pairs.setdefault(second_tm, set([])).add(current_tm)

    for group in positive_groups:
        for tm in group:
            for tm2 in group:
                positive_pairs.setdefault(tm, set([])).add(tm2)

    for tm in all_tms:
        positive_pairs.setdefault(tm, {tm})
        negative_pairs.setdefault(tm, set([]))
    # for current_tm in missing_tm:
    #     print(f'TM {current_tm} is not available on the training dataset')

    return positive_pairs, negative_pairs
