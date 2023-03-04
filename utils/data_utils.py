import json

import numpy as np
from PIL import Image

letter_ascii = {
    'α': 'alpha',
    'ε': 'epsilon',
    'μ': 'm'
}


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
    reference_group = None
    for group in groups:
        for fragment_id in items:
            if fragment_id in group:
                reference_group = group
                break
        if reference_group is not None:
            break
    if reference_group is not None:
        for fragment_id in items:
            reference_group.add(fragment_id)
    else:
        groups.append(set(items))


def load_triplet_file(filter_file, all_tms):
    all_tms = set(all_tms)
    with open(filter_file) as f:
        triplet_filter = json.load(f)
    positive_groups, negative_pairs = [], {}
    missing_tm = set([])
    for item in triplet_filter['relations']:
        current_tm = item['category']
        if current_tm not in all_tms:
            missing_tm.add(current_tm)
            continue
        for second_item in item['relations']:
            second_tm = second_item['category']
            if current_tm == '' or second_tm == '':
                continue
            if second_tm not in all_tms:
                missing_tm.add(second_tm)
                continue
            relationship = second_item['relationship']
            if relationship == 4:
                negative_pairs.setdefault(current_tm, []).append(second_tm)
                negative_pairs.setdefault(second_tm, []).append(current_tm)
            if relationship == 1:
                add_items_to_group([current_tm, second_tm], positive_groups)

    for tm in all_tms:
        add_items_to_group([tm], positive_groups)
    for current_tm in missing_tm:
        print(f'TM {current_tm} is not available on the training dataset')

    return positive_groups, negative_pairs
