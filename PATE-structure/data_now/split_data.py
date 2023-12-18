from typing import Tuple

import numpy as np


def split_uniformly(labels, sizes, seed):
    unique_labels = np.unique(labels)
    splits = [[] for _ in sizes]

    for label in unique_labels:
        # ��ȡÿ��������������
        class_indices = np.where(labels == label)[0]
        np.random.seed(seed)
        np.random.shuffle(class_indices)
        print(class_indices)
        # ����������������ͬ��С���Ӽ�
        start = 0
        for size, split in zip(sizes, splits):
            end = start + size
            split.extend(class_indices[start:end])
            start = end

    return splits


def split_data(x: np.array, y: np.array, splits_length: Tuple[int, int, int, int],
               num_classes, seed):
    """Split the data into test, private, and public sets.
        Note: this used to be random - now this is deterministic.
        7/14/22
        # split lengths : [n_test, n_private, n_public]
    Returns:
        _type_: data_test, data_private, data_public
    """
    np.random.seed(seed)
    n_data = len(y)
    print(n_data)
    print(splits_length)
    assert n_data == sum(splits_length)

    n_test, n_public, n_valid, n_private = splits_length

    # �����ݼ����ֳɴ�СΪ��10, 20, 40, 80������������
    sizes = [int(n_test // num_classes), int(n_public // num_classes), int(n_valid // num_classes), int(n_private // num_classes)]
    data_splits = split_uniformly(y, sizes, seed)

    # ��ӡ���
    # for i, split in enumerate(data_splits):
    #     print(f"Split {i + 1} size {sizes[i]}: {split}")
    # ---------------------------

    # print(splits_length)

    # ���ȵ��������ݵ�id
    # ʹ��idȡ����
    # print(data_splits)

    idx_test = data_splits[0]
    idx_public = data_splits[1]
    idx_valid = data_splits[2]
    idx_private = data_splits[3]
    print(f"Private: {len(idx_private)}")
    print(f"Public : {len(idx_public)}")
    print(f"Test : {len(idx_test)}")
    print(f"valid : {len(idx_valid)}")
    # print(f"Private: {(idx_private)}")
    # print(f"Public : {(idx_public)}")
    # print(f"Test : {(idx_test)}")
    # print(f"valid : {(idx_valid)}")
    data_test = x[idx_test], y[idx_test]
    data_private = x[idx_private], y[idx_private]
    data_public = x[idx_public], y[idx_public]
    data_valid = x[idx_valid], y[idx_valid]

    return data_test, data_private, data_public, data_valid
