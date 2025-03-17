#!/usr/bin/env python3
"""
Created on Sun Mar 16 21:15:48 2025

@author: Dilaksan Thillaithevan
"""

import pickle
import numpy as np

# https://www.cs.toronto.edu/~kriz/cifar.html
CIFAR_PATH = "./datasets/cifar-100-python"


class Dataset:
    def __init__(self, dataset_name: str, sub_dataset_name: str = None) -> None:
        assert dataset_name in DATASETS
        self.data = DATASETS[dataset_name]()
        self.dataset_type = b"coarse_labels"

        # if sub_dataset_name is not None:
        #     # labels = [_.decode('UTF-8') for _ in self.label_names]
        #     labels = self.label_names
        #     assert sub_dataset_name in labels
        #     label_id = [i for (i,v) in enumerate(labels) if v == sub_dataset_name][0]
        #     valid_indices = np.where(self.data['test'][self.dataset_type] == label_id)[0]

        #     self.data['test'][b'data'] = self.data['test'][b'data'][valid_indices]
        #     self.data['test'][b'data'] = self.data['test'][b'data'][valid_indices]

    @property
    def train_data(
        self,
    ) -> np.ndarray:
        return self.data["train"][b"data"]

    @property
    def test_data(
        self,
    ) -> np.ndarray:
        return self.data["test"][b"data"]

    @property
    def test_labels(
        self,
    ) -> np.ndarray:
        return np.array(self.data["test"][self.dataset_type]).reshape(-1, 1)

    @property
    def train_labels(
        self,
    ) -> np.ndarray:
        return np.array(self.data["train"][self.dataset_type]).reshape(-1, 1)

    @property
    def label_names(
        self,
    ) -> list[str]:
        return self.data["meta"][self.dataset_type]


def load_CIFAR() -> dict:
    try:
        test = unpickle(f"{CIFAR_PATH}/test")
        train = unpickle(f"{CIFAR_PATH}/train")
        meta = unpickle(f"{CIFAR_PATH}/meta")
    except:
        raise FileNotFoundError(
            "Download CIFAR-100 dataset from https://www.cs.toronto.edu/~kriz/cifar.html and place it in ./datasets"
        )

    out = {"test": test, "train": train, "meta": meta}

    return out


def load_sonar() -> dict:
    import pandas as pd

    data = pd.read_csv("./datasets/sonar/sonar.csv", header=None)
    X, Y = data.iloc[:, 0:60].to_numpy(), data.iloc[:, 60].to_list()
    Y = np.array([0 if _ == "R" else 1 for _ in Y], dtype=int)
    return X, Y


def unpickle(file):

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


DATASETS = {"CIFAR-10": load_CIFAR, "sonar": load_sonar}
