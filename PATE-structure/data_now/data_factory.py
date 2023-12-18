# coding=UTF8
from loguru import logger
import numpy as np
import os
from pathlib import Path

from data_now.load_datasets import load_RS_dataset, load_mnist, load_svhn, load_cifar10
# from data_now..load_datasets import load_cifar10, load_fashion_mnist, load_mnist, load_svhn, load_dataset, load_RS_dataset
from data_now.split_data import split_data
# G:\RS1130\codeNew\individualized-pate-main\individualized-pate-main\per_point_pate\data\data_factory.py

class DataFactory:
    def __init__(self, data_name: str, data_dir: Path, out_dir: Path, train_dir):
        """_summary_

        Allows to produce the private, public and test datasplits for the used dataset.

        The produced splits are written to the output directory
        under the 'data_splits' folder.

        Args:
            data_name (str): str to determine the dataset to be used (currently available: mnist, fashion_mnist, cifar10, svhn)
            data_dir (Path): Path to folder containing datasets
            out_dir (Path): Path to output folders

        Raises:
            RuntimeError: if dataset is unknown
        """
        self._loaders = {
            'mnist': load_mnist,
            'svhn': load_svhn,

            'cifar': load_cifar10,
            'NWPU-RESISC45': load_RS_dataset,
            'EuroSAT': load_RS_dataset
        }
        # TODO : classes and shape could be derived from loader
        self._classes = {
            'mnist': 10,
            'svhn': 10,
            'cifar': 10,
            'NWPU-RESISC45': 45,
            "EuroSAT": 10
        }
        self._shape = {
            'mnist': (1, 28, 28),
            'svhn': (3, 32, 32),
            'NWPU-RESISC45': (3, 224, 224),
            "EuroSAT": (3, 64, 64)
        }

        if data_name not in self._loaders.keys():
            raise RuntimeError(
                f"Dataset '{data_name}' not known to DataFactory.")

        self.data_dir = data_dir
        self.out_dir = out_dir
        self.data_name = data_name
        self.train_dir = train_dir
        # self.test_dir = test_dir

        # will cached upon calling functions
        self._data_public = None
        self._data_private = None
        self._data_test = None
        self._data_valid = None

    @property
    def n_classes(self):
        return self._classes[self.data_name]

    @property
    def example_shape(self):
        return self._shape[self.data_name]

    @property
    def splits_dir(self):
        a = os.path.join(self.out_dir, 'data_splits')
        return a

    def public_split_path(self, seed):
        a = os.path.join(self.splits_dir, f'data_{seed}_public.npz')
        return a

    def private_split_path(self, seed):
        a = os.path.join(self.splits_dir, f'data_{seed}_private.npz')
        return a

    def test_split_path(self, seed):
        a = os.path.join(self.splits_dir, f'data_{seed}_test.npz')
        return a

    def valid_split_path(self, seed):
        a = os.path.join(self.splits_dir, f'data_{seed}_valid.npz')
        return a

    # def load_dataset(self):
    #     return self._loaders[self.data_name](self.data_dir, self.dataclasses)
    def load_dataset(self):
        return self._loaders[self.data_name](self.data_dir, )

    def load_rs_dataset(self):
        return self._loaders[self.data_name](self.data_dir, self.train_dir)

    def write_splits(self,
                     seeds: int,
                     n_test: int,
                     n_public: int,
                     n_private: int,
                     n_valid: int,
                     num_classes: int):
        """
        This method creates the private, public and tests splits from the requested datasets.

        The splits are created using seeded pseudo-randomness.
        Afterwards, it stores all three subsets in separate CSV-files.
        Args: 
            seeds:       list of seeds that determine pseudo-random number generation; for each seed one split is created
            n_test:      int to specify the number of data points in the test subset
            n_public:    int to specify the number of data points in the public subset
            balanced:    bool to specify if the ratios of labels are to be kept in all subsets
        """

        # check for / create data splits folder
        if not os.path.isdir(self.splits_dir):
            os.makedirs(self.splits_dir)

        # collect seeds for which data splits were not created yet
        missing_seeds = []
        for seed in seeds:
            if os.path.isfile(self.public_split_path(seed)) \
                    and os.path.isfile(self.private_split_path(seed)) \
                    and os.path.isfile(self.test_split_path(seed)) \
                    and os.path.isfile(self.valid_split_path(seed)):

                logger.info(
                    f'Private, public, and test data splits seeded by {seed} already exist.'
                )
                continue
            else:
                missing_seeds.append(seed)

        # return if data splits already created for all seeds
        if len(missing_seeds) == 0:
            return

        if self.data_name == "EuroSAT" or self.data_name == "NWPU-RESISC45":
            x, y = self.load_rs_dataset()
        else:
            # 加载数据集
            (x, y), _, _ = self.load_dataset()

        for seed in missing_seeds:
            np.random.seed(seed)
            print(len(x))
            print(len(y))
            print(n_test, n_private, n_public, n_valid)
            data_test, data_private, data_public, data_valid = split_data(
                x=x,
                y=y,
                splits_length=(n_test, n_public, n_valid, n_private),
                num_classes=num_classes, seed=seed)

            x_test, y_test = data_test
            np.savez(self.test_split_path(seed),
                     features=x_test,
                     labels=y_test)
            x_private, y_private = data_private
            np.savez(self.private_split_path(seed),
                     features=x_private,
                     labels=y_private)
            x_public, y_public = data_public
            np.savez(self.public_split_path(seed),
                     features=x_public,
                     labels=y_public)
            x_valid, y_valid = data_valid
            np.savez(self.valid_split_path(seed),
                     features=x_valid,
                     labels=y_valid)

    def _load_split(self, path: Path):
        """
        Read a csv split file and return numpy arrays of features and labels.
        """
        data = np.load(path)
        return data['features'], data['labels']

    def data_public(self, seed):
        """
        Return the public split data (features and labels) for a seed.

        The data will be read and cached on the first call.
        """
        if self._data_public is None:
            path = self.public_split_path(seed)
            self._data_public = self._load_split(path)
        return self._data_public

    def data_private(self, seed):
        """
        Return the private split data (features and labels) for a seed.

        The data will be read and cached on the first call.
        """
        if self._data_private is None:
            path = self.private_split_path(seed)
            self._data_private = self._load_split(path)
        return self._data_private

    def data_test(self, seed):
        """
        Return the test split data (features and labels) for a seed.

        The data will be read and cached on the first call.
        """
        if self._data_test is None:
            path = self.test_split_path(seed)
            self._data_test = self._load_split(path)
        return self._data_test

    def data_valid(self, seed):
        """
        Return the test split data (features and labels) for a seed.

        The data will be read and cached on the first call.
        """
        if self._data_valid is None:
            path = self.valid_split_path(seed)
            self._data_valid = self._load_split(path)
        return self._data_valid

    def n_private_per_label(self, seed):
        """
        Return the number of private data points per label.
        """
        _, y_private = self.data_private(seed)
        n_per_label = {
            label: len(y_private[y_private == label])
            for label in np.unique(y_private)
        }
        return n_per_label
