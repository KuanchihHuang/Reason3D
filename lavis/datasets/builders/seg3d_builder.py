"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.threedrefer_datasets import ThreeDReferDataset
from lavis.datasets.datasets.threedreason_datasets import ThreeDReasonDataset

@registry.register_builder("3d_refer")
class ThreeDReferBuilder(BaseDatasetBuilder):
    train_dataset_cls = ThreeDReferDataset
    eval_dataset_cls = ThreeDReferDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/3dseg/defaults.yaml"}


@registry.register_builder("3d_reason")
class ThreeDReasonBuilder(BaseDatasetBuilder):
    train_dataset_cls = ThreeDReasonDataset
    eval_dataset_cls = ThreeDReasonDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/3dreason/defaults.yaml"}
