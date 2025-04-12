"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch
import numpy as np
import os.path as osp
import pointgroup_ops
import math
from PIL import Image
from PIL import ImageFile
import torch_scatter
from typing import Dict, Sequence, Tuple, Union
import pandas as pd

from lavis.datasets.datasets.base_dataset import BaseDataset
import glob
import random
from .scannet200_constants import CLASS_LABELS_200

def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

class ThreeDReasonDataset(BaseDataset):
    def __init__(self, text_processor, pts_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(text_processor, pts_root, ann_paths)
        self.scene_ids = {}

        self.use_xyz = True
        self.mode = 4
 
        if 'train' in ann_paths[0]:
            self.prefix = 'train'
            self.training = True
        if 'val' in ann_paths[0]:
            self.prefix = 'val'
            self.training = False
        self.with_label = True
        self.scannet200_sp_filenames = self.get_scannet200_sp_filenames()
        self.mp3d_sp_filenames = self.get_mp3d_sp_filenames()

        new_data = []
            
        for data in self.annotation:
    
            assert data['type'] in ['scannet', 'mp3d']
            sp_filename = None
            filenames = self.scannet200_sp_filenames if data['type'] == 'scannet' else self.mp3d_sp_filenames
    
            if data['type'] == 'scannet':
                # For scannet, check if scene_id is in filename
                for fn in filenames:
                    if data['scene_id'] in fn:
                        sp_filename = fn
                        break
            else:  # mp3d
                # For mp3d, extract basename and compare with scene_id
                for fn in filenames:
                    fn_base = os.path.splitext(os.path.basename(fn))[0]
                    if data['scene_id'] == fn_base:
                        sp_filename = fn
                        break
             
            assert sp_filename is not None
            data['sp_filename'] = sp_filename
            new_data.append(data)

        self.annotation = new_data
        self.short_question_list = QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.with_elastic = False
        self.aug = True      

    def get_scannet200_sp_filenames(self):
        filenames = glob.glob(osp.join(self.pts_root, 'scannetv2', self.prefix, '*' + '_reason.pth'))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames)
        return filenames

    def get_mp3d_sp_filenames(self):
        mp3d_root = os.path.join(self.pts_root, 'matterport')
        scene_list = process_txt(os.path.join(mp3d_root,'scenes_'+self.prefix+'.txt'))
        mp3d_pointcept = os.path.join(mp3d_root,'mp3d_data')
        scene_files = []
        for scene in scene_list:
            scene_files = scene_files + glob.glob(os.path.join(mp3d_pointcept, scene+"*"))
        return scene_files            
        
    def load(self, filename):
        if self.with_label:
            return torch.load(filename)
        else:
            xyz, rgb, superpoint = torch.load(filename)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            return xyz, rgb, superpoint, dummy_sem_label, dummy_inst_label
        
    def transform_train(self, xyz, rgb, superpoint, semantic_label, instance_label=None):
        #TODO: normalize
        xyz_max = np.max(xyz,axis=0)[np.newaxis, ...]
        xyz = xyz - xyz_max
        if self.aug:
            xyz_middle = self.data_aug(xyz, True, True, True)
        else:
            xyz_middle = xyz.copy()
        rgb += np.random.randn(3) * 0.1
        xyz = xyz_middle * 50 
        if self.with_elastic:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)
        xyz = xyz - xyz.min(0)
        valid_idxs = xyz.min(1) >= 0
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        #instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, superpoint, semantic_label

    def transform_test(self, xyz, rgb, superpoint, semantic_label, instance_label=None):
        #TODO: normalize
        xyz_max = np.max(xyz,axis=0)[np.newaxis, ...]
        xyz = xyz - xyz_max
        xyz_middle = xyz
        xyz = xyz_middle * 50
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        return xyz, xyz_middle, rgb, superpoint, semantic_label
        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def data_aug(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)

    def elastic(self, xyz, gran, mag):
        """Elastic distortion (from point group)

        Args:
            xyz (np.ndarray): input point cloud
            gran (float): distortion param
            mag (float): distortion scalar

        Returns:
            xyz: point cloud with elastic distortion
        """
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(xyz_):
            return np.hstack([i(xyz_)[:, None] for i in interp])

        return xyz + g(xyz) * mag

    def get_cropped_inst_label(self, instance_label: np.ndarray, valid_idxs: np.ndarray) -> np.ndarray:
        r"""
        get the instance labels after crop operation and recompact

        Args:
            instance_label (np.ndarray, [N]): instance label ids of point cloud
            valid_idxs (np.ndarray, [N]): boolean valid indices

        Returns:
            np.ndarray: processed instance labels
        """
        instance_label = instance_label[valid_idxs]
        j = 0
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label
    
    def get_ref_mask(self, instance_label, superpoint, object_id):
        ref_lbl = instance_label == object_id
        gt_spmask = torch_scatter.scatter_mean(ref_lbl.float(), superpoint, dim=-1)
        gt_spmask = (gt_spmask > 0.5).float()
        gt_pmask = ref_lbl.float()
        return gt_pmask, gt_spmask
    
    def __len__(self):
        return len(self.scanrefer)
    
    def __getitem__(self, index: int) -> Tuple:
        data = self.annotation[index]
        scan_id = data["scene_id"]
        object_id = int(data["object_id"])
        object_name = data["object_name"]
        question = data['question']
        data_type = data['type']
        ann_id = index 
        question_template = random.choice(self.short_question_list)

        sp_filename = data["sp_filename"]
        data = self.load(sp_filename)
        data = self.transform_train(*data) if self.training else self.transform_test(*data)
        xyz, xyz_middle, rgb, superpoint, semantic_label = data

        unique_class = np.unique(semantic_label)
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle).float()
        feat = torch.from_numpy(rgb).float()
        superpoint = torch.from_numpy(superpoint)
        semantic_label = torch.from_numpy(semantic_label).long()
        question_template = random.choice(self.short_question_list)
        question = question_template.format(description=question)

        gt_pmask, gt_spmask = self.get_ref_mask(semantic_label, superpoint, object_id)
        assert gt_pmask.max().item() == True
        answers = [random.choice(self.answer_list)]

        return {
            'ann_ids': ann_id,
            'scan_ids': scan_id,
            'coord': coord,
            'coord_float': coord_float,
            'feat': feat,
            'superpoint': superpoint,
            'object_id': object_id,
            'gt_pmask': gt_pmask,
            'gt_spmask': gt_spmask,
            'sp_ref_mask': None,
            'lang_tokens': None,
            'answers': answers,
            "text_input": question,
            "data_type": data_type,
            "sp_filename": sp_filename,
        }

    def collater(self, batch):
        ann_ids, scan_ids, coords, coords_float, feats, superpoints, object_ids, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, lang_words, answerss, text_input_list, data_type_list, sp_filename_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        batch_offsets = [0]
        n_answers = []
        superpoint_bias = 0

        for i, data in enumerate(batch):
            ann_id, scan_id, coord, coord_float, feat, src_superpoint, object_id, gt_pmask, gt_spmask, sp_ref_mask, lang_tokens, answers, captions, data_type, sp_filename = list(data.values())
            
            superpoint = src_superpoint + superpoint_bias
            superpoint_bias = superpoint.max().item() + 1
            batch_offsets.append(superpoint_bias)

            ann_ids.append(ann_id)
            scan_ids.append(scan_id)
            coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            superpoints.append(superpoint)
            
            object_ids.append(object_id)
            
            gt_pmasks.append(gt_pmask)
            gt_spmasks.append(gt_spmask)
            sp_ref_masks.append(sp_ref_mask)
            answerss.extend(answers)
            text_input_list.append(captions)
            data_type_list.append(data_type)
            sp_filename_list.append(sp_filename)

            n_answers.append(len(answers))

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int [B+1]
        coords = torch.cat(coords, 0)  # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        coords_float = torch.cat(coords_float, 0)  # float [B*N, 3]
        feats = torch.cat(feats, 0)  # float [B*N, 3]
        superpoints = torch.cat(superpoints, 0).long()  # long [B*N, ]
        if self.use_xyz:
            feats = torch.cat((feats, coords_float), dim=1)
        # voxelize
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), 128, None)  # long [3]
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coords, len(batch), self.mode)

        return {
            'ann_ids': ann_ids,
            'scan_ids': scan_ids,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'spatial_shape': spatial_shape,
            'feats': feats,
            'superpoints': superpoints,
            'batch_offsets': batch_offsets,
            'object_ids': object_ids,
            'gt_pmasks': gt_pmasks,
            'gt_spmasks': gt_spmasks,
            'sp_ref_masks': sp_ref_masks,
            "answer": answerss,
            "text_input": text_input_list,
            'lang_tokenss': None,
            'lang_masks': None,
            'n_answers': torch.LongTensor(n_answers),
            'data_types': data_type_list,
            'sp_filenames': sp_filename_list,
        }

    def __len__(self):
        return len(self.annotation)

QUESTION_LIST = [
    "Please segment the object according to the given 3D scene and the description: {description}.",
    "Given the 3D scene, segment this object according to the description: {description}.",
    "Respond the segmentation mask of the object: {description}.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]
