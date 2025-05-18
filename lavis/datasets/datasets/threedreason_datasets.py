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
from lavis.datasets.datasets.base_dataset import BaseDataset
import re
import pandas as pd
import glob
import random
from pathlib import Path
import re
from collections import defaultdict


room_type_list = ["bathroom", "bedroom", "closet", "dining room", "entryway", "familyroom", "garage", "hallway", "library", "laundryroom", "kitchen", "living room", "conference room", "lounge", "office", "porch", "game", "stairs", "toilet", "utility room", "tv", "workout", "outdoor areas", "balcony", "other room", "bar", "classroom", "dining booth", "spa", "junk"]

def get_neighbors(idx, idx_range):
    start, end = idx_range
    if idx == start:
        return [start, start + 1, start + 2]
    elif idx == end:
        return [end - 2, end - 1, end]
    else:
        return [idx - 1, idx, idx + 1]

def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

class ThreeDReasonDataset(BaseDataset):
    def __init__(self, text_processor, pts_root, ann_paths):
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

        self.sp_filenames = self.get_mp3d_sp_filenames()        

        self.short_question_list = QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.with_elastic = False
        self.aug = False        

        with open("data/matterport/mp3d_room_type.json",'r') as f:
            self.room_type = json.load(f)

        room_ids = self.room_type.keys()

        ranges = defaultdict(list)

        # Extract IDs and region numbers
        for item in room_ids:
            match = re.match(r'(.+)_region(\d+)', item)
            if match:
                id_, region_num = match.groups()
                ranges[id_].append(int(region_num))

        # Determine the range for each ID
        self.room_id_ranges = {id_: (min(nums), max(nums)) for id_, nums in ranges.items()}
        
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
        xyz, valid_idxs = self.crop(xyz)
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        if instance_label != None:
            instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
            return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label
        else:
            return xyz, xyz_middle, rgb, superpoint, semantic_label

    def transform_test(self, xyz, rgb, superpoint, semantic_label, instance_label=None):
        xyz_middle = xyz
        xyz = xyz_middle * 50
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        if instance_label != None:
            instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
            return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label
        else:
            return xyz, xyz_middle, rgb, superpoint, semantic_label

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

    def crop(self, xyz: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        r"""
        crop the point cloud to reduce training complexity

        Args:
            xyz (np.ndarray, [N, 3]): input point cloud to be cropped

        Returns:
            Union[np.ndarray, np.ndarray]: processed point cloud and boolean valid indices
        """
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([512] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while valid_idxs.sum() > 250000:
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs

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
    
    def __getitem__(self, index: int) -> Tuple:
        data = self.annotation[index]
        scan_id = data['scene_name']
        ann_id = index
        
        scene = scan_id.split("_")[0]
        all_sp_filename = [f for f in self.sp_filenames if scene in f]

        sp_filename = None
        for fn in self.sp_filenames:
            fn_short = fn.split("/")[-1].split('.')[0]
            if scan_id == fn_short:
                sp_filename = fn
                break
        assert sp_filename != None

        xyz_list = []
        xyz_middle_list = []
        rgb_list = []
        superpoint_list = []
        semantic_label_list = []
        room_mask_list = []
        room_sp_mask_list = []
        
        # load all files
        start_num = 0
        room_idx = -1
        for i, sp in enumerate(all_sp_filename):
            if sp == sp_filename:
                room_idx = i
                break
        assert room_idx != -1

        #TODO: we simply get neighbor rooms to mimic three-room house case
        idx_range = self.room_id_ranges[scene]
        room_list = get_neighbors(room_idx, idx_range)
        # room_list = [room_idx-1, room_idx, room_idx+1]

        superpoint_bias = 0

        for i,sp in enumerate(all_sp_filename):
            
            if i not in room_list:
                continue

            region = Path(sp).stem
            room = self.room_type[region]

            data = self.load(sp)
            data = self.transform_train(*data) if self.training else self.transform_test(*data)
            xyz, xyz_middle, rgb, superpoint, semantic_label = data
            
            superpoint = superpoint + start_num
            start_num = superpoint.max().item() + 1

            xyz_list.append(torch.from_numpy(xyz).long())
            xyz_middle_list.append(torch.from_numpy(xyz_middle).float())
            rgb_list.append(torch.from_numpy(rgb).float())

            superpoint_list.append(torch.from_numpy(superpoint))
            semantic_label_list.append(torch.from_numpy(semantic_label).long())
            
            if sp_filename == sp:
                room_mask_list.append(torch.zeros(xyz.shape[0]).bool())        
                room_sp_mask_list.append(torch.zeros(np.unique(superpoint).shape[0]).bool())        
            else:
                room_mask_list.append(torch.ones(xyz.shape[0]).bool())        
                room_sp_mask_list.append(torch.ones(np.unique(superpoint).shape[0]).bool())        

        coord = torch.cat(xyz_list)
        coord_float = torch.cat(xyz_middle_list)
        feat = torch.cat(rgb_list)
        superpoint = torch.cat(superpoint_list)

        assert len(torch.unique(superpoint)) == start_num

        semantic_label = torch.cat(semantic_label_list)
        room_mask = torch.cat(room_mask_list)
        room_sp_mask = torch.cat(room_sp_mask_list)
        
        room_target = None

        for room_type in room_type_list:
            if room_type in self.annotation[index]['question']:
                room_target = room_type
                break
        assert room_target != None            

        question_template = random.choice(self.short_question_list)
        caption_ori = question_template.format(description=self.annotation[index]['question'])

        object_id = self.annotation[index]['object_id']
        gt_pmask, gt_spmask = self.get_ref_mask(semantic_label, superpoint, object_id)
            
        gt_pmask[room_mask.bool()] = False
        gt_spmask[room_sp_mask.bool()] = False

        assert gt_pmask.max().item() == True
        
        answers = [random.choice(self.answer_list).format(location=room_target)]

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
            "text_input": caption_ori,
            'room_mask': ~room_mask,
            'room_sp_mask': ~room_sp_mask,
        }

    def collater(self, batch):
        ann_ids, scan_ids, coords, coords_float, feats, superpoints, object_ids, gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, lang_words, answerss, text_input_list, room_mask_list, room_sp_mask_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        batch_offsets = [0]
        n_answers = []
        superpoint_bias = 0

        j = 0
        
        num_room_batch = []

        for i, data in enumerate(batch):
            ann_id, scan_id, coord, coord_float, feat, src_superpoint, object_id, gt_pmask, gt_spmask, sp_ref_mask, lang_tokens, answers, captions, room_mask, room_sp_mask = list(data.values())
            
            superpoint = src_superpoint + superpoint_bias
            superpoint_bias = superpoint.max().item() + 1
            batch_offsets.append(superpoint_bias)

            ann_ids.append(ann_id)
            scan_ids.append(scan_id)
            coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1))

            num_room_batch.append(len(coord))

            coords_float.append(coord_float)
            feats.append(feat)
            superpoints.append(superpoint)
            
            object_ids.append(object_id)
            
            gt_pmasks.append(gt_pmask)
            gt_spmasks.append(gt_spmask)
            sp_ref_masks.append(sp_ref_mask)
            answerss.extend(answers)
            text_input_list.append(captions)
            
            n_answers.append(len(answers))
            
            room_mask_list.append(room_mask)
            room_sp_mask_list.append(room_sp_mask)

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int [B+1]

        coords = torch.cat(coords, 0)  # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        coords_float = torch.cat(coords_float, 0)  # float [B*N, 3]
        feats = torch.cat(feats, 0)  # float [B*N, 3]
        superpoints = torch.cat(superpoints, 0).long()  # long [B*N, ]
        if self.use_xyz:
            feats = torch.cat((feats, coords_float), dim=1)
        # voxelize
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), 128, None)
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
            'gt_pmasks_region': room_mask_list,
            'gt_spmasks_region': room_sp_mask_list,
            'gt_spmasks': gt_spmasks,
            'sp_ref_masks': sp_ref_masks,
            "answer": answerss,
            "text_input": text_input_list,
            'n_answers': torch.LongTensor(n_answers),
        }

    def __len__(self):
        return len(self.annotation)

QUESTION_LIST = [
    "Please identify the room first then segment the object according to the given 3D scene and the description: {description}.",
    "Given the 3D scene, provide the room first then segment the object according to the description: {description}.",
    "Respond the room mask first then the segmentation mask of the object: {description}.",
]

ANSWER_LIST = [
    "The location of {location} is [LOC]. The mask is [SEG].",
    "Sure, the location of {location} is [LOC] and the mask is [SEG].",
    "Sure, the location of {location} is [LOC]. The segmentation result is [SEG].",
    "The location of {location} is [LOC]. The mask is [SEG].",
]
