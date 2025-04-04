"""Modified from SparseConvNet data preparation: https://github.com/facebookres
earch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py."""

import argparse
import glob
import json
import multiprocessing as mp
import numpy as np
import open3d as o3d
import plyfile
import torch
from scannet200_constants import VALID_CLASS_IDS_200, VALID_CLASS_IDS_20
import pandas as pd

try:
    import segmentator
except ImportError:
    print("[WARN] Using segmentator_pytorch as segmentator. Note: The results may be slightly worse.")
    import segmentator_pytorch as segmentator


IGNORE_INDEX=-100
CLASS_IDS200 = VALID_CLASS_IDS_200
CLASS_IDS20 = VALID_CLASS_IDS_20


labels_pd = pd.read_csv(
    "scannetv2-labels.combined.tsv",
    sep="\t",
    header=0,
)


def point_indices_from_group(seg_indices, group, labels_pd):
    group_segments = np.array(group["segments"])
    label = group["label"]

    # Map the category name to id
    label_id20 = labels_pd[labels_pd["raw_category"] == label]["nyu40id"]
    label_id20 = int(label_id20.iloc[0]) if len(label_id20) > 0 else 0
    label_id200 = labels_pd[labels_pd["raw_category"] == label]["id"]
    label_id200 = int(label_id200.iloc[0]) if len(label_id200) > 0 else 0

    # Only store for the valid categories
    if label_id20 in CLASS_IDS20:
        label_id20 = CLASS_IDS20.index(label_id20)
    else:
        label_id20 = IGNORE_INDEX

    if label_id200 in CLASS_IDS200:
        label_id200 = CLASS_IDS200.index(label_id200)
    else:
        label_id200 = IGNORE_INDEX

    # get points, where segment indices (points labelled with segment ids) are in the group segment list
    point_idx = np.where(np.isin(seg_indices, group_segments))[0]
    return point_idx, label_id20, label_id200


# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train / val)', default='train')
opt = parser.parse_args()

split = opt.data_split

assert split in ['train', 'val']

print('data split: {}'.format(split))
files = sorted(glob.glob(split + '/*_vh_clean_2.ply'))
if opt.data_split != 'test':
    files2 = sorted(glob.glob(split + '/*_vh_clean_2.labels.ply'))
    files3 = sorted(glob.glob(split + '/*_vh_clean_2.0.010000.segs.json'))
    files4 = sorted(glob.glob(split + '/*[0-9].aggregation.json'))
    assert len(files) == len(files2)
    assert len(files) == len(files3)
    assert len(files) == len(files4), '{} {}'.format(len(files), len(files4))


def read_aggre(name):
    f = open(name, 'r')
    results = {}
    d = json.load(f)
    l = d['segGroups']
    for i in l:
        for s in i['segments']: 
            results[s] = i['id'] 
    return results

def read_segs(name, aggregation):
    f = open(name, 'r')
    d = json.load(f)
    indices = np.array(d['segIndices'])
    results = np.zeros_like(indices) - 1
    for i in aggregation:
        m = indices == i 
        results[m] = aggregation[i] 
    return results

def f(fn):
    fn2 = fn[:-3] + 'labels.ply'
    fn3 = fn[:-15] + '_vh_clean_2.0.010000.segs.json'
    fn4 = fn[:-15] + '.aggregation.json'

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    f2 = plyfile.PlyData().read(fn2)
    sem_labels = remapper[np.array(f2.elements[0]['label'])]

    instance_labels = read_segs(fn3, read_aggre(fn4))
    mesh = o3d.io.read_triangle_mesh(fn)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))


    #NOTE
    with open(fn3) as f:
        segments = json.load(f)
        seg_indices = np.array(segments['segIndices'])

    with open(fn4) as f:
        aggregation = json.load(f)
        seg_groups = np.array(aggregation['segGroups'])

    #Generate new_label     
    semantic_gt20 = np.ones((vertices.shape[0])) * -100
    semantic_gt200 = np.ones((vertices.shape[0])) * -100
    instance_ids = np.ones((vertices.shape[0])) * -100

    for group in seg_groups:
        point_idx, label_id20, label_id200 = point_indices_from_group(
            seg_indices, group, labels_pd
        )

        semantic_gt20[point_idx] = label_id20
        semantic_gt200[point_idx] = label_id200
        instance_ids[point_idx] = group["id"]

    superpoint = segmentator.segment_mesh(vertices, faces).numpy()
    torch.save((coords, colors, superpoint, semantic_gt200, instance_labels), fn[:-15] + '_reason.pth')
    print('Saving to ' + fn[:-15] + '_reason.pth')

p = mp.Pool(processes=mp.cpu_count())
p.map(f, files)
p.close()
p.join()

def point_indices_from_group(seg_indices, group, labels_pd):
    group_segments = np.array(group["segments"])
    label = group["label"]

    # Map the category name to id
    label_id20 = labels_pd[labels_pd["raw_category"] == label]["nyu40id"]
    label_id20 = int(label_id20.iloc[0]) if len(label_id20) > 0 else 0
    label_id200 = labels_pd[labels_pd["raw_category"] == label]["id"]
    label_id200 = int(label_id200.iloc[0]) if len(label_id200) > 0 else 0

    # Only store for the valid categories
    if label_id20 in CLASS_IDS20:
        label_id20 = CLASS_IDS20.index(label_id20)
    else:
        label_id20 = IGNORE_INDEX

    if label_id200 in CLASS_IDS200:
        label_id200 = CLASS_IDS200.index(label_id200)
    else:
        label_id200 = IGNORE_INDEX

    # get points, where segment indices (points labelled with segment ids) are in the group segment list
    point_idx = np.where(np.isin(seg_indices, group_segments))[0]
    return point_idx, label_id20, label_id200
