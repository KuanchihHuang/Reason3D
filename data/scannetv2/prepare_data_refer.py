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

try:
    import segmentator
except ImportError:
    print("[WARN] Using segmentator_pytorch as segmentator. Note: The results may be slightly worse.")
    import segmentator_pytorch as segmentator

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

parser = argparse.ArgumentParser()
parser.add_argument('--data_split', help='data split (train / val / test)', default='train')
opt = parser.parse_args()

split = opt.data_split
print('data split: {}'.format(split))
files = sorted(glob.glob(split + '/*_vh_clean_2.ply'))
if opt.data_split != 'test':
    files2 = sorted(glob.glob(split + '/*_vh_clean_2.labels.ply'))
    files3 = sorted(glob.glob(split + '/*_vh_clean_2.0.010000.segs.json'))
    files4 = sorted(glob.glob(split + '/*[0-9].aggregation.json'))
    assert len(files) == len(files2)
    assert len(files) == len(files3)
    assert len(files) == len(files4), '{} {}'.format(len(files), len(files4))


def f_test(fn):
    print(fn)

    f = plyfile.PlyData().read(fn)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    mesh = o3d.io.read_triangle_mesh(fn)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = segmentator.segment_mesh(vertices, faces).numpy()

    torch.save((coords, colors, superpoint), fn[:-15] + '_refer.pth')
    print('Saving to ' + fn[:-15] + '_refer.pth')

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
    print(fn)

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
    superpoint = segmentator.segment_mesh(vertices, faces).numpy()

    torch.save((coords, colors, superpoint, sem_labels, instance_labels), fn[:-15] + '_refer.pth')
    print('Saving to ' + fn[:-15] + '_refer.pth')

p = mp.Pool(processes=mp.cpu_count())
if opt.data_split == 'test':
    p.map(f_test, files)
else:
    p.map(f, files)
p.close()
p.join()
