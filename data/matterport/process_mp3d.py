import glob
import os
import numpy as np
import plyfile
import torch
import pandas as pd
import open3d as o3d
import multiprocessing as mp
try:
    import segmentator
except ImportError:
    print("[WARN] Using segmentator_pytorch as segmentator. Note: The results may be slightly worse.")
    import segmentator_pytorch as segmentator

num_classes = 2000
out_dir = "mp3d_data"
os.makedirs(out_dir, exist_ok=True)

def process_one_scene(fn):
    '''process one scene.'''

    scene_name = fn.split('/')[-3]
    region_name = fn.split('/')[-1].split('.')[0]
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    colors = np.ascontiguousarray(v[:, -3:]) / 127.5 - 1

    category_id = a['face']['category_id']
    
    category_id[category_id==-1] = 0
    mapped_labels = category_id

    triangles = a['face']['vertex_indices']
    vertex_labels = np.zeros((coords.shape[0], num_classes+1), dtype=np.int32)

    # calculate per-vertex labels
    for row_id in range(triangles.shape[0]):
        for i in range(3):
            vertex_labels[triangles[row_id][i],
                            mapped_labels[row_id]] += 1

    vertex_labels = np.argmax(vertex_labels, axis=1)

    mesh = o3d.io.read_triangle_mesh(fn)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = segmentator.segment_mesh(vertices, faces).numpy()

    torch.save((coords, colors, superpoint, vertex_labels), os.path.join(out_dir,  scene_name+'_' + region_name + '.pth'))

    print(fn)

matterport_path = "./scans"

scene_list = os.listdir(matterport_path)
files = []
for scene in scene_list:
    files = files + glob.glob(os.path.join(matterport_path, scene, 'region_segmentations', '*ply'))

p = mp.Pool(processes=mp.cpu_count())
p.map(process_one_scene, files)
p.close()
p.join()
