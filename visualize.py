import pickle
import os
import torch
import numpy as np
import open3d as o3d
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize and save point cloud.')
    parser.add_argument('--idx', type=int, default=0, help='Index of the input file')
    parser.add_argument('--result_dir', type=str, default="pred_results", help='Directory containing prediction results')
    args = parser.parse_args()

    idx = args.idx
    result_dir = args.result_dir
    
    with open(os.path.join(result_dir, f"{idx}.pkl"), 'rb') as f:
        pred  = pickle.load(f)
    
    sp_filename = pred['sp_filename']
    text_input = pred['text_input']
    data = torch.load(sp_filename)
    xyz = data[0]
    rgb = data[1]
    rgb = (rgb + 1.)/2.

    rgb[pred['gt_pmask'].astype(bool)] = np.array([1,0,0]) #ground truth
    rgb[(pred['pred_pmask'] > 0.5).astype(bool)] = np.array([0,1,0]) #prediction
    
    print(f"Text input: {text_input}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Save point cloud
    output_filename = f"{idx}.ply"
    o3d.io.write_point_cloud(output_filename, pcd)
    print(f"Saved point cloud to {output_filename}")

    # visualize
    #o3d.visualization.draw_geometries([pcd])

