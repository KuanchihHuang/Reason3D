import functools
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean
from .backbone import ResidualBlock, UBlock

class PointExtractor(nn.Module):
    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        pool='mean',
        fix_module=[],
        pretrained=None
    ):
        super().__init__()

        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            ))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        self.pool = pool

        self.init_weights()

        for module in fix_module:
            if '.' in module:
                module, params = module.split('.')
                module = getattr(self, module)
                params = getattr(module, params)
                for param in params.parameters():
                    param.requires_grad = False
            else:
                module = getattr(self, module)
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        return self.feat(**batch)

    def feat(self, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, superpoints, batch_offsets, **kwargs):        
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        x = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = x.features[p2v_map.long()]  # (B*N, media)
        # superpoint pooling
        if self.pool == 'mean':
            x = scatter_mean(x, superpoints, dim=0)  # (B*M, media)
        elif self.pool == 'max':
            x, _ = scatter_max(x, superpoints, dim=0)  # (B*M, media)
        return x
    
