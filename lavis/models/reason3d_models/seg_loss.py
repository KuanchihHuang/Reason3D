import gorilla
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.nn.utils.rnn import pad_sequence
from typing import Union


def get_iou(inputs: torch.Tensor, targets: torch.Tensor, pad_mask: Union[torch.Tensor, None]=None, pred_confidence=0.5):
    '''
    padding modified
    '''
    if pad_mask is not None:
        inputs = inputs.sigmoid()*pad_mask
    else:
        inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= pred_confidence)
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score

def get_iou_prob(inputs: torch.Tensor, targets: torch.Tensor, pad_mask: Union[torch.Tensor, None]=None):
    '''
    padding modified
    '''
    if pad_mask is not None:
        inputs = inputs*pad_mask
    else:
        inputs = inputs
    # thresholding
    binarized_inputs = (inputs >= 0.5)#.float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score

@torch.jit.script
def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()


#@torch.jit.script
def dice_loss(
    inputs,
    targets,
    pad_mask
):
    """
    padding modified
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pad_mask: A float tensor with the same shape as inputs. Stores the binary, 0 for padding, 1 for non-padding.
    """
    if pad_mask is not None:
        inputs = inputs.sigmoid()*pad_mask
    else:
        inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)  # why+1？
    return loss.mean()

@torch.jit.script
def dice_loss_prob(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pad_mask: Union[torch.Tensor, None]=None
):
    """
    padding modified
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pad_mask: A float tensor with the same shape as inputs. Stores the binary, 0 for padding, 1 for non-padding.
    """
    if pad_mask is not None:
        inputs = inputs*pad_mask
    else:
        inputs = inputs
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #proposals, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #proposals) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #proposals, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        weights = weights.unsqueeze(-1)
        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


@gorilla.LOSSES.register_module()
class Criterion(nn.Module):

    def __init__(
        self,
        loss_weight=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_fun='bce'
    ):
        super().__init__()
        self.loss_fun = loss_fun
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, layer, aux_outputs, pad_masks, gt_spmasks):
        loss_out = {}

        pred_scores = aux_outputs['scores'].squeeze()
        pred_masks = aux_outputs['masks'].squeeze()
        tgt_padding = pad_sequence(gt_spmasks, batch_first=True)

        # score loss
        with torch.no_grad():
            tgt_scores = get_iou(pred_masks, tgt_padding.float(), pad_masks)
        score_mask = (tgt_scores>0.5)
        if score_mask.sum() > 0:
            score_loss = torch.masked_select(F.mse_loss(pred_scores, tgt_scores, reduction='none'), score_mask).mean()
        else:
            score_loss = torch.tensor(0.0, device=pred_scores.device)

        # mask loss
        mask_bce_loss = F.binary_cross_entropy_with_logits(pred_masks, tgt_padding.float(), reduction='none')
        mask_bce_loss = (mask_bce_loss*pad_masks).sum(-1) / pad_masks.sum(-1)
        mask_bce_loss = mask_bce_loss.mean()

        mask_dice_loss = dice_loss(pred_masks, tgt_padding.float(), pad_masks)

        loss_out['score_loss'] = score_loss
        loss_out['mask_bce_loss'] = mask_bce_loss
        loss_out['mask_dice_loss'] = mask_dice_loss

        loss = (
            self.loss_weight[0] * mask_bce_loss + self.loss_weight[1] * mask_dice_loss + self.loss_weight[2] * score_loss)

        loss_out = {f'layer_{layer}_' + k: v for k, v in loss_out.items()}
        return loss, loss_out

    def forward(self, pred, gt_pmasks, gt_spmasks, sp_ref_masks=None):
    
        loss_out = {}

        pred_scores = pred['scores'].squeeze()
        pred_masks = pred['masks'].squeeze()

        pad_masks = ~pred['batch_mask'].squeeze()
        tgt_padding = pad_sequence(gt_spmasks, batch_first=True)
        # score loss
        with torch.no_grad():
            tgt_scores = get_iou(pred_masks, tgt_padding.float(), pad_masks)
        score_mask = (tgt_scores>0.5)
        if score_mask.sum() > 0:
            score_loss = torch.masked_select(F.mse_loss(pred_scores, tgt_scores, reduction='none'), score_mask).mean()
        else:
            score_loss = torch.tensor(0.0, device=pred_scores.device)

        # sample loss
        if sp_ref_masks is not None:
            # [B, M]
            ref_padding = pad_sequence(sp_ref_masks, batch_first=True)
            # [B, M]
            ref_scores = pred['ref_scores']

            if self.loss_fun=='focal':
                sample_criterion = SigmoidFocalClassificationLoss()
                cls_weights = pad_masks.float()
                cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
                cls_weights /= torch.clamp(cls_normalizer, min=1.0)
                # focal loss
                sample_loss = sample_criterion(ref_scores.unsqueeze(-1), ref_padding.unsqueeze(-1).float(), weights=cls_weights)
                sample_loss = (sample_loss.squeeze(-1)*pad_masks).sum(-1) # / pad_masks.sum(-1)
                sample_loss = sample_loss.mean()
                
            elif self.loss_fun=='bce':
                # bce loss
                sample_loss = F.binary_cross_entropy_with_logits(ref_scores, ref_padding.float(), reduction='none')
                sample_loss = (sample_loss*pad_masks).sum(-1) / pad_masks.sum(-1)
                sample_loss = sample_loss.mean()
            else:
                raise NotImplementedError
            
        # mask loss
        mask_bce_loss = F.binary_cross_entropy_with_logits(pred_masks, tgt_padding.float(), reduction='none')
        mask_bce_loss = (mask_bce_loss*pad_masks).sum(-1) / pad_masks.sum(-1)
        mask_bce_loss = mask_bce_loss.mean()

        # dice loss
        mask_dice_loss = dice_loss(pred_masks, tgt_padding.float(), pad_masks)

        loss_out['score_loss'] = score_loss
        loss_out['mask_bce_loss'] = mask_bce_loss
        loss_out['mask_dice_loss'] = mask_dice_loss
        if sp_ref_masks is not None:
            loss_out['sample_loss'] = sample_loss
            loss = (
                self.loss_weight[0] * mask_bce_loss + self.loss_weight[1] * mask_dice_loss +
                self.loss_weight[2] * score_loss + self.loss_weight[3] * sample_loss)
        else:   
            loss = (
                self.loss_weight[0] * mask_bce_loss + self.loss_weight[1] * mask_dice_loss +
                self.loss_weight[2] * score_loss)
        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss_i, loss_out_i = self.get_layer_loss(i, aux_outputs, pad_masks, gt_spmasks)
                loss += loss_i
                loss_out.update(loss_out_i)

        loss_out['loss'] = loss

        return loss, loss_out
