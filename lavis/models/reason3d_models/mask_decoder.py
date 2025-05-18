import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskDecoder(nn.Module):
    def __init__(
        self,
        #decoder=None,
        d_text=512,
        num_layer=6,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        attn_mask=True,
        media=32
    ):
        super().__init__()
        self.num_layer = num_layer
        self.input_proj = nn.Sequential(nn.Linear(media, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.region_mask_proj = nn.Sequential(nn.Linear(1, media), nn.LayerNorm(media), nn.ReLU())
        self.lang_proj = nn.Linear(d_text, d_model)

        self.sa_layers = nn.ModuleList([])
        self.sa_ffn_layers = nn.ModuleList([])
        self.ca_layers = nn.ModuleList([])
        self.ca_ffn_layers = nn.ModuleList([])
        for i in range(num_layer):
            self.sa_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.sa_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
            self.ca_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.ca_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(nn.Linear(media, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.attn_mask = attn_mask
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_batches(self, x, batch_offsets):
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        if torch.is_tensor(max_len):
            max_len = max_len.item()
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(max_len - cur_len, x.shape[1]).to(x.device)], dim=0)

            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask
    
    def get_mask(self, query, mask_feats, batch_mask):
        pred_masks = torch.einsum('bnd,bmd->bnm', query, mask_feats)
        if self.attn_mask:
            attn_masks = (pred_masks.sigmoid() < 0.5).bool() # [B, 1, num_sp]
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks[torch.where(attn_masks.sum(-1) == attn_masks.shape[-1])] = False
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks = attn_masks.detach()
        else:
            attn_masks = None
        return pred_masks, attn_masks

    def prediction_head(self, query, mask_feats, batch_mask):
        query = self.out_norm(query)
        pred_scores = self.out_score(query)
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_mask)
        return pred_scores, pred_masks, attn_masks
    
    def forward(self, sp_feats, batch_offsets, text_features=None, region=None):    
        
        x = sp_feats

        if region != None:
            region_proj = self.region_mask_proj(region)
            x = x + region_proj


        batch_lap_pos_enc = None
        query = self.lang_proj(text_features)              
        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)
        inst_feats, batch_mask = self.get_batches(inst_feats, batch_offsets)
        mask_feats, _ = self.get_batches(mask_feats, batch_offsets)
        prediction_masks = []
        prediction_scores = []
        B = len(batch_offsets) - 1
        
        # 0-th prediction
        pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)
        prediction_scores.append(pred_scores)
        prediction_masks.append(pred_masks)

        # multi-round
        for i in range(self.num_layer):
            query = self.sa_layers[i](query, None)
            query = self.sa_ffn_layers[i](query)

            query, _ = self.ca_layers[i](inst_feats, query, batch_mask, attn_masks)
            query = self.ca_ffn_layers[i](query)

            pred_scores, pred_masks, _ = self.prediction_head(query, mask_feats, batch_mask)
            _, _, attn_masks = self.prediction_head(query, mask_feats, batch_mask)
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)
        return {
            'masks':
            pred_masks,
            'batch_mask':
            batch_mask,
            'scores':
            pred_scores,
            'aux_outputs': [{
                'masks': a,
                'scores': b
            } for a, b in zip(
                prediction_masks[:-1],
                prediction_scores[:-1],
            )],
        }

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_mask, attn_mask=None, pe=None):
        """
        source (B, N_p, d_model)
        batch_offsets Tensor (b, n_p)
        query Tensor (b, n_q, d_model)
        attn_masks Tensor (b, n_q, n_p)
        """
        B = query.shape[0]
        query = self.with_pos_embed(query, pe)
        k = v = source
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, query.shape[1], k.shape[1])
            output, output_weight = self.attn(query, k, v, key_padding_mask=batch_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, output_weight = self.attn(query, k, v, key_padding_mask=batch_mask)
        self.dropout(output)
        output = output + query
        self.norm(output)

        return output, output_weight # (b, n_q, d_model), (b, n_q, n_v)

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_mask=None, attn_mask=None, pe=None):
        """
        x Tensor (b, n_w, c)
        x_mask Tensor (b, n_w)
        """
        B = x.shape[0]
        q = k = self.with_pos_embed(x, pe)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, q.shape[1], k.shape[1])
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output

class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output
