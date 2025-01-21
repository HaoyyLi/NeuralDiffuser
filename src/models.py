import os
import torch
from torch import nn

from diffusers.models.vae import Decoder
class Voxel2StableDiffusionModel(torch.nn.Module):
    def __init__(self, h=4096, n_blocks=4, use_cont=False, ups_mode='4x'):
        super().__init__()

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, h, bias=False),
                nn.LayerNorm(h),
                nn.SiLU(inplace=True),
                nn.Dropout(0.25)
            ) for _ in range(n_blocks)
        ])
        self.ups_mode = ups_mode
        if ups_mode=='4x':
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 64)
            
            self.upsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256],
                layers_per_block=1,
            )

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()
        
        if ups_mode=='8x':  # prev best
            self.lin1 = nn.Linear(h, 16384, bias=False)
            self.norm = nn.GroupNorm(1, 256)
            
            self.upsampler = Decoder(
                in_channels=256,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()
        
        if ups_mode=='16x':
            self.lin1 = nn.Linear(h, 8192, bias=False)
            self.norm = nn.GroupNorm(1, 512)
            
            self.upsampler = Decoder(
                in_channels=512,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D", "UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256, 256, 512],
                layers_per_block=1,
            )
            self.maps_projector = nn.Identity()

            if use_cont:
                self.maps_projector = nn.Sequential(
                    nn.Conv2d(64, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=False),
                    nn.GroupNorm(1,512),
                    nn.ReLU(True),
                    nn.Conv2d(512, 512, 1, bias=True),
                )
            else:
                self.maps_projector = nn.Identity()

    def forward(self, x, return_transformer_feats=False):
        residual = x
        for res_block in self.mlp:
            x = res_block(x)
            x = x + residual
            residual = x
        x = x.reshape(len(x), -1)
        x = self.lin1(x)  # bs, 4096

        if self.ups_mode == '4x':
            side = 16
        if self.ups_mode == '8x':
            side = 8
        if self.ups_mode == '16x':
            side = 4
        
        # decoder
        x = self.norm(x.reshape(x.shape[0], -1, side, side).contiguous()).float()
        # with torch.cuda.amp.autocast(enabled=False):
        if return_transformer_feats:
            return self.upsampler(x), self.maps_projector(x).flatten(2).permute(0,2,1)
        return self.upsampler(x)

class BrainNetwork(nn.Module):
    def __init__(self, h=4096, out_dim=768, seq_len=2, n_blocks=4, drop=.15, clip_size=768, clip_scale=1):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.clip_scale = clip_scale
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])
        
        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True) 
        self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)
 
    def projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )
    
    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )
    
    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),  # Token mixing
        )

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self.mlp(seq_len, seq_len, drop)  # Channel mixing
        )
        
    def forward(self, x):
        # make empty tensors
        c= torch.Tensor([0.])
        
        # Mixer blocks
        residual1 = x
        residual2 = x.permute(0,2,1)
        for block1, block2 in zip(self.mixer_blocks1,self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0,2,1)
            
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0,2,1)
            
        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
        if self.clip_scale>0:
            c = self.clip_proj(backbone)

        return backbone, c


class BrainNetwork2(nn.Module):
    def __init__(self, h=4096, in_dim=15724, out_dim=768, seq_len=2, n_blocks=4, drop=.15, clip_size=768, blurry_recon=True, clip_scale=1):
        super().__init__()
        # same size with input matrix (for adding with input matrix)
        pos_emb = torch.zeros(seq_len, h)
        pos = torch.arange(0, seq_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position
        _2i = torch.arange(0, h, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        pos_emb[:, 0::2] = torch.sin(pos / (10000 ** (_2i / h)))
        pos_emb[:, 1::2] = torch.cos(pos / (10000 ** (_2i / h)))

        self.pos_emb = nn.parameter.Parameter(pos_emb, requires_grad=False)
        # compute positional encoding to consider positional information of words

        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.blurry_recon = blurry_recon
        self.clip_scale = clip_scale
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])
        
        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True) 
        self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)
        
        if self.blurry_recon:
            self.blin1 = nn.Linear(h*seq_len,4*28*28,bias=True)
            self.bdropout = nn.Dropout(.3)
            self.bnorm = nn.GroupNorm(1, 64)
            self.bupsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[32, 64, 128],
                layers_per_block=1,
            )
            self.b_maps_projector = nn.Sequential(
                nn.Conv2d(64, 512, 1, bias=False),
                nn.GroupNorm(1,512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias=False),
                nn.GroupNorm(1,512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias=True),
            )
            
    def projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )
    
    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )
    
    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),  # Token mixing
        )

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self.mlp(seq_len, seq_len, drop)  # Channel mixing
        )
        
    def forward(self, x):
        # make empty tensors
        c,b = torch.Tensor([0.]), torch.Tensor([[0.],[0.]])
        x += self.pos_emb
        # Mixer blocks
        residual1 = x
        residual2 = x.permute(0,2,1)
        for block1, block2 in zip(self.mixer_blocks1,self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0,2,1)
            
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0,2,1)
            
        x = x.reshape(x.size(0), -1)

        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
        if self.clip_scale>0:
            c = self.clip_proj(backbone)

        if self.blurry_recon:
            b = self.blin1(x)
            b = self.bdropout(b)
            b = b.reshape(b.shape[0], -1, 7, 7).contiguous()
            b = self.bnorm(b)
            b_aux = self.b_maps_projector(b).flatten(2).permute(0,2,1)
            b_aux = b_aux.view(len(b_aux), 49, 512)
            b = (self.bupsampler(b), b_aux)
        
        return backbone, c, b

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

import math
class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask=None):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class BrainNetwork_Transformer(nn.Module):
    def __init__(self, h=4096, in_dim=15724, n_head=8, out_dim=768, seq_len=2, n_blocks=4, drop=.15, clip_size=768, blurry_recon=True, clip_scale=1):
        super().__init__()
        # same size with input matrix (for adding with input matrix)
        pos_emb = torch.zeros(seq_len, h)
        pos = torch.arange(0, seq_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position
        _2i = torch.arange(0, h, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        pos_emb[:, 0::2] = torch.sin(pos / (10000 ** (_2i / h)))
        pos_emb[:, 1::2] = torch.cos(pos / (10000 ** (_2i / h)))

        self.pos_emb = nn.parameter.Parameter(pos_emb, requires_grad=False)
        # compute positional encoding to consider positional information of words


        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.blurry_recon = blurry_recon
        self.clip_scale = clip_scale

        self.blocks = nn.ModuleList([EncoderLayer(d_model=h,
                                                  ffn_hidden=h,
                                                  n_head=n_head,
                                                  drop_prob=drop)
                                     for _ in range(n_blocks)])
        
        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True) 
        self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)
        
        if self.blurry_recon:
            self.blin1 = nn.Linear(h*seq_len,4*28*28,bias=True)
            self.bdropout = nn.Dropout(.3)
            self.bnorm = nn.GroupNorm(1, 64)
            self.bupsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[32, 64, 128],
                layers_per_block=1,
            )
            self.b_maps_projector = nn.Sequential(
                nn.Conv2d(64, 512, 1, bias=False),
                nn.GroupNorm(1,512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias=False),
                nn.GroupNorm(1,512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias=True),
            )
    def trans_enc(self, d_model, ffn_hidden, n_head, drop_prob):
        return nn.Sequential(
            nn.LayerNorm(ffn_hidden),
            nn.GELU(),
            EncoderLayer(d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, drop_prob=drop_prob),
        )
    def projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )
        
    def forward(self, x):
        # make empty tensors
        c,b = torch.Tensor([0.]), torch.Tensor([[0.],[0.]])
        
        # add pos emb
        x += self.pos_emb
        
        
        # Mixer blocks
        residual = x
        for block in self.blocks:
            x = block(x) + residual
            residual = x
        
        x = x.reshape(x.size(0), -1)

        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
        if self.clip_scale>0:
            c = self.clip_proj(backbone)

        if self.blurry_recon:
            b = self.blin1(x)
            b = self.bdropout(b)
            b = b.reshape(b.shape[0], -1, 7, 7).contiguous()
            b = self.bnorm(b)
            b_aux = self.b_maps_projector(b).flatten(2).permute(0,2,1)
            b_aux = b_aux.view(len(b_aux), 49, 512)
            b = (self.bupsampler(b), b_aux)
        
        return backbone, c, b
    
class BrainNetwork_prez(nn.Module):
    def __init__(self, h=4096, in_dim=15724, out_dim=768, seq_len=2, n_blocks=4, drop=.15, clip_size=768, blurry_recon=True, clip_scale=1):
        super().__init__()
        self.seq_len = seq_len
        self.h = h
        self.clip_size = clip_size
        self.blurry_recon = blurry_recon
        self.clip_scale = clip_scale
        self.mixer_blocks1 = nn.ModuleList([
            self.mixer_block1(h, drop) for _ in range(n_blocks)
        ])
        self.mixer_blocks2 = nn.ModuleList([
            self.mixer_block2(seq_len, drop) for _ in range(n_blocks)
        ])
        
        # Output linear layer
        self.backbone_linear = nn.Linear(h * seq_len, out_dim, bias=True) 
        self.clip_proj = self.projector(clip_size, clip_size, h=clip_size)
        
        if self.blurry_recon:
            self.blin1 = nn.Linear(h*seq_len,4*64*64,bias=True)
            self.bdropout = nn.Dropout(.3)
            self.bnorm = nn.GroupNorm(1, 64)
            self.bupsampler = Decoder(
                in_channels=64,
                out_channels=4,
                up_block_types=["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
                block_out_channels=[64, 128, 256],
                layers_per_block=1,
            )
            self.b_maps_projector = nn.Sequential(
                nn.Conv2d(64, 512, 1, bias=False),
                nn.GroupNorm(1,512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias=False),
                nn.GroupNorm(1,512),
                nn.ReLU(True),
                nn.Conv2d(512, 512, 1, bias=True),
            )
            
    def projector(self, in_dim, out_dim, h=2048):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Linear(h, out_dim)
        )
    
    def mlp(self, in_dim, out_dim, drop):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(out_dim, out_dim),
        )
    
    def mixer_block1(self, h, drop):
        return nn.Sequential(
            nn.LayerNorm(h),
            self.mlp(h, h, drop),  # Token mixing
        )

    def mixer_block2(self, seq_len, drop):
        return nn.Sequential(
            nn.LayerNorm(seq_len),
            self.mlp(seq_len, seq_len, drop)  # Channel mixing
        )
        
    def forward(self, x):
        # make empty tensors
        c,b = torch.Tensor([0.]), torch.Tensor([[0.],[0.]])
        
        # Mixer blocks
        residual1 = x
        residual2 = x.permute(0,2,1)
        for block1, block2 in zip(self.mixer_blocks1,self.mixer_blocks2):
            x = block1(x) + residual1
            residual1 = x
            x = x.permute(0,2,1)
            
            x = block2(x) + residual2
            residual2 = x
            x = x.permute(0,2,1)
            
        x = x.reshape(x.size(0), -1)
        backbone = self.backbone_linear(x).reshape(len(x), -1, self.clip_size)
        if self.clip_scale>0:
            c = self.clip_proj(backbone)

        if self.blurry_recon:
            b = self.blin1(x)
            b = self.bdropout(b)
            b = b.reshape(b.shape[0], -1, 16, 16).contiguous()
            b = self.bnorm(b)
            b_aux = self.b_maps_projector(b).flatten(2).permute(0,2,1)
            b_aux = b_aux.view(len(b_aux), 49, 512)
            b = (self.bupsampler(b), b_aux)
        
        return backbone, c, b

if __name__=='__main__':
    model = BrainNetwork_Transformer(h=1024, n_head=8, out_dim=77*768, seq_len=9, n_blocks=4).cuda()
    b = torch.randn(2, 9,1024).cuda()
    backbone_, clip_, blur_ = model(b)
    print(backbone_.shape, clip_.shape, blur_[0].shape, blur_[1].shape)


    # el = EncoderLayer(d_model=1024, ffn_hidden=1024, n_head=8, drop_prob=0.15).cuda()
    # b = torch.randn(2, 9,1024).cuda()
    # print(el(b).shape)
