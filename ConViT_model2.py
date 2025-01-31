
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:56:21 2025

@author: Haliz369
I would like to express my gratitude to Junjie Zhang (https://github.com/ZJier) for 
sharing the source code of the MVAHN model. Some lines of this script are extracted 
from their code.

"""

#%%"""#**Conv block**"""

import torch.nn.functional as F
import torch.nn as nn

class DensnetConvBlock(nn.Module):
    def __init__(self, nb_input_channels, growth_rate, dropout_rate=None):
        super(DensnetConvBlock, self).__init__()

        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate


        self.conv = nn.Conv2d(in_channels=nb_input_channels, out_channels=growth_rate, kernel_size=3, padding=1, bias=False)

        self.batch_norm = nn.BatchNorm2d(growth_rate)

        if self.dropout_rate:
            self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):


        x = self.conv(x)
        x = self.batch_norm(x)
        x=F.relu(x)

        # Adding dropout
        if self.dropout_rate:
            x = self.dropout(x)

        return x

import torchvision

class DenseBlock(nn.Module):
    def __init__(self, nb_input_channels, nb_layers, growth_rate, dropout_rate=None):
        super(DenseBlock, self).__init__()

        self.nb_layers = nb_layers
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate

        self.squeeze_excite_blocks = nn.ModuleList()
        self.convolution_blocks = nn.ModuleList()

        for i in range(nb_layers):
            # Add squeeze-and-excite block
            SE = torchvision.ops.SqueezeExcitation(input_channels= nb_input_channels , squeeze_channels=int(0.25 * nb_input_channels))
            self.squeeze_excite_blocks.append(SE)

            # Add convolution block
            CB = DensnetConvBlock( nb_input_channels = nb_input_channels, growth_rate= self.growth_rate, dropout_rate = self.dropout_rate)
            self.convolution_blocks.append(CB)

            nb_input_channels += growth_rate  # Update total number of channels


    def forward(self, x):
        for i in range(self.nb_layers):
            # Apply squeeze-and-excite
            se = self.squeeze_excite_blocks[i](x)

            # Apply convolution block
            cb = self.convolution_blocks[i](se)

            # Concatenate the outputs
            x = torch.cat((cb, x), dim=1)

        return x


#%% """**Embeddings**"""

# import einops
from einops.layers.torch import Rearrange
import torch


class PatchEmbeddings(nn.Module):

    def __init__(self,  image_dim: int, emb_dim: int):
        super().__init__()
        # patch_size = 1 #pixel
        self.patchify = Rearrange(
            "b c (h p1) (w p2) -> b (h w) c p1 p2",
            p1=1, p2=1)

        self.flatten = nn.Flatten(start_dim=2)
        self.proj = nn.Linear(in_features=image_dim, out_features=emb_dim, bias=False) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patchify(x)
        x = self.flatten(x)
        x = self.proj(x)

        return x


class PositionalEmbeddings(nn.Module):

    def __init__(self, num_seq: int, emb_dim: int):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(num_seq, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos

class LocalEmbeddings(nn.Module):

    def __init__(self, image_dim: int ,patch_size: int, emb_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=image_dim, out_channels=image_dim, kernel_size=patch_size, groups=image_dim, bias=False)
        self.conv2 = nn.Conv2d(in_channels=image_dim, out_channels=emb_dim, kernel_size=1, bias=False)
                
        self.patchify = Rearrange(
            "b c (h p1) (w p2) -> b (h w) c p1 p2",
            p1=1, p2=1)

        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.conv2(self.conv(x))

        return self.flatten(self.patchify(x))


#%%"""**Transformer Encoder**"""

import einops
# import torch
# # import math
# import torch.nn as nn
# # import torch.nn.functional as F
# # from einops.layers.torch import Rearrange
# import torch.fft


class Conv_MHSA(nn.Module):
#Convolutional multi-head self-attention mechanism
    def __init__(self, emb_dim: int, head_dim: int, num_heads: int, img_dim: int, dropkey):
        super().__init__()
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        # self.num_patch = num_patch # num patch in one dimension for example an input image of 9*9 with a patch size of 1 pix will resulted in 3 patches in x dimension and 3 patches in y (resulting in 3*3=9 patches in total)
        # self.patch_size = patch_size
        self.num_heads = num_heads
        self.inner_dim = head_dim * num_heads
        self.img_dim = img_dim
        self.dropkey = dropkey

        self.scale = head_dim ** -0.5
        self.attn = nn.Softmax(dim=-1)

        self.qkv = nn.Linear(in_features=emb_dim, out_features=3*emb_dim, bias=False)

        ##### depth sep conv

        self.conv1 = nn.Conv2d(self.img_dim, self.img_dim, kernel_size=3, padding=1, groups=self.img_dim, bias=False)
        self.conv2 = nn.Conv2d(self.img_dim, self.emb_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(emb_dim)


        self.patchify = Rearrange(
            "b c (h p1) (w p2) -> b (h w) c p1 p2",
            p1=1, p2=1)
        self.flatten = nn.Flatten(start_dim=2)


    def forward(self, x: torch.Tensor, y: torch.Tensor): # , y: torch.Tensor
        b, n, d = x.shape

        qkv = self.qkv(x)
        # qkvc-->[b, num_seq, num_patch]

        qkv = qkv.chunk(3, dim=-1)
        # qkv-->q, k, v, c-->[b, num_seq, emb_dim]

        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv)
        # q, k, v-->[b, num_heads, num_seq, head_dim]

        scores = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        # Dot product of q and k

        scores = scores * self.scale
        # Scale scores [b, num_heads, num_seq, num_seq] (similarity matrix)

        # Use DropKey as a regularizer
        # dropkey=0.2
        if self.dropkey:
            m_r = torch.ones_like(scores) * self.dropkey
            scores = scores + torch.bernoulli(m_r) * -1e12

        attn = self.attn(scores)
        # Normalize scores to pdist [b, num_heads, num_seq, num_seq]

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        # Apply attention to values [b, num_heads, num_seq, head_dim]

        out = einops.rearrange(out, "b h n d -> b n (h d)")
        # Reshape to [b, num_seq, emb_dim=num_heads*head_dim]


       # Depthwise+Seprabale conv
        y = F.relu(self.bn(self.conv2(self.conv1(y))))

        y2 = self.flatten(self.patchify(y))


        out = out + y2


        return out, y


class MLP(nn.Module):

    def __init__(self, emb_dim: int, MLP_dim: int):
        super().__init__()

        self.mlp1 = nn.Linear(in_features=emb_dim, out_features=MLP_dim, bias=True)
        self.mlp2 = nn.Linear(in_features=MLP_dim, out_features=emb_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class CTB(nn.Module):
    # Convolutional-Transformer Block (CTB)
    def __init__(self, emb_dim, num_layers, num_heads, head_dim, image_dim, MLP_dim, dropkey_rate, att_drop_rate=None):
        super(CTB, self).__init__()

        self.att_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.att_layers.append(nn.ModuleList([
                nn.LayerNorm(emb_dim),
                Conv_MHSA(emb_dim, head_dim, num_heads,  image_dim, dropkey_rate),
                nn.Dropout(att_drop_rate)
            ]))
            image_dim=emb_dim

            self.mlp_layers.append(nn.ModuleList([
                nn.LayerNorm(emb_dim),
                MLP(emb_dim, MLP_dim),
                nn.Dropout(att_drop_rate)
            ]))



    def forward(self, x, y):
        holder=[]
        for (attn_norm, attn_mh, attn_dropout), (mlp_norm, mlp, mlp_dropout) in zip(self.att_layers, self.mlp_layers):
            holder.append(x)
            x1, y = attn_mh(attn_norm(x), y)
            x1 = attn_dropout(x1)
            x = x1 + x

            x = mlp_dropout(mlp(mlp_norm(x))) + x

        holder.append(x)


        return holder , y

#%% Classificaiton layers



class Classifier(nn.Module):

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
            
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

#%% 
class ConViT(nn.Module):
    # the proposed model

    def __init__(self, image_dim, num_dense_layers, growth_rate, denseblk_dropout_rate ,
                 patch_size, emb_dim,
                 num_encod_layers, num_heads, MLP_dim, att_drop_rate, dropkey_rate,
                 classifier_drop_rate, num_classes):
        super().__init__()

        # Params:
        
        # image_dim: total number of image bands  
        # num_dense_layers: total number of layers inside the convolutional block
        # growth_rate: growth rade of the convolutional block
        # denseblk_dropout_rate:  drop rate used inside the convolutional block
        # patch_size: size of input patches. For example, for (200,9,9) the patch_size is 9
        
        # num_encod_layers: number of encoder layers in the Convolutional transformer block (CTB)
        # num_heads: numbers of total head in CTB (it have to be devisible by emb_dim)
        # MLP_dim: MLP dimension
        # att_drop_rate: Attention drop rate in CTB
        # dropkey_rate: Drop key rate in CTB
        # classifier_drop_rate: Drop rate in classifer
        
        self.image_dim = image_dim
        

        self.nb_dense_layers = num_dense_layers
        self.growth_rate = growth_rate
        self.dense_dropout_rate = denseblk_dropout_rate

        
        self.patch_size = patch_size
        self.emb_dim = emb_dim

        self.num_encod_layers = num_encod_layers
        self.num_heads = num_heads
        self.head_dim = emb_dim/num_heads
        self.MLP_dim = MLP_dim
        self.att_drop_rate = att_drop_rate
        self.dropkey_rate = dropkey_rate

        self.num_classes = num_classes
        self.classifier_drop_rate = classifier_drop_rate

        # Conv Block

        self.dens_blk = DenseBlock(nb_input_channels= self.image_dim, nb_layers= self.nb_dense_layers, growth_rate= self.growth_rate, dropout_rate= self.dense_dropout_rate)
        self.image_dim += self.growth_rate*self.nb_dense_layers


        # # Embeddings
        self.patch_embeddings = PatchEmbeddings(image_dim= self.image_dim, emb_dim= self.emb_dim)
        self.pos_embeddings = PositionalEmbeddings(num_seq= self.patch_size**2, emb_dim= self.emb_dim)
        self.local_embeddings =LocalEmbeddings(image_dim= self.image_dim, patch_size=self.patch_size, emb_dim= self.emb_dim)




        # # Encoder
        self.encoders = CTB(emb_dim=self.emb_dim, num_layers=self.num_encod_layers, num_heads=self.num_heads,
                                      head_dim=self.head_dim,
                                      image_dim=self.image_dim, MLP_dim=self.MLP_dim, dropkey_rate= self.dropkey_rate, att_drop_rate=self.att_drop_rate)

        # # Feature fusion
        self.patchify = Rearrange(
            "b c (h p1) (w p2) -> b (h w) c p1 p2",
            p1=1, p2=1)
        self.flatten = nn.Flatten(start_dim=2)

  
        # # Linear Classifier with Pooling, No ClsToken
        self.dropout = nn.Dropout(self.classifier_drop_rate)
        self.classifier = Classifier(dim= emb_dim*(self.num_encod_layers+1) +emb_dim + self.image_dim , num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = [batch, image_dim, height, width]
        b, c, h, w = x.shape

        # Conv Block
        # x = [batch, new_imag_dim=image_dim+nb_layers*growth_rate, height, width]
        x0 = self.dens_blk(x)
        
        # Embeddingd
        # x1= [batch, num_seq= num_patches, emb_dim]
        x1 = self.patch_embeddings(x0)
        x2 = self.pos_embeddings(x0)
        x3 = self.local_embeddings(x0)
        x1 = x1 + x2 + x3

        x1 = self.dropout(x1)
        

        # # 
        # x1= [batch, num_seq, emb_dim]
        # x2= [batch, emb_dim, height, width]
        x1, x2 = self.encoders(x1, x0)

        x1 = torch.concat(x1, dim=2)

        # # Fusion
        x0 = self.flatten(self.patchify(x0)) #  [batch, num_seq, new_imag_dim]
        x2 = self.flatten(self.patchify(x2)) #  [batch, num_seq, emb_dim]
        
    
        x1 = torch.cat((x1, x2, x0), dim=2) #  [batch, num_seq, 2*emb_dim + new_imag_dim]

     
        # # Linear Classifier with Pooling, No ClsToken
        
        x1 = self.dropout(x1).mean(dim=1) #  [batch, 1 , 2*emb_dim + new_imag_dim]
        # x1 = x1.mean(dim=1) #  [batch, 1 , 2*emb_dim + new_imag_dim]
        x1 = self.classifier(x1) #  [batch, num_classes]
        return x1