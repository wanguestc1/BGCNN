import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np




class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)




    def forward(self, x):

        B, N, C = x.shape


        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)


        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                  act_layer=GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class CNNEmbed(nn.Module):


    def __init__(self, in_channels=3, out_channels=32, kernel_size=3,stride=1,flagpool=False):
        super().__init__()
        self.flagpool=flagpool
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.pool=nn.AvgPool2d(2)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        B, C, H, W = x.shape
        x =  self.conv(x)
        x = self.norm (x)
        x =  self.act(x)
        if self.flagpool:
            x = self.pool(x)


        return x


class CNNTransformer(nn.Module):
    def __init__(self, num_classes=8, channel=[32, 32, 32, 32],
                 num_heads=[4, 4, 4, 4], mlp_ratios=[1, 1, 1, 1], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes


        self.Embed1 = CNNEmbed(in_channels=3,out_channels=channel[0],kernel_size=3,stride=2, flagpool =False)

        self.Embed2 = CNNEmbed(in_channels=channel[0], out_channels=channel[1], kernel_size=3, stride=2,flagpool =True)


        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 5*5+1, channel[1]))

        self.pos_embed2 = nn.Parameter(torch.zeros(1, 2*6+1, channel[2]))

        self.norm = norm_layer(channel[2])


        self.Tblock1= TransformBlock(
            dim=channel[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)


        self.Tblock2 = TransformBlock(
            dim=channel[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)


        self.Tblockcomb = TransformBlock(
            dim=channel[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)

        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2,padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.Avgpool = nn.AvgPool1d(2)
        self.pool1 = nn.AdaptiveMaxPool1d(1)
        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, channel[3]))

        self.cls_token_comb = nn.Parameter(torch.zeros(1, 1, channel[3]))
        self.pos_embed_comb = nn.Parameter(torch.zeros(1, 39 + 1, channel[1]))
        self.avp = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(channel[2], num_classes)
        self.headdis = nn.Linear(channel[2], num_classes)





    def forward(self, x):
        B = x.shape[0]

        # stage cnn
        x = self.Embed1(x)
        x = self.Embed2(x)

        # stage transformer

        x =x.flatten(2)
        x = x .transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens,x), dim=1)

        x = x + self.pos_embed1

        x = self.Tblock1(x)
        x1 = x


        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.transpose(1, 2)

        x = x + self.pos_embed2

        x = self.Tblock2(x)

        x2 = x

        x_comb = torch.cat((x1,x2), dim=1)



        cls_token_comb = self.cls_token_comb.expand(B, -1, -1)
        x_comb = torch.cat((cls_token_comb,x_comb), dim=1)

        x_comb = x_comb + self.pos_embed_comb

        x_comb = self.Tblockcomb(x_comb)

        cls = x_comb[:, 0]



        x_cls = self.head(cls)


        return x_cls







def BGCNN_MT_net( **kwargs):
    model = CNNTransformer(
        channel=[32, 32, 32, 32], num_heads=[4, 4, 4, 4], mlp_ratios=[1, 1, 1, 1], qkv_bias=True,
        norm_layer= partial(nn.LayerNorm, eps=1e-6),
        **kwargs)

    return model

#

# #
