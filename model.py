import os
import math
import random
import time
import inspect
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


def loss_func(data, preds):
    return torch.sqrt(((data['x'] - preds[:, 0])**2).mean()) + torch.sqrt(
        ((data['y'] - preds[:, 1])**2).mean())


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SensorCNN1D(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.cnn = torch.nn.ModuleDict({
            'conv1': nn.Conv1d(config.n_inputs, config.n_out_cnn1, kernel_size=1),
            'lrelu': nn.LeakyReLU(),
            'conv2': nn.Conv1d(config.n_out_cnn1, config.n_out_cnn2, kernel_size=3, stride=2)
        })

        self.ln1 = nn.LayerNorm(config.n_out_cnn2)
        self.linear1 = nn.Linear(config.n_out_cnn2, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.linear_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.cnn(x['sensor'])
        x = x + self.linear_proj((self.gelu(self.linear1(self.ln1(x)))))
        return x


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=False)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, x_sensor):
        x = x + self.attn(self.ln_1(x))
        x = x + \
            F.scaled_dot_product_attention(
                self.ln_2(x), x_sensor, x_sensor, is_causal=False)
        x = x + self.mlp(self.ln_3(x))
        return x


class Anchor2Token(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.bssid_embedding = nn.Embedding(config.num_wifi, config.n_embd)
        self.pos_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))

    def forward(self, x):
        B, T = x['rssi'].size()
        pos = torch.arange(0, T + 1, dtype=torch.long,
                           device=x.device)  # cls token
        pos_emb = self.pos_embedding(pos)
        bssid_emb = self.bssid_embedding(x['bssid'])
        numeric_feats = x['rssi']
        feats = bssid_emb + numeric_feats
        feats = torch.cat((self.cls_token.expand(
            feats.shape[0], -1, -1), feats), 1)
        out = feats + pos_emb
        return out


class IndoorTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            sensorenc=SensorCNN1D(config),
            wifienc=Anchor2Token(config),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.fc_xy = nn.Linear(config.n_embd, 2, bias=False)
        self.fc_floor = nn.Linear(config.n_embd, 1)

        # init params
        self.apply(self._init_weights)

    def forward(self, x, targets):
        sensor_feats = self.transformer.sensorenc(x)
        wifi_feats = self.transformer.wifienc(x)
        for block in self.transformer.h:
            x = block(sensor_feats, wifi_feats)
        x = self.transformer.ln_f(x)
        xy = self.fc_xy(x[:, 0])
        floor = self.fc_floor(x[:, 0])
        loss = loss_func(targets, x)
        return xy, floor, loss


class TrainingLoop():
    pass
