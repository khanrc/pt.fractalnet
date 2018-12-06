""" Fractal Model """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """ Conv - Dropout - BN - ReLU """
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, dropout=None):
        super().__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False)
        if dropout is not None and dropout > 0.:
            self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        else:
            self.dropout = None
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x):
        out = self.conv(x)
        if self.dropout:
            out = self.dropout(out)
        out = self.bn(out)
        out = F.relu_(out)

        return out


class FractalBlock(nn.Module):
    def __init__(self, n_columns, C_in, C_out, p_local_drop, p_dropout, global_drop_ratio):
        """ Fractal block
        Args:
            - n_columns: # of columns
            - C_in: channel_in
            - C_out: channel_out
            - p_local_drop: local droppath prob
            - p_dropout: dropout prob
            - global_drop_ratio: global droppath ratio
        """
        super().__init__()

        self.n_columns = n_columns
        self.columns = nn.ModuleList([nn.ModuleList() for _ in range(n_columns)])
        self.max_depth = 2 ** (n_columns-1)
        self.p_local_drop = p_local_drop
        self.global_drop_ratio = global_drop_ratio

        depth = self.max_depth
        self.count = np.zeros([self.max_depth], dtype=np.int)
        for col in self.columns:
            for i in range(self.max_depth):
                if (i+1) % depth == 0:
                    c_in = C_in if i+1 == depth else C_out
                    module = ConvBlock(c_in, C_out, dropout=p_dropout)
                    self.count[i] += 1
                else:
                    module = None

                col.append(module)

            depth //= 2

    def local_drop_sampler(self, N):
        """ drop path probs sampler """
        drops = np.random.binomial(1, self.p_local_drop, size=[N]).astype(np.bool)
        if drops.all(): # all droped case
            i = np.random.randint(0, N)
            drops[i] = False

        return drops

    def join(self, outs):
        """ join with local drop path
        outs: [cols, N, C, H, W] (list)
        [!] make it better:
            - per-sample (not per-batch) local drop path
            - fully numpy or torch based (no for loop)
        """
        if len(outs) == 1:
            return outs[0]

        # apply local drop path only in training
        if self.training:
            # local drop path
            drops = self.local_drop_sampler(len(outs))
            outs = [o for drop, o in zip(drops, outs) if not drop]

        out = torch.stack(outs)
        return out.mean(dim=0)

    def forward_global(self, x, global_col):
        """ Global drop path """
        dist = 2 ** (self.n_columns-1 - global_col) # distance between module
        out = x
        for i in range(dist-1, self.max_depth, dist):
            out = self.columns[global_col][i](out)

        return out

    def forward_local(self, x):
        """ Local drop path """
        outs = [x for _ in range(self.n_columns)]
        for i in range(self.max_depth):
            st = self.n_columns - self.count[i]
            cur_out = [] # outs of current depth

            for c in range(st, self.n_columns):
                cur_out.append(self.columns[c][i](outs[c]))

            # join
            #print("join in depth = {}, # of in_join = {}".format(i, len(cur_out)))
            joined = self.join(cur_out)

            for c in range(st, self.n_columns):
                outs[c] = joined

        return outs[0]

    def forward(self, x, deepest=False):
        if self.training == False:
            # eval
            if deepest:
                deepest_col = self.n_columns-1
                return self.forward_global(x, deepest_col)
            else:
                return self.forward_local(x)
        else:
            # training
            if np.random.rand() < self.global_drop_ratio:
                global_col = np.random.randint(0, self.n_columns)
                return self.forward_global(x, global_col)
            else:
                return self.forward_local(x)


class FractalNet(nn.Module):
    def __init__(self, data_shape, n_columns, channels, p_local_drop, dropout_probs,
                 global_drop_ratio, gap=False):
        """
        Args:
            - data_shape: (C, H, W, n_classes). e.g. (3, 32, 32, 10) - CIFAR 10.
            - n_columns: the number of columns
            - channels: channel outs (list)
            - p_local_drop: local drop prob
            - dropout_probs: dropout probs (list)
            - global_drop_ratio: global droppath ratio
        """
        super().__init__()
        self.B = len(channels) # the number of blocks
        C_in, H, W, n_classes = data_shape
        assert len(channels) == len(dropout_probs)
        assert H == W
        size = H

        layers = []
        C_out = C_in # work like C_out of block0 == data channels.
        total_layers = 0
        for b, (C, p_dropout) in enumerate(zip(channels, dropout_probs)):
            C_in, C_out = C_out, C
            #print("Channel in = {}, Channel out = {}".format(C_in, C_out))
            fb = FractalBlock(n_columns, C_in, C_out, p_local_drop, p_dropout, global_drop_ratio)
            layers.append(fb)
            if gap == False or b < self.B-1:
                # Originally, every pool is max-pool in the paper (No GAP).
                layers.append(nn.MaxPool2d(2))
            else:
                # gap == True and last block
                layers.append(nn.AdaptiveAvgPool2d(1)) # GAP

            size //= 2
            total_layers += fb.max_depth

        print("Last featuremap size = {}".format(size))
        print("Total layers = {}".format(total_layers))

        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1] * size * size, n_classes)

        # xavier init as in the paper
        for n, p in self.named_parameters():
            if p.dim() > 1: # weights only
                nn.init.xavier_uniform_(p)
            else: # bn w/b or bias
                if 'bn.weight' in n:
                    nn.init.ones_(p)
                else:
                    nn.init.zeros_(p)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)
