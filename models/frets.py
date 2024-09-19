import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# class FreTs(nn.Module):
#     def __init__(self):
#         super(FreTs, self).__init__()
#         self.embed_size = 128 #embed_size
#         self.hidden_size = 256 #hidden_size
#         self.pre_length = 11
#         self.feature_size = 2
#         self.seq_length = 128
#         self.channel_independence = 0
#         self.sparsity_threshold = 0.01
#         self.scale = 0.02
#         self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
#         self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
#         self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
#         self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
#         self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
#         self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
#         self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
#         self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
#         self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
#         self.flatten = nn.Flatten()
#         self.fc = nn.Sequential(
#             nn.Linear(self.seq_length * self.embed_size, self.hidden_size),
#             nn.LeakyReLU(),
#             nn.Linear(self.hidden_size, self.pre_length)
#         )
#         self.fc1 = nn.Linear(22,11)
#
#     # dimension extension
#     def tokenEmb(self, x):
#         # x: [Batch, Input length, Channel]
#         x = x.permute(0, 2, 1)
#         x = x.unsqueeze(3)
#         # N*T*1 x 1*D = N*T*D
#         y = self.embeddings
#         return x * y
#
#     # frequency temporal learner
#     def MLP_temporal(self, x, B, N, L):
#         # [B, N, T, D]
#         x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension
#         y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
#         x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
#         return x
#
#     # frequency channel learner
#     def MLP_channel(self, x, B, N, L):
#         # [B, N, T, D]
#         x = x.permute(0, 2, 1, 3)
#         # [B, T, N, D]
#         x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on N dimension
#         y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
#         x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
#         x = x.permute(0, 2, 1, 3)
#         # [B, N, T, D]
#         return x
#
#     # frequency-domain MLPs
#     # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
#     # rb: the real part of bias, ib: the imaginary part of bias
#     def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
#         o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
#                               device=x.device)
#         o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
#                               device=x.device)
#
#         o1_real = F.relu(
#             torch.einsum('bijd,dd->bijd', x.real, r) - \
#             torch.einsum('bijd,dd->bijd', x.imag, i) + \
#             rb
#         )
#
#         o1_imag = F.relu(
#             torch.einsum('bijd,dd->bijd', x.imag, r) + \
#             torch.einsum('bijd,dd->bijd', x.real, i) + \
#             ib
#         )
#
#         y = torch.stack([o1_real, o1_imag], dim=-1)
#         y = F.softshrink(y, lambd=self.sparsity_threshold)
#         y = torch.view_as_complex(y)
#         return y
#
#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#         x = x.transpose(1,2)
# #         print(x.shape)
#         B, T, N = x.shape
#         # embedding x: [B, N, T, D]
#         x = self.tokenEmb(x)
#         bias = x
#         # [B, N, T, D]
#         if self.channel_independence == '1':
#             x = self.MLP_channel(x, B, N, T)
#         # [B, N, T, D]
#         x = self.MLP_temporal(x, B, N, T)
#         x = x + bias
#         x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
#         x = self.flatten(x)
#         x = self.fc1(x)
#         d =[]
#         return x,d
# class FreTs(nn.Module):
#     def __init__(self):
#         super(FreTs, self).__init__()
#
#
#         # LSTM units
#         self.lstm1 = nn.LSTM(input_size=2, hidden_size=128, batch_first=True,num_layers = 2)
#         self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
#         self.trans = nn.TransformerEncoderLayer(128,1)
#         # Fully connected layer
#         self.fc = nn.Linear(128, 11)
#
#
#
#     def forward(self, x):
#         # LSTM layers
#         x = x.transpose(1,2)
# #         print(x.shape)
# #         x = self.trans(x)
# #         x = x.transpose(1,2)
# #         print(x.shape)
#         x, _ = self.lstm1(x)
# #         print(x.shape)
# #         x, _ = self.lstm2(x)
# #         print(x.shape)
# #         print(x)
#         x = self.fc(x[:, :, -1])  # Only use the last timestep output of the second LSTM
# #         print(x.shape)
# #         print(asd)
#         d = []
#         return x,d
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
from torchmetrics.classification import MulticlassF1Score

# from dataloader import get_datasets
# from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D


class ICB(L.LightningModule):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1)) # * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if True:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


class TSLANet_layer(L.LightningModule):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # Check if both ASB and ICB are true
        # if args.ICB and args.ASB:
        x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # # If only ICB is true
        # elif args.ICB:
        #     x = x + self.drop_path(self.icb(self.norm2(x)))
        # # If only ASB is true
        # elif args.ASB:
        #     x = x + self.drop_path(self.asb(self.norm1(x)))
        # # If neither is true, just pass x through
        return x


class FreTs(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(
            seq_len=128, patch_size=8,
            in_chans=2, embed_dim=128
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, 128), requires_grad=True)
        self.pos_drop = nn.Dropout(p=0.15)

        self.input_layer = nn.Linear(8, 128)

        dpr = [x.item() for x in torch.linspace(0, 0.15, 2)]  # stochastic depth decay rule

        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=128, drop=0.15, drop_path=dpr[i])
            for i in range(2)]
        )

        # Classifier head
        self.head = nn.Linear(128, 11)

        # init weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pretrain(self, x_in):
        x = self.patch_embed(x_in)
        x = x + self.pos_embed
        x_patched = self.pos_drop(x)

        x_masked, _, self.mask, _ = random_masking_3D(x, mask_ratio=0.4)
        self.mask = self.mask.bool()  # mask: [bs x num_patch x n_vars]

        for tsla_blk in self.tsla_blocks:
            x_masked = tsla_blk(x_masked)

        return x_masked, x_patched

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)

        x = x.mean(1)
        x = self.head(x)
        d = []
        return x,d

