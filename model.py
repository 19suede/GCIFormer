import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class Patch_Embedding(nn.Module):
    def __init__(self, patch_size, num_ch, model_dim):
        super(Patch_Embedding, self).__init__()
        self.patch_depth = patch_size * patch_size * num_ch
        self.map = nn.Linear(self.patch_depth, model_dim)

    def forward(self, patch):
        patch_embed = self.map(patch)  # [bs, num_patches, model_dim]
        return patch_embed

class Position_Encoding(nn.Module):
    def __init__(self, model_dim, max_position_len=10000):
        super(Position_Encoding, self).__init__()

        self.max_position_len = max_position_len
        self.model_dim = model_dim

        pos_mat = torch.arange(max_position_len).reshape((-1, 1))  # [max_position_len, 1]
        i_mat = torch.pow(10000, torch.arange(0, model_dim, 2).reshape((1, -1)) / model_dim)  # [1, model_dim/2]
        pe_embedding_table = torch.zeros(max_position_len, model_dim)  # [max_position_len, model_dim]
        pe_embedding_table[:, 0::2] = torch.sin(pos_mat / i_mat)
        pe_embedding_table[:, 1::2] = torch.cos(pos_mat / i_mat)

        self.pe_embedding = nn.Embedding(max_position_len, model_dim)
        self.pe_embedding.weight = nn.Parameter(pe_embedding_table, requires_grad=False)

    def forward(self, input):
        seq_len = input.size(1)
        position_indices = torch.arange(seq_len).unsqueeze(0).repeat(input.size(0), 1).to(input.device)
        position_indices = position_indices.long()

        input_pe_encoding = self.pe_embedding(position_indices)
        output = input + input_pe_encoding

        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.Q = nn.Linear(model_dim, model_dim)
        self.K = nn.Linear(model_dim, model_dim)
        self.V = nn.Linear(model_dim, model_dim)

        self.final_linear_layer = nn.Linear(model_dim, model_dim)

    def forward(self, q, k, v):
        bs, seq_len, model_dim = q.shape
        head_dim = model_dim // self.num_heads

        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)

        q = q.reshape(bs, seq_len, self.num_heads, head_dim).transpose(1, 2)
        q = q.reshape(bs * self.num_heads, seq_len, head_dim)

        k = k.reshape(bs, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.reshape(bs * self.num_heads, seq_len, head_dim)

        v = v.reshape(bs, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(bs * self.num_heads, seq_len, head_dim)

        attn_prob = F.softmax(torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(head_dim), dim=-1)

        output = torch.bmm(attn_prob, v)
        output = output.reshape(bs, self.num_heads, seq_len, head_dim).transpose(1, 2)
        output = output.reshape(bs, seq_len, model_dim)

        output = self.final_linear_layer(output)

        return output

class FeedForwardNet(nn.Module):
    def __init__(self, model_dim):
        super(FeedForwardNet, self).__init__()

        self.fc1 = nn.Linear(model_dim, model_dim * 4)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(model_dim * 4, model_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch, scale):
        super(DownSample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=scale, stride=scale),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch, scale):
        super(UpSample, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, scale, scale, 0),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DWConv(nn.Module):
    def __init__(self, in_ch, k=3, dilation=1):
        super(DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=k, padding=(math.floor(dilation*(k-1)/2), math.floor(dilation*(k-1)/2)), dilation=dilation, groups=in_ch, padding_mode="reflect"),
            nn.BatchNorm2d(in_ch),
        )

    def forward(self, x):
        return self.conv(x)

class FEB(nn.Module):
    def __init__(self, in_ch):
        super(FEB, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, 1),
            DWConv(in_ch, 3),
            nn.GELU(),
            DWConv(in_ch, 3),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, 1, 1),
        )

    def forward(self, x):
        out = self.conv(x) + x

        return out

class ResConv(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super(ResConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, 1, 1),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, in_ch, 3, 1, 1),
        )

    def forward(self, x):
        out = self.conv(x) + x
        return out

class Encoder(nn.Module):
    def __init__(self, num_convs, in_ch):
        super(Encoder, self).__init__()
        self.num_convs = num_convs
        self.feb = ResConv(in_ch, in_ch)
        self.down = DownSample(in_ch, in_ch, 2**num_convs)
        self.proj_conv = nn.Conv2d(in_ch, in_ch, 1, 1)

    def forward(self, x):
        x = self.feb(x)
        x = self.down(x)
        x = self.proj_conv(x)

        return x

class GLEncoder(nn.Module):
    def __init__(self, patch_size, window_size, model_dim, num_ch):
        super(GLEncoder, self).__init__()
        self.patch_size = patch_size
        self.window_size = window_size
        self.num_ch = num_ch
        self.model_dim = model_dim
        self.num_patches_in_window = window_size * window_size

        self.patch_embed = Patch_Embedding(self.patch_size, self.num_ch, self.model_dim)
        self.pos_enc = Position_Encoding(self.model_dim)
        self.proj = nn.Linear(self.model_dim, self.patch_size * self.patch_size * self.num_ch)

        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.mhsa = MultiHeadSelfAttention(model_dim)
        self.ffn = FeedForwardNet(model_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.g_layer_norm1 = nn.LayerNorm(model_dim)
        self.g_layer_norm2 = nn.LayerNorm(model_dim)
        self.g_mhsa = MultiHeadSelfAttention(model_dim)
        self.g_ffn = FeedForwardNet(model_dim)
        nn.init.normal_(self.cls_token, std=0.02)
        self.fc = nn.Sequential(
            nn.Linear(model_dim*2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, x, encoder, decoder):
        B, C, H, W = x.shape
        self.num_patches = (H // self.patch_size) * (W // self.patch_size)
        self.num_windows = (H // (self.patch_size * self.window_size)) * (W // (self.patch_size * self.window_size))

        win_img = encoder(x)
        win_patch = self.img2patch(win_img, self.patch_size)
        window_embed = self.patch_embed(win_patch)
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        window_embed = torch.cat((cls_tokens, window_embed), dim=1)
        window_embed = self.pos_enc(window_embed)
        window_embed_ = self.g_layer_norm1(window_embed)
        window_embed = self.g_mhsa(window_embed_, window_embed_, window_embed_) + window_embed
        window_embed = self.g_ffn(self.g_layer_norm2(window_embed)) + window_embed

        cls_tokens = window_embed[:, 0]
        cls_tokens = cls_tokens.unsqueeze(1).repeat(1, self.num_patches, 1)

        patch = self.img2patch(x, self.patch_size)
        _, N, _ = patch.shape
        patch_embed = self.patch_embed(patch)
        patch_embed = torch.concat([patch_embed, cls_tokens], dim=2)
        patch_embed = self.fc(patch_embed)

        wmhsa_out = self.window_multi_head_self_attention(patch_embed, self.pos_enc, self.mhsa, self.layer_norm1)
        wmhsa_out = wmhsa_out.reshape(B, self.num_patches, self.model_dim)
        wmhsa_out = self.ffn(self.layer_norm2(wmhsa_out)) + wmhsa_out

        patch_embed = wmhsa_out.reshape(B, self.num_patches, self.model_dim)
        img = self.proj(patch_embed)
        img = torch.transpose(img, -1, -2)
        img = self.patch2img(img, H, W, self.patch_size)

        return img

    def window_multi_head_self_attention(self, patch_embedding, pos_enc, mhsa, ln, global_seq=None):
        window = self.patch2window(patch_embedding)
        bs, num_windows, num_patches_in_window, patch_embedding_dim = window.shape
        window = window.reshape(bs * num_windows, num_patches_in_window, patch_embedding_dim)

        if global_seq != None:
            window = torch.concat((window, global_seq), dim=1)
        window = pos_enc(window)
        window_ = ln(window)
        wmhsa_out = mhsa(window_, window_, window_) + window

        return wmhsa_out

    def patch2window(self, patch_embedding):
        bs, num_patches, patch_embedding_dim = patch_embedding.shape
        image_height = image_width = int(math.sqrt(num_patches))
        window = rearrange(patch_embedding, 'b (h w) ed -> b ed h w', h=image_height, w=image_width)
        window = rearrange(window, 'b ed (h1 p1) (w1 p2) -> b (h1 w1) (p1 p2) ed', p1=self.window_size, p2=self.window_size) # (bs, num_windows, num_patches_in_window, patch_embedding_dim)

        return window

    def img2patch(self, image, patch_size):
        patch = rearrange(image, 'b c (h1 p1) (w1 p2) -> b (h1 w1) (p1 p2 c)', p1=patch_size, p2=patch_size) # (bs, num_patches, patch_depth)
        return patch

    def patch2img(self, patch, img_h, img_w, patch_size):
        image = F.fold(patch, output_size=(img_h, img_w), kernel_size=(patch_size, patch_size), stride=patch_size)
        return image

class Conv1x1(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super(Conv1x1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, out_ch, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, out_ch, 1, 1),
        )
    def forward(self, x):
        return self.conv1(x)

class CrossFuse(nn.Module):
    def __init__(self, hidden_ch):
        super(CrossFuse, self).__init__()
        self.pool_h1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h2 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w2 = nn.AdaptiveAvgPool2d((1, None))

        self.conv_poolc = nn.Sequential(
            Conv1x1(hidden_ch, hidden_ch, hidden_ch),
        )
        self.conv_c = nn.Sequential(
            nn.Conv2d(hidden_ch*2, hidden_ch, 1, 1),
            nn.BatchNorm2d(hidden_ch),
            FEB(hidden_ch),
            FEB(hidden_ch),
        )
        self.conv_1 = nn.Sequential(
            nn.BatchNorm2d(hidden_ch),
            FEB(hidden_ch),
            FEB(hidden_ch),
        )
        self.conv_2 = nn.Sequential(
            nn.BatchNorm2d(hidden_ch),
            FEB(hidden_ch),
            FEB(hidden_ch),
        )
        self.conv_3 = nn.Sequential(
            nn.BatchNorm2d(hidden_ch),
            FEB(hidden_ch),
            FEB(hidden_ch),
        )

    def forward(self, x, y):
        b, c, h, w = x.shape
        x_h = self.pool_h1(x)
        x_w = self.pool_w1(x)
        y_h = self.pool_h2(y)
        y_w = self.pool_w2(y)

        pool_c = self.conv_poolc(torch.concat([x_h, y_h, x_w.permute(0, 1, 3, 2), y_w.permute(0, 1, 3, 2)], dim=2))
        x_h, y_h, x_w, y_w = torch.split(pool_c, [h, h, w, w], dim=2)

        xy = self.conv_c(torch.concat([x, y], dim=1))
        x_hw = self.conv_1(torch.matmul(x_h, x_w.permute(0, 1, 3, 2)))*xy
        y_hw = self.conv_2(torch.matmul(y_h, y_w.permute(0, 1, 3, 2)))*xy
        out = self.conv_3(x_hw+y_hw) + xy

        return out

class MainBlock(nn.Module):
    def __init__(self, patch_size, window_size, model_dim, hidden_ch, num_blocks):
        super(MainBlock, self).__init__()
        self.num_blocks = num_blocks
        num_convs = int(math.log(window_size, 2))
        self.spa_encoder = nn.ModuleList([
            GLEncoder(patch_size, window_size, model_dim, hidden_ch) for i in range(num_blocks)
        ])
        self.spe_encoder = nn.ModuleList([
            GLEncoder(patch_size, window_size, model_dim, hidden_ch) for i in range(num_blocks)
        ])
        self.mix_block = nn.ModuleList([
            CrossFuse(hidden_ch) for i in range(num_blocks)
        ])

        self.encoder1 = nn.ModuleList([
            Encoder(num_convs, hidden_ch) for i in range(num_blocks)
        ])
        self.encoder2 = nn.ModuleList([
            Encoder(num_convs, hidden_ch) for i in range(num_blocks)
        ])

    def forward(self, spa, spe):
        spa_all = spa
        spe_all = spe
        mix_outs = []

        for i in range(self.num_blocks):
            spa_ = self.spa_encoder[i](spa_all, self.encoder1[i], None)
            spa_all = spa_all + spa_
            spe_ = self.spe_encoder[i](spe_all, self.encoder2[i], None)
            spe_all = spe_all + spe_
            spe_all = self.mix_block[i](spa_all, spe_all)
            mix_outs.append(spe_all)

        mix_outs = torch.concat(mix_outs, dim=1)
        return mix_outs, spa_all

class Base1(nn.Module):
    def __init__(self, config):
        super(Base1, self).__init__()
        self.patch_size = config.patch_size
        self.window_size = config.window_size
        self.model_dim = config.model_dim
        self.hidden_ch = config.hidden_ch
        self.num_blocks = config.num_blocks

        self.laplacian_kernel = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,
                                          stride=1, padding=1,
                                          groups=1, bias=False)
        laplacian_weights = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        self.laplacian_kernel.weight.data = laplacian_weights
        self.laplacian_kernel.weight.requires_grad = False

        self.spa_conv = nn.Conv2d(in_channels=2+config.lms_ch, out_channels=self.hidden_ch, kernel_size=3, stride=1, padding="same")
        self.spe_conv = nn.Conv2d(in_channels=config.lms_ch, out_channels=self.hidden_ch, kernel_size=3, stride=1, padding="same")

        self.main_block = MainBlock(self.patch_size, self.window_size, self.model_dim, self.hidden_ch, self.num_blocks)

        self.res_conv = nn.Sequential(
            nn.Conv2d(self.hidden_ch*self.num_blocks, self.hidden_ch, 3, 1, 1),
            ResConv(self.hidden_ch, self.hidden_ch),
            ResConv(self.hidden_ch, self.hidden_ch),
            ResConv(self.hidden_ch, self.hidden_ch),
            ResConv(self.hidden_ch, self.hidden_ch),
            ResConv(self.hidden_ch, self.hidden_ch),
            ResConv(self.hidden_ch, self.hidden_ch),
            nn.Conv2d(self.hidden_ch, config.lms_ch, 3, 1, 1)
        )
        self.spa_rec = nn.Sequential(
            ResConv(self.hidden_ch, self.hidden_ch),
            ResConv(self.hidden_ch, self.hidden_ch),
            ResConv(self.hidden_ch, self.hidden_ch),
            ResConv(self.hidden_ch, self.hidden_ch),
            nn.Conv2d(self.hidden_ch, 1, 3, 1, 1),
        )

    def forward(self, pan, lms, ms):
        pan_hp = self.get_edge(pan)
        spa_ = torch.concat([pan, pan_hp], dim=1)

        spa = self.spa_conv(torch.concat([spa_, lms], dim=1))
        spe = self.spe_conv(lms)

        out, spa_rec = self.main_block(spa, spe)

        out = self.res_conv(out) + lms
        spa_rec = self.spa_rec(spa_rec)

        return out, spa_rec

    def get_edge(self, data):
        rs = self.laplacian_kernel(data)
        return rs
