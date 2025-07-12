import torch
import torch.nn as nn

class Criterion(nn.Module):
    def __init__(self, ch, dr, beta=0.5):
        super(Criterion, self).__init__()
        self.dr = dr
        self.beta = beta
        self.l1_1 = nn.L1Loss()
        self.l1_2 = nn.L1Loss()
        self.l1_3 = nn.L1Loss()

        self.laplacian_kernel = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1,
                                          bias=False)
        # 设置卷积核的权重
        laplacian_weights = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        self.laplacian_kernel.weight.data = laplacian_weights
        self.laplacian_kernel.weight.requires_grad = False

    def forward(self, fused, gt, spa_rec, pan):
        self.laplacian_kernel = self.laplacian_kernel.to(fused.device)
        loss1 = self.l1_1(fused, gt)

        pan_hp = self.laplacian_kernel(pan)
        rec_hp = self.laplacian_kernel(spa_rec)
        loss2 = self.l1_2(pan_hp, rec_hp)
        loss3 = self.l1_3(pan, spa_rec)

        loss = loss1 + self.beta*(loss2 + loss3)

        return loss
