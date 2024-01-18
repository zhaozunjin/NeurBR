import torch

from .common import *

class Conv_AP_BN_LR(nn.Module):
    def __init__(self, in_channels, out_channels, stride_conv, kernel_size_conv, padding_conv, kernel_size_pool, stride_pool):
        super(Conv_AP_BN_LR, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride_conv, kernel_size=kernel_size_conv, padding=padding_conv),
            nn.AvgPool2d(kernel_size=kernel_size_pool, stride=stride_pool, padding=0),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class Conv_BN_LR(nn.Module):
    def __init__(self, in_channels, out_channels, stride_conv, kernel_size_conv, padding_conv):
        super(Conv_BN_LR, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride_conv,
                      kernel_size=kernel_size_conv, padding=padding_conv),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class upsample_BN(nn.Module):
    def __init__(self, out_channels):
        super(upsample_BN, self).__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear"),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride_conv, kernel_size_conv, padding_conv, kernel_size_pool, stride_pool, use_gird=True):
        super(Encoder, self).__init__()
        self.use_gird = use_gird
        # block1
        self.block1 = nn.Sequential(
            Conv_AP_BN_LR(in_channels, out_channels=out_channels[0],
                          stride_conv=stride_conv,
                          kernel_size_conv=kernel_size_conv,
                          padding_conv=padding_conv,
                          kernel_size_pool=kernel_size_pool,
                          stride_pool=stride_pool),
            Conv_BN_LR(out_channels[0], out_channels[0], stride_conv, kernel_size_conv, padding_conv)
        )

        # block2
        self.block2 = nn.Sequential(
            Conv_AP_BN_LR(out_channels[0], out_channels=out_channels[1],
                          stride_conv=stride_conv,
                          kernel_size_conv=kernel_size_conv,
                          padding_conv=padding_conv,
                          kernel_size_pool=kernel_size_pool,
                          stride_pool=stride_pool),
            Conv_BN_LR(out_channels[1], out_channels[1], stride_conv, kernel_size_conv, padding_conv)
        )

        # block3
        self.block3 = nn.Sequential(
            Conv_AP_BN_LR(out_channels[1], out_channels=out_channels[2],
                          stride_conv=stride_conv,
                          kernel_size_conv=kernel_size_conv,
                          padding_conv=padding_conv,
                          kernel_size_pool=kernel_size_pool,
                          stride_pool=stride_pool),
            Conv_BN_LR(out_channels[2], out_channels[2], stride_conv, kernel_size_conv, padding_conv)
        )


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        if self.use_gird:
            # output bilateral_grid
            x = nn.Tanh()(x)
            # x = nn.Sigmoid()(x)
            x = torch.split(x, 3, 1) # set the number of each grid cell as 3, than the depth of bilateral grid is C/3
            x = torch.stack(x, 2)
        return x


class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        # Nx12x8x16x16
        device = bilateral_grid.get_device()
        N, H, W = guidemap.shape
        hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        if device >= 0:
            hg = hg.to(device)
            wg = wg.to(device)
        hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        guidemap = guidemap.unsqueeze(1) #Nx1xHxW
        guidemap = guidemap.permute(0, 2, 3, 1).contiguous()#NxHxWx1
        guidemap_guide = torch.cat([hg, wg, guidemap], dim=3).unsqueeze(1)  # Nx1xHxWx3
        coeff = nn.functional.grid_sample(bilateral_grid, guidemap_guide,
                                          mode='bilinear', padding_mode='border', align_corners=True)
        coeff = coeff.squeeze(2)
        return coeff



class BilateralDIP(nn.Module):
    def __init__(self, in_channels=3, out_channels_encoder=[8, 16, 24],
                 stride_conv=1, kernel_size_conv=11, padding_conv=5, kernel_size_pool=2, stride_pool=2, use_gird=False):
        super(BilateralDIP, self).__init__()
        self.encoder = Encoder(in_channels,out_channels_encoder, stride_conv, kernel_size_conv, padding_conv, kernel_size_pool, stride_pool, use_gird=use_gird)
        self.slice = Slice()


    def forward(self, inputs, fullres_img):
        bilateral_grid = self.encoder(inputs)
        output = self.slice(bilateral_grid, fullres_img)
        return bilateral_grid, output

