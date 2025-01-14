from torch import nn
import torch


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))

class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            # print(x.shape)
        
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
            # print(x.shape)
        return ftrs

import torchvision

class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])

    def forward(self, x, encoder_features):
        
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)

            # print("{}: {}".format(i, x.shape))
        
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, enc_chs=(13,64,128,256,512), dec_chs=(512, 256,128,64), num_class=1, retain_dim=False, out_sz=(572,572)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        #self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        #self.retain_dim  = retain_dim
        #output_size      = (112,112)
        #self.avgpool     = nn.AdaptiveAvgPool2d(output_size) 
        self.last = nn.ConvTranspose2d(64, 1, kernel_size=(1,1), stride=6, padding=2, output_padding=1)
    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        # out      = self.avgpool(out)
        out = self.last(out)
        #if self.retain_dim:
        #    out = F.interpolate(out, out_sz)
        return out

if __name__ == '__main__':
    unet = UNet(num_class = 1)
    x = torch.randn(64,13,112,112)
    print(unet(x).shape)
