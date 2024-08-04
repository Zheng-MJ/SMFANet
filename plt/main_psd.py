import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
from feature_show import feature_show

class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim,hidden_dim,3,1,1,groups=dim),
            nn.Conv2d(hidden_dim,hidden_dim,1,1,0)
        )
        self.act =nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        return x

class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim,hidden_dim,1,1,0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim ,3,1,1)

        self.act =nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        if self.training:
            x = self.act(self.conv_0(x))
            x1, x2 = torch.split(x,[self.p_dim,self.hidden_dim-self.p_dim],dim=1)
            x1 = self.act(self.conv_1(x1))
            x = self.conv_2(torch.cat([x1,x2], dim=1))
        else:
            x = self.act(self.conv_0(x))
            x[:,:self.p_dim,:,:] = self.act(self.conv_1(x[:,:self.p_dim,:,:]))
            x = self.conv_2(x)
        return x

class SMFA(nn.Module):
    def __init__(self, dim=36, id = 0):
        super(SMFA, self).__init__()

        self.id = id

        self.linear_0 = nn.Conv2d(dim,dim*2,1,1,0)
        self.linear_1 = nn.Conv2d(dim,dim,1,1,0)
        self.linear_2 = nn.Conv2d(dim,dim,1,1,0)

        self.lde = DMlp(dim,2)

        self.dw_conv = nn.Conv2d(dim,dim,3,1,1,groups=dim)

        self.gelu = nn.GELU()
        self.down_scale = 8

        self.alpha = nn.Parameter(torch.ones((1,dim,1,1)))
        self.belt = nn.Parameter(torch.zeros((1,dim,1,1)))

    def forward(self, f):
        
        feature_show(f, f"{self.id}_input")

        _,_,h,w = f.shape

        y, x = self.linear_0(f).chunk(2, dim=1)

        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2,-1), keepdim=True)
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h,w), mode='nearest')

        y_d = self.lde(y)

        out = self.linear_2(x_l + y_d)

        feature_show(x_l, f"{self.id}_easa")
        feature_show(y_d, f"{self.id}_lde")
        feature_show(out, f"{self.id}_output")

        return out

class FMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0, id = 0):
        super().__init__()

        self.smfa = SMFA(dim, id)
        self.pcfn = PCFN(dim, ffn_scale)

    def forward(self, x):
        x = self.smfa(F.normalize(x)) + x
        x = self.pcfn(F.normalize(x)) + x
        return x
 
class SMFANet(nn.Module):
    def __init__(self, dim=36, n_blocks=8, ffn_scale=2, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.feats = nn.Sequential(*[FMB(dim, ffn_scale, id = i) for i in range(n_blocks)])
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )
    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x

def img_read(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)
    
def main():
    model = SMFANet(dim=48, n_blocks=12, upscaling_factor= 4).eval().cuda()
    model_path = os.path.join('pretrain', 'SMFANet_plus_DF2K_100w_x4SR.pth')
    model.load_state_dict(torch.load(model_path)['params'], strict=True)

    with torch.no_grad():
        img = img_read(os.path.join('plt', '0862.png')).cuda()
        model(img)

if __name__ == "__main__":
    main()




