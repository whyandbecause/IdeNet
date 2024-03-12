import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import *
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_unknown_tensor_from_pred
from einops import rearrange
import numbers

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

#####------------------------------------
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class SAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(SAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(nn.ReLU(inplace=True))
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.SA = SpatialAttention(7)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        att = self.SA(res)
        res = res * att
        return res + x

##---------- Resizing Modules ----------
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
                                  )

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels + s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

#--------------------------start  FE-------------------------------------------------
class IAM(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super(IAM, self).__init__()
        #out_channels = 128
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = [1,1,1], [1,2,3], [1,3,7]
        modules.append(IAMConv(in_channels, in_channels, rate1, 3, [1, 1, 1]))
        modules.append(IAMConv(in_channels, in_channels, rate2, 3, [1, 2, 3])) #ke = k + (k ? 1)(r ? 1)  p = (ke -1)//2
        modules.append(IAMConv(in_channels, in_channels, rate3, 5, [2, 6, 14]))
        modules.append(IAMPooling(in_channels, in_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        res, full = [], []
        for conv in self.convs:
            f = conv(x)
            res.append(x-f)
            full.append(f)
        full = torch.cat(full, dim=1)
        res = torch.cat(res, dim=1)
        return self.project(full)+self.project(res)

class IAMConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, kernel_size, padding):
        modules = [
            nn.Conv2d(in_channels, in_channels//2, kernel_size, padding=padding[0], dilation=dilation[0], bias=False),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels//2, kernel_size, padding=padding[1], dilation=dilation[1], bias=False),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, in_channels, kernel_size, padding=padding[2], dilation=dilation[2], bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        ]
        super(IAMConv, self).__init__(*modules)

class IAMPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(IAMPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
#------------------------------------------end FE-----------------------------------------------------------
#-----------------------------------------start fs----------------------------------------------------------
class IFM(nn.Module):
    def __init__(self, groups=32, channels=128, c_scale=2):
        super(IFM, self).__init__()
        self.up_c = nn.Conv2d(channels, channels*c_scale, kernel_size=1, stride=1, padding=0)
        self.groups = groups
        self.out_max, self.out_mean = [], []
        self.conv = nn.Conv2d(groups*2, channels, kernel_size=3, stride=1, padding=1)
        
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

    def forward(self, x):
        ori = x
        x = self.up_c(x)
        self.out_max, self.out_mean = [], []
        x = self.channel_shuffle(x, self.groups)
        x_groups = x.chunk(self.groups, 1)
        for x_i in x_groups:
            self.out_max.append(torch.max(x_i, dim=1)[0].unsqueeze(1))
            self.out_mean.append(torch.mean(x_i, dim=1).unsqueeze(1))
        out_max = torch.cat(self.out_max, dim=1)
        out_mean = torch.cat(self.out_mean, dim=1)
        out = torch.cat((out_max, out_mean), dim=1)
        x = self.conv(out) + ori
        return x
#-----------------------------------------end fs-------------------------------------------------------------   
class ICM(nn.Module):
     def __init__(self, n_feat, kernel_size, reduction, bias, act, train_mode):
        super(ICM, self).__init__()
        
        self.cab = CAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.sab = SAB(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.train_mode = train_mode
        self.out = nn.Conv2d(n_feat*3, n_feat, 1)
        
     def forward(self, x, mask, rand_width):
        w = 0.5
        if rand_width != -1:
          w = get_unknown_tensor_from_pred(mask, rand_width=rand_width, train_mode=self.train_mode)
        confident = x * (1-w)
        u_confident = x * w
        confident = self.cab(confident)
        u_confident = self.sab(u_confident)
        x1 = self.cab(x)
        x2 = self.sab(x)
        out = torch.cat((x1+x2, confident, u_confident), dim=1)
        out = self.out(out) 
        return out   
                  
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim,bias=bias)
    
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x,mask=None):
        b,c,h,w = x.shape
        q=self.qkv1conv(self.qkv_0(x))
        k=self.qkv2conv(self.qkv_1(x))
        v=self.qkv3conv(self.qkv_2(x))
        if mask is not None:
            q=q*mask
            k=k*mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
     
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
        
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)  
          
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        
        
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
class Local(nn.Module):
    def __init__(self, dim):
        super(Local, self).__init__()
        
        
        self.conv1 = BasicConv2d(dim, dim, 1)
        self.conv3_1 = BasicConv2d(dim//2, dim//2, 3,padding=1)
        self.conv3_2 = BasicConv2d(dim//2, dim//2, 3,padding=1)
        self.conv5_1 = BasicConv2d(dim//2, dim//2, 5,padding=2)
        self.conv5_2 = BasicConv2d(dim//2, dim//2, 5,padding=2)
        self.fusion1 = nn.Conv2d(dim, dim, 1)
        self.fusion2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        part_1, part_2 = x.chunk(2, 1)
        x1 = self.conv1(x)
        x3 = self.fusion1(torch.cat((self.conv3_1(part_1), self.conv3_2(part_2)), 1))
        x5 = self.fusion2(torch.cat((self.conv5_1(part_1), self.conv5_2(part_2)), 1))
        return x1 + x3 + x5   
                            
class ILM(nn.Module):
    def __init__(self, n_feat):
        super(ILM, self).__init__()
        self.norm = LayerNorm(n_feat)
        self.global_ = Attention(n_feat, 8, False)
        self.local = Local(n_feat)
        
    def forward(self, x):
        x_global = self.global_(self.norm(x)) 
        x_local = self.local(x)  
        x_ = x_global + x_local
        
        return x_ + x
           
class IdeNet(nn.Module):
    def __init__(self, n_feat=64,kernel_size=3,reduction=4,bias=False,act=nn.PReLU(), train_mode=True):
        super(IdeNet, self).__init__()

        self.backbone = pvt_v2_b4()  # [64, 128, 320, 512]
        path = './pretrained_pvt/pvt_v2_b4.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer1_1 = BasicConv2d(64, n_feat//2, 1)
        self.Translayer2_1 = BasicConv2d(128, n_feat//2, 1)
        self.Translayer3_1 = BasicConv2d(320, n_feat//2, 1)
        self.Translayer4_1 = BasicConv2d(512, n_feat//2, 1)
        
        self.fe = IAM(n_feat//2, n_feat)
        self.fs = IFM(groups=32, channels=n_feat*2, c_scale=2)
        
        self.compress = nn.Conv2d(n_feat*3, n_feat*2, 1)
        self.expand = nn.Conv2d(n_feat, n_feat*2, 1)
        
        self.decoder_level4 = ICM(n_feat*2, kernel_size, reduction, bias=bias, act=act, train_mode=train_mode)
        self.decoder_level3 = ICM(n_feat*2, kernel_size, reduction, bias=bias, act=act, train_mode=train_mode)
        self.decoder_level2 = ICM(n_feat*2, kernel_size, reduction, bias=bias, act=act, train_mode=train_mode)
        self.decoder_level1 = ICM(n_feat*2, kernel_size, reduction, bias=bias, act=act, train_mode=train_mode)
        
        self.gl4 = ILM(n_feat*2)
        self.gl3 = ILM(n_feat*2)
        self.gl2 = ILM(n_feat*2)
        self.gl1 = ILM(n_feat*2)
        
        self.out_p1 = nn.Conv2d(n_feat*2, 1, 3, padding=1)
        self.out_p2 = nn.Conv2d(n_feat*2, 1, 3, padding=1)
        self.out_p3 = nn.Conv2d(n_feat*2, 1, 3, padding=1)
        self.out_p4 = nn.Conv2d(n_feat*2, 1, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        
        
    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        x1_t = self.Translayer1_1(x1)#####channel=32
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)

        x1_fe, x2_fe, x3_fe, x4_fe = self.fe(x1_t), self.fe(x2_t), self.fe(x3_t), self.fe(x4_t) #channel=64
        
        
        
        x4_fs = self.fs(self.expand(x4_fe))#128
        #x4_fs = self.fs(x4_fe)
        x4_fs = self.decoder_level4(self.gl4(x4_fs), None, -1)#c=128
        p1 = self.out_p1(x4_fs)
        
        p1_up = self.upsample(p1)

        x4_fs_up = self.upsample(x4_fs)
        #x3_fs = self.fs(self.compress(torch.cat((x4_fs_up, x3_fe), 1)))#c=128
        x3_fs = self.fs(self.compress(torch.cat((x4_fs_up, x3_fe), 1)))#c=128
        x3_fs= self.decoder_level3(self.gl3(x3_fs), p1_up, 30)
        p2 = self.out_p2(x3_fs)
        
        p2_up = self.upsample(p2)
        
        x3_fs_up = self.upsample(x3_fs)
        #x2_fs = self.fs(self.compress(torch.cat((x3_fs_up, x2_fe), 1)))
        x2_fs = self.fs(self.compress(torch.cat((x3_fs_up, x2_fe), 1)))
        x2_fs = self.decoder_level2(self.gl2(x2_fs), p2_up, 20) 
        p3 = self.out_p3(x2_fs)
        
        p3_up = self.upsample(p3)
        
        x2_fs_up = self.upsample(x2_fs)
        #x1_fs = self.fs(self.compress(torch.cat((x2_fs_up, x1_fe), 1)))#c=128
        x1_fs = self.fs(self.compress(torch.cat((x2_fs_up, x1_fe), 1)))#c=128
        x1_fs = self.decoder_level1(self.gl1(x1_fs), p3_up, 15) 
        p4 = self.out_p4(x1_fs)
       
        p4 = F.interpolate(p4, scale_factor=4, mode='bilinear')
        p3 = F.interpolate(p3, scale_factor=8, mode='bilinear')
        p2 = F.interpolate(p2, scale_factor=16, mode='bilinear')
        p1 = F.interpolate(p1, scale_factor=32, mode='bilinear')
        
        return [p1, p2, p3, p4]



if __name__ == '__main__':
    model = Hitnet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())
