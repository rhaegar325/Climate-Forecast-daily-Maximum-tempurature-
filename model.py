import torch
import torch.nn as nn
import math
import numpy as np
from math import sqrt
from torchvision import datasets, models, transforms

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
        
class vdsr(nn.Module):
    def __init__(self):
        super(vdsr, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out

    
class cfsr(nn.Module):
    def __init__(self):
        super(cfsr, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        
        self.residual_layer_2 = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        
        
        return out
    
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std,channels, sign=-1):
        super(MeanShift, self).__init__(channels, channels, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(channels).view(channels, channels, 1, 1)
        self.weight.data.div_(std.view(channels, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False    
    
class vdsr_dem(nn.Module):
    def __init__(self):
        super(vdsr_dem, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        rgb_mean_dem=[0.05986051]
        rgb_std_dem = [1.0]
        self.sub_mean_dem = MeanShift(2228.3303, rgb_mean_dem, rgb_std_dem,1)  
        
        rgb_mean_pr=[0.00216697]
        rgb_std_pr = [1.0]
        self.sub_mean_pr =MeanShift(993.9646, rgb_mean_pr, rgb_std_pr,1)
        self.add_mean = MeanShift(993.9646, rgb_mean_pr, rgb_std_pr,1,1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    
    

    def forward(self, x,dem):
        dem=self.sub_mean_dem(dem)
        residual = self.sub_mean_pr(x)
        x_1=torch.cat((x,dem),dim=1)
        
        out = self.relu(self.input(x_1))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        out=self.add_mean(out)
        return out

        torch.autograd.set_detect_anomaly(True)
class YNet30_test(nn.Module):
    def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=4,use_climatology=False):
        super(YNet30_test, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale = scale
        self.use_climatology = use_climatology

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=0),
                                         nn.ReLU(inplace=True)))
        for i in range(self.num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=0),
                                             nn.ReLU(inplace=True)))

        for i in range(self.num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=0,output_padding=0),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=0, output_padding=0),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False),
                                                 nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True))
        #self.upsample = nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False)
        if self.use_climatology:
            self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels+2,self.num_features,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))#,
                                              #nn.ReLU(inplace=True))

    def forward(self, x, x2=None, x3=None):
        residual = x
        #residual_up = nn.functional.interpolate(residual,scale_factor=self.scale,mode='bilinear',align_corners=False)
        #residual_up = self.upsample(x)
        
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
    #        print(np.shape(x))
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
    #            print(np.shape(x))
                conv_feats.append(x)
        #print('after conv: x.size()={}\n'.format(x.size()))

        #for i in range(self.num_layers):
        #    x = self.deconv_layers[i](x)
        #      print(np.shape(x),i)
        
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
    #        print(np.shape(x),i)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats): 
    #        if (i + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
    #            print(np.shape(x),np.shape(conv_feat),i,self.num_layers)
                x = x + conv_feat
                x = self.relu(x)
        #print('torch.sum(x)={}'.format(torch.sum(x)))
        #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
        #x += residual
    #    print(x.size())
        x = x+residual
    #    print(x.size())
        x = self.relu(x)
    #    print('before subpixel conv: x.size()={}\n'.format(x.size()))
        x = self.subpixel_conv_layer(x)
    #    print('after subpixel conv: x.size()={}\n'.format(x.size()))
        #x = x+residual_up
        #x = x+x3
    #    x = torch.cat([x,x3],dim=1)
        #print('x2.size()={}\n'.format(x2.size()))
               
        if self.use_climatology and (x2 is not None):
            x = self.fusion_layer(torch.cat([x,x2],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
            #print('using fusion')
        
        #transform_sr=transforms.Compose([transforms.Resize((691,886)),
        #                                transforms.ToTensor()])
        #x=transform_sr(x.numpy())

        return x


class YNet30_test_dem(nn.Module):
    def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=4,use_climatology=False):
        super(YNet30_test_dem, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale = scale
        self.use_climatology = use_climatology

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=0),
                                         nn.ReLU(inplace=True)))
        for i in range(self.num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=0),
                                             nn.ReLU(inplace=True)))

        for i in range(self.num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=0,output_padding=0),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=0, output_padding=0),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False),
                                                 nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True))
        #self.upsample = nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False)
        if self.use_climatology:
            self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels,self.num_features,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))#,
                                              #nn.ReLU(inplace=True))

    def forward(self, x, x2=None, x3=None):
        residual = x
        #residual_up = nn.functional.interpolate(residual,scale_factor=self.scale,mode='bilinear',align_corners=False)
        #residual_up = self.upsample(x)
        
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            #print('conv:',x.shape)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                #print(x.shape)
                conv_feats.append(x)
        #print('after conv: x.size()={}\n'.format(x.size()))
        
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            #print("decon:",x.shape)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                #print(x.shape,conv_feat.shape)
                x = x + conv_feat
                x = self.relu(x)
        #print('torch.sum(x)={}'.format(torch.sum(x)))
        #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
        #x += residual
        x = x+residual
        x = self.relu(x)
        #print('before subpixel conv: x.size()={}\n'.format(x.size()))
        x = self.subpixel_conv_layer(x)
        #print('after subpixel conv: x.size()={}\n'.format(x.size()))
        #x = x+residual_up
        #x = x+x3
        #x = torch.cat([x,x3],dim=1)
        #print('x2.size()={}\n'.format(x2.size()))
               
        if self.use_climatology and (x2 is not None):
            x = self.fusion_layer(torch.cat([x,x2],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
            #print('using fusion')
        
        return x


class YNet30_test_dem_v1(nn.Module):
    def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=4,use_climatology=False):
        super(YNet30_test_dem, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale = scale
        self.use_climatology = use_climatology

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=0),
                                         nn.ReLU(inplace=True)))
        for i in range(self.num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=0),
                                             nn.ReLU(inplace=True)))

        for i in range(self.num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=0,output_padding=0),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=0, output_padding=0),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False),
                                                 nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True))
        #self.upsample = nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False)
        if self.use_climatology:
            self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels,self.num_features,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))#,
                                              #nn.ReLU(inplace=True))

    def forward(self, x, x2=None, x3=None):
        residual = x
        residual_up = nn.functional.interpolate(residual,scale_factor=self.scale,mode='bilinear',align_corners=False)
        #residual_up = self.upsample(x)
        
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            #print('conv:',x.shape)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                #print(x.shape)
                conv_feats.append(x)
        #print('after conv: x.size()={}\n'.format(x.size()))
        
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            #print("decon:",x.shape)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                #print(x.shape,conv_feat.shape)
                x = x + conv_feat
                x = self.relu(x)
        #print('torch.sum(x)={}'.format(torch.sum(x)))
        #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
        #x += residual
        x = x+residual
        x = self.relu(x)
        #print('before subpixel conv: x.size()={}\n'.format(x.size()))
        x = self.subpixel_conv_layer(x)
        #print('after subpixel conv: x.size()={}\n'.format(x.size()))
        x = x+residual_up
        #x = x+x3
        #x = torch.cat([x,x3],dim=1)
        #print('x2.size()={}\n'.format(x2.size()))
               
        if self.use_climatology and (x2 is not None):
            x = self.fusion_layer(torch.cat([x,x2],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
            #print('using fusion')
        
        return x


class YNet30_test_dem_v1(nn.Module):
    def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=4,use_climatology=False):
        super(YNet30_test_dem_v1, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale = scale
        self.use_climatology = use_climatology

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=0),
                                         nn.ReLU(inplace=True)))
        for i in range(self.num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=0),
                                             nn.ReLU(inplace=True)))

        for i in range(self.num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=0,output_padding=0),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=0, output_padding=0),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False),
                                                 nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True))
        #self.upsample = nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False)
        if self.use_climatology:
            self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels,self.num_features,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))#,
                                              #nn.ReLU(inplace=True))

    def forward(self, x, x2=None, x3=None):
        residual = x
        residual_up = nn.functional.interpolate(residual,scale_factor=self.scale,mode='bilinear',align_corners=False)
        #residual_up = self.upsample(x)
        
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            #print('conv:',x.shape)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                #print(x.shape)
                conv_feats.append(x)
        #print('after conv: x.size()={}\n'.format(x.size()))
        
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            #print("decon:",x.shape)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                #print(x.shape,conv_feat.shape)
                x = x + conv_feat
                x = self.relu(x)
        #print('torch.sum(x)={}'.format(torch.sum(x)))
        #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
        #x += residual
        x = x+residual
        x = self.relu(x)
        #print('before subpixel conv: x.size()={}\n'.format(x.size()))
        x = self.subpixel_conv_layer(x)
        #print('after subpixel conv: x.size()={}\n'.format(x.size()))
        x = x+residual_up
        #x = x+x3
        #x = torch.cat([x,x3],dim=1)
        #print('x2.size()={}\n'.format(x2.size()))
               
        if self.use_climatology and (x2 is not None):
            x = self.fusion_layer(torch.cat([x,x2],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
            #print('using fusion')
        
        return x

class YNet30_test_dem_v2(nn.Module):
    def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=4,use_climatology=False):
        super(YNet30_test_dem_v2, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale = scale
        self.use_climatology = use_climatology

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=0),
                                         nn.ReLU(inplace=True)))
        for i in range(self.num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=0),
                                             nn.ReLU(inplace=True)))

        for i in range(self.num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=0,output_padding=0),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=0, output_padding=0),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False),
                                                 nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True))
        #self.upsample = nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False)
        if self.use_climatology:
            self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels+1,self.num_features,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))#,
                                              #nn.ReLU(inplace=True))

    def forward(self, x, x2=None, x3=None):
        residual = x
        residual_up = nn.functional.interpolate(residual,scale_factor=self.scale,mode='bilinear',align_corners=False)
        #residual_up = self.upsample(x)
        
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            #print('conv:',x.shape)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                #print(x.shape)
                conv_feats.append(x)
        #print('after conv: x.size()={}\n'.format(x.size()))
        
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            #print("decon:",x.shape)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                #print(x.shape,conv_feat.shape)
                x = x + conv_feat
                x = self.relu(x)
        #print('torch.sum(x)={}'.format(torch.sum(x)))
        #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
        #x += residual
        x = x+residual
        x = self.relu(x)
        #print('before subpixel conv: x.size()={}\n'.format(x.size()))
        x = self.subpixel_conv_layer(x)
        #print('after subpixel conv: x.size()={}\n'.format(x.size()))
        #x = x+residual_up
        #x = x+x3
        x = torch.cat([x,residual_up],dim=1)
        #print('x2.size()={}\n'.format(x2.size()))
               
        if self.use_climatology and (x2 is not None):
            x = self.fusion_layer(torch.cat([x,x2],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
            #print('using fusion')
        
        return x


    
    
class YNet30_test_up_bicubic(nn.Module):
    def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=4,use_climatology=False):
        super(YNet30_test_up_bicubic, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale = scale
        self.use_climatology = use_climatology

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=0),
                                         nn.ReLU(inplace=True)))
        for i in range(self.num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=0),
                                             nn.ReLU(inplace=True)))

        for i in range(self.num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=0,output_padding=0),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=0, output_padding=0),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Upsample(scale_factor=self.scale,mode='bicubic',align_corners=False),
                                                 nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True))
        #self.upsample = nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False)
        if self.use_climatology:
            self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels+2,self.num_features,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))#,
                                              #nn.ReLU(inplace=True))

    def forward(self, x, x2=None, x3=None):
        residual = x
        #residual_up = nn.functional.interpolate(residual,scale_factor=self.scale,mode='bilinear',align_corners=False)
        #residual_up = self.upsample(x)
        
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
    #        print(np.shape(x))
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
    #            print(np.shape(x))
                conv_feats.append(x)
        #print('after conv: x.size()={}\n'.format(x.size()))

        #for i in range(self.num_layers):
        #    x = self.deconv_layers[i](x)
        #      print(np.shape(x),i)
        
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
    #        print(np.shape(x),i)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats): 
    #        if (i + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
    #            print(np.shape(x),np.shape(conv_feat),i,self.num_layers)
                x = x + conv_feat
                x = self.relu(x)
        #print('torch.sum(x)={}'.format(torch.sum(x)))
        #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
        #x += residual
    #    print(x.size())
        x = x+residual
    #    print(x.size())
        x = self.relu(x)
    #    print('before subpixel conv: x.size()={}\n'.format(x.size()))
        x = self.subpixel_conv_layer(x)
    #    print('after subpixel conv: x.size()={}\n'.format(x.size()))
        #x = x+residual_up
        #x = x+x3
    #    x = torch.cat([x,x3],dim=1)
        #print('x2.size()={}\n'.format(x2.size()))
               
        if self.use_climatology and (x2 is not None):
            x = self.fusion_layer(torch.cat([x,x2],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
            #print('using fusion')
        
        #transform_sr=transforms.Compose([transforms.Resize((691,886)),
        #                                transforms.ToTensor()])
        #x=transform_sr(x.numpy())

        return x

class YNet30_test_up_lr_upsample(nn.Module):
    def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=4,use_climatology=False):
        super(YNet30_test_up_lr_upsample, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale = scale
        self.use_climatology = use_climatology

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=0),
                                         nn.ReLU(inplace=True)))
        for i in range(self.num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=0),
                                             nn.ReLU(inplace=True)))

        for i in range(self.num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=0,output_padding=0),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=0, output_padding=0),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False),
                                                 nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True))
        self.upsample = nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False)
        if self.use_climatology:
            self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels+2,self.num_features,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))#,
                                              #nn.ReLU(inplace=True))

    def forward(self, x, x2=None, x3=None):
        residual = x
        #residual_up = nn.functional.interpolate(residual,scale_factor=self.scale,mode='bilinear',align_corners=False)
        residual_up = self.upsample(x)
        
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
    #        print(np.shape(x))
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
    #            print(np.shape(x))
                conv_feats.append(x)
        #print('after conv: x.size()={}\n'.format(x.size()))

        #for i in range(self.num_layers):
        #    x = self.deconv_layers[i](x)
        #      print(np.shape(x),i)
        
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
    #        print(np.shape(x),i)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats): 
    #        if (i + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
    #            print(np.shape(x),np.shape(conv_feat),i,self.num_layers)
                x = x + conv_feat
                x = self.relu(x)
        #print('torch.sum(x)={}'.format(torch.sum(x)))
        #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
        #x += residual
    #    print(x.size())
        x = x+residual
    #    print(x.size())
        x = self.relu(x)
    #    print('before subpixel conv: x.size()={}\n'.format(x.size()))
        x = self.subpixel_conv_layer(x)
    #    print('after subpixel conv: x.size()={}\n'.format(x.size()))
        x = x+residual_up
        #x = x+x3
    #    x = torch.cat([x,x3],dim=1)
        #print('x2.size()={}\n'.format(x2.size()))
               
        if self.use_climatology and (x2 is not None):
            x = self.fusion_layer(torch.cat([x,x2],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
            #print('using fusion')
        
        
        #transform_sr=transforms.Compose([transforms.Resize((691,886)),
        #                                transforms.ToTensor()])
        #x=transform_sr(x.numpy())

        return x


class YNet30_test_up_lr_fusion(nn.Module):
    def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=4,use_climatology=False):
        super(YNet30_test_up_lr_fusion, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale = scale
        self.use_climatology = use_climatology

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=0),
                                         nn.ReLU(inplace=True)))
        for i in range(self.num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=0),
                                             nn.ReLU(inplace=True)))

        for i in range(self.num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=0,output_padding=0),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=0, output_padding=0),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False),
                                                 nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True))
        #self.upsample = nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False)
        if self.use_climatology:
            self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels+2,self.num_features,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))#,
                                              #nn.ReLU(inplace=True))
        self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels,self.num_features,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))#,
                                              #nn.ReLU(inplace=True))                                    

    def forward(self, x, x2=None, x3=None):
        residual = x
        residual_up = nn.functional.interpolate(residual,scale_factor=self.scale,mode='bilinear',align_corners=False)
        #residual_up = self.upsample(x)
        
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
    #        print(np.shape(x))
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
    #            print(np.shape(x))
                conv_feats.append(x)
        #print('after conv: x.size()={}\n'.format(x.size()))

        #for i in range(self.num_layers):
        #    x = self.deconv_layers[i](x)
        #      print(np.shape(x),i)
        
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
    #        print(np.shape(x),i)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats): 
    #        if (i + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
    #            print(np.shape(x),np.shape(conv_feat),i,self.num_layers)
                x = x + conv_feat
                x = self.relu(x)
        #print('torch.sum(x)={}'.format(torch.sum(x)))
        #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
        #x += residual
    #    print(x.size())
        x = x+residual
    #    print(x.size())
        x = self.relu(x)
    #    print('before subpixel conv: x.size()={}\n'.format(x.size()))
        x = self.subpixel_conv_layer(x)
    #    print('after subpixel conv: x.size()={}\n'.format(x.size()))
        #x = x+residual_up
        #x = x+x3
    #    x = torch.cat([x,x3],dim=1)
        #print('x2.size()={}\n'.format(x2.size()))
               
        if self.use_climatology and (x2 is not None):
            x = self.fusion_layer(torch.cat([x,x2],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
            #print('using fusion')
        x = self.fusion_layer(torch.cat([x,residual_up],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
        
        #transform_sr=transforms.Compose([transforms.Resize((691,886)),
        #                                transforms.ToTensor()])
        #x=transform_sr(x.numpy())

        return x


class YNet30_test_up_bilinear_a(nn.Module):
    def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=4,use_climatology=False):
        super(YNet30_test_up_bilinear_a, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale = scale
        self.use_climatology = use_climatology

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=0),
                                         nn.ReLU(inplace=True)))
        for i in range(self.num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=0),
                                             nn.ReLU(inplace=True)))

        for i in range(self.num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=0,output_padding=0),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=0, output_padding=0),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=True),
                                                 nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True))
        #self.upsample = nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False)
        if self.use_climatology:
            self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels+2,self.num_features,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))#,
                                              #nn.ReLU(inplace=True))

    def forward(self, x, x2=None, x3=None):
        residual = x
        #residual_up = nn.functional.interpolate(residual,scale_factor=self.scale,mode='bilinear',align_corners=False)
        #residual_up = self.upsample(x)
        
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
    #        print(np.shape(x))
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
    #            print(np.shape(x))
                conv_feats.append(x)
        #print('after conv: x.size()={}\n'.format(x.size()))

        #for i in range(self.num_layers):
        #    x = self.deconv_layers[i](x)
        #      print(np.shape(x),i)
        
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
    #        print(np.shape(x),i)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats): 
    #        if (i + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
    #            print(np.shape(x),np.shape(conv_feat),i,self.num_layers)
                x = x + conv_feat
                x = self.relu(x)
        #print('torch.sum(x)={}'.format(torch.sum(x)))
        #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
        #x += residual
    #    print(x.size())
        x = x+residual
    #    print(x.size())
        x = self.relu(x)
    #    print('before subpixel conv: x.size()={}\n'.format(x.size()))
        x = self.subpixel_conv_layer(x)
    #    print('after subpixel conv: x.size()={}\n'.format(x.size()))
        #x = x+residual_up
        #x = x+x3
    #    x = torch.cat([x,x3],dim=1)
        #print('x2.size()={}\n'.format(x2.size()))
               
        if self.use_climatology and (x2 is not None):
            x = self.fusion_layer(torch.cat([x,x2],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
            #print('using fusion')
        
        #transform_sr=transforms.Compose([transforms.Resize((691,886)),
        #                                transforms.ToTensor()])
        #x=transform_sr(x.numpy())

        return x





class YNet30_test_section_1(nn.Module):
    def __init__(self, num_layers=15, num_features=64,input_channels=1,output_channels=1,scale=8,use_climatology=False):
        super(YNet30_test_section_1, self).__init__()
        #self.num_layers = num_layers
        self.atom_num_layers = 9
        self.num_layers = self.atom_num_layers * (math.log2(scale))
        self.num_features = num_features
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale = scale
        self.use_climatology = use_climatology

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=0),
                                         nn.ReLU(inplace=True)))
        for i in range(self.atom_num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=0),
                                             nn.ReLU(inplace=True)))

        for i in range(self.atom_num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=0,output_padding=0),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=0, output_padding=0),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Upsample(scale_factor=self.scale/4,mode='bilinear',align_corners=False),
                                                 nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True))
        
        self.upsample_8 = nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False)

        self.upsample_2 = nn.Upsample(scale_factor=self.scale/4,mode='bilinear',align_corners=False)
        
        if self.use_climatology:
            self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels+2,self.num_features,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))#,
                                              #nn.ReLU(inplace=True))
        self.fusion_layer = nn.Sequential(nn.Conv2d(2*self.input_channels,self.num_features,kernel_size=3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))#,
                                              #nn.ReLU(inplace=True))                                

    def forward(self, x, x2=None, x3=None):
        residual = x
        #residual_up = nn.functional.interpolate(residual,scale_factor=self.scale,mode='bilinear',align_corners=False)
        residual_up_8 = self.upsample_8(x)

        list_us=[]

        for j in range(int(math.log2(self.scale))):
            residual_1 = x
            residual_up_2 = self.upsample_2(x)
            list_us.append(residual_up_2)


            conv_feats = []
            for i in range(self.atom_num_layers):
                x = self.conv_layers[i](x)
        #        print(np.shape(x))
                if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
        #            print(np.shape(x))
                    conv_feats.append(x)
            #print('after conv: x.size()={}\n'.format(x.size()))

            #for i in range(self.num_layers):
            #    x = self.deconv_layers[i](x)
            #      print(np.shape(x),i)
            
            conv_feats_idx = 0 
            for i in range(self.atom_num_layers):
                x = self.deconv_layers[i](x)
        #        print(np.shape(x),i)
                if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats): 
        #        if (i + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                    conv_feat = conv_feats[-(conv_feats_idx + 1)]
                    conv_feats_idx += 1
        #            print(np.shape(x),np.shape(conv_feat),i,self.num_layers)
                    x = x + conv_feat
                    x = self.relu(x)
            #print('torch.sum(x)={}'.format(torch.sum(x)))
            #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
            #x += residual
        #    print(x.size())
            x = x+residual_1
        #    print(x.size())
            x = self.relu(x)
        #    print('before subpixel conv: x.size()={}\n'.format(x.size()))
            x = self.subpixel_conv_layer(x)

            x = x+residual_up_2



        
        """ conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
    #        print(np.shape(x))
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
    #            print(np.shape(x))
                conv_feats.append(x)
        #print('after conv: x.size()={}\n'.format(x.size()))

        #for i in range(self.num_layers):
        #    x = self.deconv_layers[i](x)
        #      print(np.shape(x),i)
        
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
    #        print(np.shape(x),i)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats): 
    #        if (i + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
    #            print(np.shape(x),np.shape(conv_feat),i,self.num_layers)
                x = x + conv_feat
                x = self.relu(x)
        #print('torch.sum(x)={}'.format(torch.sum(x)))
        #print('after convtrans: x.size()={},residual.size()={}'.format(x.size(),residual.size()))
        #x += residual
    #    print(x.size())
        x = x+residual
    #    print(x.size())
        x = self.relu(x)
    #    print('before subpixel conv: x.size()={}\n'.format(x.size()))
        x = self.subpixel_conv_layer(x) """
    #    print('after subpixel conv: x.size()={}\n'.format(x.size()))
        #x = x+residual_up_8
        #list_us.append(residual_up_8)
        #x = x+x3
    #    x = torch.cat([x,x3],dim=1)
        #print('x2.size()={}\n'.format(x2.size()))
               
        if self.use_climatology and (x2 is not None):
            x = self.fusion_layer(torch.cat([x,x2],dim=1)) # [Nbatch,Nchannel,Nlat,Nlon]
            #print('using fusion')
        
        #x = self.fusion_layer(torch.cat([x,residual_up_8,list_us[0],list_us[1],list_us[2]],dim=1))
        #for i in range(len(list_us)):
        x = self.fusion_layer(torch.cat([x,residual_up_8],dim=1))
        
        #transform_sr=transforms.Compose([transforms.Resize((691,886)),
        #                                transforms.ToTensor()])
        #x=transform_sr(x.numpy())

        return x


class REDNet30(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_layers=15, num_features=64):
        super(REDNet30, self).__init__()
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.output_channels = output_channels

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, self.output_channels, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)
        self.upsample_8 = nn.Upsample(scale_factor=8,mode='bilinear',align_corners=False)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                print(x.shape,conv_feat.shape)
                x = x + conv_feat
                x = self.relu(x)
        print(x.shape,residual.shape)
        resize_red=transforms.Resize([93,81])
        x=resize_red(x)

        x += residual
        x = self.upsample_8(x)
        x = self.relu(x)

        return x


class ESPCNNet(nn.Module):
    def __init__(self, upscale_factor,input_channels=1,output_channels=1):
        super(ESPCNNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = nn.Conv2d(self.input_channels, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, self.output_channels * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        #x = F.tanh(self.conv1(x))
        #x = F.tanh(self.conv2(x))
        #x = F.sigmoid(self.pixel_shuffle(self.conv3(x)))
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv3(x)))
        return x


