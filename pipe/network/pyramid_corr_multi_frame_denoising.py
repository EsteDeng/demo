import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, './module/')
from dataset import *
from module.activation import *
from module.conv import conv
from module.dfus_block import dfus_block_add_output_conv
from module.utils import bilinear_warp, costvolumelayer

PI = 3.14159265358979323846
flg = False
dtype = torch.float32


class FeatureExtractorSubnet(nn.Module):
    def __init__(self, in_channels=2, use_bn=False, num=1, flg=None, regular=None, batch_size=None, deformable_range=None):
        super().__init__()
        self.n_filters = [16, 32, 64, 96, 128, 192]
        self.pool_sizes = [1, 2, 2, 2, 2, 2]
        self.use_bn = use_bn
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        for i in range(len(self.n_filters)):
            in_ch = in_channels if i == 0 else self.n_filters[i - 1]
            self.conv_layers.append(nn.Conv2d(in_ch, self.n_filters[i], kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm2d(self.n_filters[i]) if use_bn else nn.Identity())
            self.pool_layers.append(
                nn.MaxPool2d(kernel_size=self.pool_sizes[i], stride=self.pool_sizes[i])
                if self.pool_sizes[i] > 1 else nn.Identity()
            )
            self.attention_blocks.append(
                ResidualChannelAttentionBlock(self.n_filters[i], reduction_num=2, flg=flg, regular=regular, batch_size=batch_size, deformable_range=deformable_range)
            )
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range

    def forward(self, x):
        features = []
        for i in range(len(self.n_filters)):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = self.pool_layers[i](x)
            x = self.attention_blocks[i](x)
            features.append(x)
        return features

class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, in_channel, reduction_num, flg=None, regular=None, batch_size=None, deformable_range=None):
        super(ResidualChannelAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channel, in_channel // reduction_num, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channel // reduction_num, in_channel, kernel_size=1)
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range
        
    def forward(self, x):
        f = self.conv1(x)
        f = F.relu(f)
        f = self.conv2(f)
        
        y = torch.mean(f, dim=[2, 3], keepdim=True)
        y = self.conv3(y)
        y = F.relu(y)
        y = self.conv4(y)
        y = torch.sigmoid(y)
        
        return x + f * y

class DepthResidualRegressionSubnet(nn.Module):
    def __init__(self, subnet_num, flg=None, regular=None, batch_size=None, deformable_range=None):
        super(DepthResidualRegressionSubnet, self).__init__()
        self.pref = f'depth_regression_subnet_{subnet_num}_'
        
        self.n_filters = [128, 96, 64, 32, 16, 1]
        self.filter_sizes = [3] * len(self.n_filters)
        
        self.conv_layers = nn.ModuleList()
        for i in range(len(self.n_filters)):
            in_channels = 128 if i == 0 else self.n_filters[i-1]
            self.conv_layers.append(
                nn.Conv2d(in_channels, 
                         self.n_filters[i],
                         kernel_size=self.filter_sizes[i],
                         padding=1)
            )
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range
            
    def forward(self, x):
        current_input = x
        for i in range(len(self.n_filters)-1):
            current_input = self.conv_layers[i](current_input)
            current_input = F.relu(current_input)
            
        current_input = self.conv_layers[-1](current_input)
        return current_input

class CorrFeatureRegressionSubnet(nn.Module):
    def __init__(self, subnet_num, flg=None, regular=None, batch_size=None, deformable_range=None):
        super(CorrFeatureRegressionSubnet, self).__init__()
        self.pref = f'corr_feature_regression_subnet_{subnet_num}_'
        
        self.n_filters = [32, 16]
        self.filter_sizes = [3] * len(self.n_filters)
        
        self.conv_layers = nn.ModuleList()
        for i in range(len(self.n_filters)):
            in_channels = 128 if i == 0 else self.n_filters[i-1]
            self.conv_layers.append(
                nn.Conv2d(in_channels,
                         self.n_filters[i],
                         kernel_size=self.filter_sizes[i],
                         padding=1)
            )
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range
            
    def forward(self, x):
        current_input = x
        for i in range(len(self.n_filters)-1):
            current_input = self.conv_layers[i](current_input)
            current_input = F.relu(current_input)
            
        current_input = self.conv_layers[-1](current_input)
        return current_input

class MaskRegressionSubnet(nn.Module):
    def __init__(self, subnet_num, flg=None, regular=None, batch_size=None, deformable_range=None):
        super(MaskRegressionSubnet, self).__init__()
        self.pref = f'mask_regression_subnet_{subnet_num}_'
        
        self.n_filters = [32, 1]
        self.filter_sizes = [3] * len(self.n_filters)
        
        self.conv_layers = nn.ModuleList()
        for i in range(len(self.n_filters)):
            in_channels = 128 if i == 0 else self.n_filters[i-1]
            self.conv_layers.append(
                nn.Conv2d(in_channels,
                         self.n_filters[i],
                         kernel_size=self.filter_sizes[i],
                         padding=1)
            )
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range
            
    def forward(self, x):
        current_input = x
        for i in range(len(self.n_filters)-1):
            current_input = self.conv_layers[i](current_input)
            current_input = F.relu(current_input)
            
        current_input = self.conv_layers[-1](current_input)
        return current_input

class ResidualOutputSubnet(nn.Module):
    def __init__(self, subnet_num, flg=None, regular=None, batch_size=None, deformable_range=None):
        super(ResidualOutputSubnet, self).__init__()
        self.pref = f'residual_output_subnet_{subnet_num}_'
        
        self.conv = nn.Conv2d(5, 1, kernel_size=1)
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range
        
    def forward(self, x):
        return self.conv(x)

class DepthOutputSubnet(nn.Module):
    def __init__(self, kernel_size, flg=None, regular=None, batch_size=None, deformable_range=None):
        super(DepthOutputSubnet, self).__init__()
        self.pref = 'depth_output_subnet_'
        
        self.conv = nn.Conv2d(64, kernel_size**2, kernel_size=1)
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range
        
    def forward(self, x):
        current_input = self.conv(x)
        return torch.sigmoid(current_input)

class DearKPN(nn.Module):
    def __init__(self, kernel_size=3, flg=None, regular=None, batch_size=None, deformable_range=None):
        super(DearKPN, self).__init__()
        self.kernel_size = kernel_size
        self.unet = UNetSubnet(flg, regular, batch_size, deformable_range)
        self.depth_output = DepthOutputSubnet(self.kernel_size, flg, regular, batch_size, deformable_range)
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range
        
    def forward(self, x):
        features = self.unet(x)
        weights = self.depth_output(features)
        weights = weights / (torch.sum(torch.abs(weights) + 1e-6, dim=1, keepdim=True))
        
        column = im2col(x, kernel_size=self.kernel_size)
        current_output = torch.sum(column * weights, dim=1, keepdim=True)
        
        return current_output

class UNetSubnet(nn.Module):
    def __init__(self, flg=None, regular=None, batch_size=None, deformable_range=None):
        super(UNetSubnet, self).__init__()
        self.pref = 'unet_subnet_'
        
        self.n_filters = [16, 16, 32, 32, 64, 64, 128, 128]
        self.filter_sizes = [3] * len(self.n_filters)
        self.pool_sizes = [1, 1, 2, 1, 2, 1, 2, 1]
        self.pool_strides = [1, 1, 2, 1, 2, 1, 2, 1]
        self.skips = [False, False, True, False, True, False, True, False]
        
        # Encoder
        self.encoder_conv = nn.ModuleList()
        self.encoder_pool = nn.ModuleList()
        
        for i in range(len(self.n_filters)):
            in_channels = 2 if i == 0 else self.n_filters[i-1]
            self.encoder_conv.append(
                nn.Conv2d(in_channels,
                         self.n_filters[i],
                         kernel_size=self.filter_sizes[i],
                         padding=1)
            )
            
        # Decoder
        self.decoder_conv = nn.ModuleList()
        
        for i in range(len(self.n_filters)-2, 0, -1):
            if self.skips[i] and self.skips[i-1]:
                in_channels = self.n_filters[i] + self.n_filters[i-1]
            else:
                in_channels = self.n_filters[i]
                
            self.decoder_conv.append(
                nn.ConvTranspose2d(in_channels,
                                 self.n_filters[i-1],
                                 kernel_size=self.filter_sizes[i],
                                 stride=self.pool_strides[i],
                                 padding=1)
            )
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        self.batch_size = batch_size
        self.deformable_range = deformable_range
            
    def forward(self, x):
        # Encoder
        conv_outputs = []
        pool_outputs = [x]
        current_input = x
        
        for i in range(len(self.n_filters)):
            current_input = self.encoder_conv[i](current_input)
            current_input = F.relu(current_input)
            
            if self.pool_sizes[i] > 1:
                current_input = F.max_pool2d(current_input,
                                           kernel_size=self.pool_sizes[i],
                                           stride=self.pool_strides[i])
                
            conv_outputs.append(current_input)
            pool_outputs.append(current_input)
            
        # Decoder
        current_input = pool_outputs[-1]
        for i in range(len(self.decoder_conv)):
            current_input = self.decoder_conv[i](current_input)
            current_input = F.relu(current_input)
            
            if self.skips[i] and self.skips[i-1]:
                current_input = torch.cat([current_input, pool_outputs[i+1]], dim=1)
                
        return current_input

class PyramidCorrMaskMultiFrameDenoising(nn.Module):
    def __init__(self, batch_size, deformable_range, flg=None, regular=None):
        super(PyramidCorrMaskMultiFrameDenoising, self).__init__()
        self.batch_size = batch_size
        self.deformable_range = deformable_range
        self.depth_residual_weight = [0.32, 0.08, 0.02, 0.01, 0.005]
        
        # Feature extractors
        self.feature_extractor1 = FeatureExtractorSubnet(num=1, flg=flg, regular=regular, batch_size=batch_size, deformable_range=deformable_range)
        self.feature_extractor2 = FeatureExtractorSubnet(num=1, flg=flg, regular=regular, batch_size=batch_size, deformable_range=deformable_range)
        
        # Correlation feature regression
        self.corr_feature_regression = nn.ModuleList([
            CorrFeatureRegressionSubnet(i, flg=flg, regular=regular, batch_size=batch_size, deformable_range=deformable_range) for i in range(1, 3)
        ])
        
        # Mask regression
        self.mask_regression = nn.ModuleList([
            MaskRegressionSubnet(i, flg=flg, regular=regular, batch_size=batch_size, deformable_range=deformable_range) for i in range(1, 6)
        ])
        
        # Depth residual regression
        self.depth_residual_regression = nn.ModuleList([
            DepthResidualRegressionSubnet(i, flg=flg, regular=regular, batch_size=batch_size, deformable_range=deformable_range) for i in range(1, 6)
        ])
        
        # Residual output
        self.residual_output = ResidualOutputSubnet(0, flg=flg, regular=regular, batch_size=batch_size, deformable_range=deformable_range)
        
        # Final KPN
        self.dear_kpn = DearKPN(flg=flg, regular=regular, batch_size=batch_size, deformable_range=deformable_range)
        
        # 保留参数以兼容接口
        self.flg = flg
        self.regular = regular
        
    def forward(self, x):
        depth_residual = []
        depth_residual_input = []
        d_conf_list = []
        corr_feature_list = []
        corr_feature_in_list = []

        # Extract depth and amplitude
        print("x:", x.shape) 
        depth = x[:, :, :, 0:1]
        amplitude = x[:, :, :, 1:2]
        depth_2 = x[:, :, :, 2:3]
        amplitude_2 = x[:, :, :, 3:4]
        print("depth shape:", depth.shape)  # 应该是[B, 2, H, W]
        print("amplitude shape:", amplitude.shape)  # 可能是[B, 10, H, W]等

        x1_input = torch.cat([depth, amplitude], dim=1)
        x2_input = torch.cat([depth_2, amplitude_2], dim=1)
        print("feature_extractor1 input shape:", x1_input.shape)  # 应该是[B, 2, H, W]

        
        # Extract features
        features1 = self.feature_extractor1(x1_input)
        features2 = self.feature_extractor2(x2_input)

        low_num = 2
        for i in range(1, len(features1) + 1):
            if i == 1:
                cost = costvolumelayer(features1[len(features1)-i], features2[len(features1)-i], search_range=3)
                cost_in = costvolumelayer(features1[len(features1)-i], features2[len(features1)-i], search_range=3)
                feature_input = features1[len(features1)-i]
                feature_input_2 = features2[len(features1)-i]
                inputs = torch.cat([feature_input, cost_in], dim=1)
                m_inputs = torch.cat([feature_input, feature_input_2, cost], dim=1)
            else:
                feature_input = features1[len(features1)-i]
                feature_input_2 = features2[len(features1)-i]
                
                # Resize previous outputs
                depth_coarse_input = F.interpolate(depth_residual[-1], 
                                                 size=(feature_input.shape[2], feature_input.shape[3]),
                                                 mode='bicubic',
                                                 align_corners=True)
                d_conf = F.interpolate(d_conf_list[-1],
                                     size=(feature_input.shape[2], feature_input.shape[3]),
                                     mode='bicubic',
                                     align_corners=True)
                corr_feature = F.interpolate(corr_feature_list[-1],
                                          size=(feature_input.shape[2], feature_input.shape[3]),
                                          mode='bicubic',
                                          align_corners=True)
                corr_feature_in = F.interpolate(corr_feature_in_list[-1],
                                             size=(feature_input.shape[2], feature_input.shape[3]),
                                             mode='bicubic',
                                             align_corners=True)
                
                if i < low_num:
                    cost = costvolumelayer(features1[len(features1)-i], features2[len(features2)-i], search_range=3)
                    cost_in = costvolumelayer(features1[len(features1)-i], features2[len(features1)-i], search_range=3)
                    
                m_inputs = torch.cat([feature_input, feature_input_2, corr_feature, d_conf], dim=1)
                inputs = torch.cat([feature_input, corr_feature_in, depth_coarse_input], dim=1)
                
            if i < low_num:
                corr_feature = self.corr_feature_regression[i-1](m_inputs)
                corr_feature_in = self.corr_feature_regression[i-1](inputs) # Corrected index here from i to i-1
                corr_feature_list.append(corr_feature)
                corr_feature_in_list.append(corr_feature_in)
            else:
                # If i >= low_num, these aren't updated in the loop. Need to ensure they are defined.
                # For now, let's just append the last valid values. This might need more thought based on original TF logic.
                # Assuming they are still needed for inputs/m_inputs even if not updated by subnet.
                if not corr_feature_list: # First iteration when low_num is higher than 1
                    # This case needs to be handled based on where corr_feature and corr_feature_in originate in the TF code
                    pass # This branch needs careful review of TF code if it's hit
                else:
                    corr_feature = corr_feature_list[-1] # Use last generated if not updated
                    corr_feature_in = corr_feature_in_list[-1] # Use last generated if not updated

            d_conf = self.mask_regression[i-1](m_inputs)
            d_conf = F.softmax(d_conf, dim=1)
            d_conf_list.append(d_conf)
            
            inputs = torch.cat([inputs, corr_feature], dim=1)
            current_depth_residual = self.depth_residual_regression[i-1](inputs)
            current_depth_residual = current_depth_residual * d_conf
            depth_residual.append(current_depth_residual)
            
            current_depth_residual_input = F.interpolate(current_depth_residual,
                                                      size=(x.shape[2], x.shape[3]), # Corrected size (H, W)
                                                      mode='bicubic',
                                                      align_corners=True)
            depth_residual_input.append(current_depth_residual_input)

        depth_coarse_residual_input = torch.cat(depth_residual_input, dim=1)
        final_depth_residual_output = self.residual_output(depth_coarse_residual_input)
        current_final_depth_output = depth + final_depth_residual_output
        final_depth_output = self.dear_kpn(current_final_depth_output)

        depth_residual_input.append(final_depth_residual_output)
        depth_residual_input.append(final_depth_output - current_final_depth_output)
        
        return final_depth_output, torch.cat(depth_residual_input, dim=1)
    