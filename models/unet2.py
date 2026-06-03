import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0: 
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

# ========== 新增：TV Loss 类 ==========
class TVLoss(nn.Module):
    """总变分损失，用于抑制棋盘格伪影"""
    def __init__(self, weight=1e-7):
        super(TVLoss, self).__init__()
        self.weight = weight
    
    def forward(self, x):
        batch_size = x.size(0)
        h_tv = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        w_tv = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return self.weight * (h_tv + w_tv) * batch_size


class UNet(nn.Module):
    '''
    UNet - 修复棋盘格伪影的版本
    upsample_mode: ['bilinear', 'nearest', 'deconv']，但建议使用'bilinear'避免伪影
    '''
    def __init__(self, num_input_channels=3, num_output_channels=3, 
                       feature_scale=4, more_layers=0, concat_x=False,
                       upsample_mode='bilinear', pad='zero', norm_layer=nn.InstanceNorm2d, 
                       need_sigmoid=True, need_bias=True, use_tv_loss=False, tv_weight=1e-7):
        super(UNet, self).__init__()

        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x
        self.use_tv_loss = use_tv_loss
        self.tv_weight = tv_weight
        
        # 如果使用TV Loss，初始化
        if self.use_tv_loss:
            self.tv_loss = TVLoss(weight=tv_weight)
        
        # 强制使用安全上采样模式
        if upsample_mode == 'deconv':
            print("警告: 'deconv'模式可能导致棋盘格伪影，建议使用'bilinear'")
            # 自动修正为bilinear
            upsample_mode = 'bilinear'
        
        if upsample_mode not in ['bilinear', 'nearest', 'deconv']:
            raise ValueError(f"upsample_mode必须是'bilinear', 'nearest'或'deconv'，当前为'{upsample_mode}'")

        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        self.start = unetConv2(num_input_channels, filters[0] if not concat_x else filters[0] - num_input_channels, 
                              norm_layer, need_bias, pad)

        self.down1 = unetDown(filters[0], filters[1] if not concat_x else filters[1] - num_input_channels, 
                             norm_layer, need_bias, pad)
        self.down2 = unetDown(filters[1], filters[2] if not concat_x else filters[2] - num_input_channels, 
                             norm_layer, need_bias, pad)
        self.down3 = unetDown(filters[2], filters[3] if not concat_x else filters[3] - num_input_channels, 
                             norm_layer, need_bias, pad)
        self.down4 = unetDown(filters[3], filters[4] if not concat_x else filters[4] - num_input_channels, 
                             norm_layer, need_bias, pad)

        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [
                unetDown(filters[4], filters[4] if not concat_x else filters[4] - num_input_channels, 
                        norm_layer, need_bias, pad) for i in range(self.more_layers)]
            self.more_ups = [SafeUnetUp(filters[4], upsample_mode, need_bias, pad, same_num_filt=True) 
                            for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups   = ListModule(*self.more_ups)

        # ========== 使用安全上采样替代原始上采样 ==========
        self.up4 = SafeUnetUp(filters[4], upsample_mode, need_bias, pad)
        self.up3 = SafeUnetUp(filters[3], upsample_mode, need_bias, pad)
        self.up2 = SafeUnetUp(filters[2], upsample_mode, need_bias, pad)
        self.up1 = SafeUnetUp(filters[1], upsample_mode, need_bias, pad)

        self.final = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad)
        
        # 添加输出归一化稳定训练
        if need_sigmoid: 
            self.final = nn.Sequential(self.final, nn.Sigmoid())
        else:
            self.final = nn.Sequential(self.final, nn.Identity())

    def forward(self, inputs):
        # Downsample 
        downs = [inputs]
        down = nn.AvgPool2d(2, 2)
        for i in range(4 + self.more_layers):
            downs.append(down(downs[-1]))

        in64 = self.start(inputs)
        if self.concat_x:
            in64 = torch.cat([in64, downs[0]], 1)

        down1 = self.down1(in64)
        if self.concat_x:
            down1 = torch.cat([down1, downs[1]], 1)

        down2 = self.down2(down1)
        if self.concat_x:
            down2 = torch.cat([down2, downs[2]], 1)

        down3 = self.down3(down2)
        if self.concat_x:
            down3 = torch.cat([down3, downs[3]], 1)

        down4 = self.down4(down3)
        if self.concat_x:
            down4 = torch.cat([down4, downs[4]], 1)

        if self.more_layers > 0:
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                out = d(prevs[-1])
                if self.concat_x:
                    out = torch.cat([out,  downs[kk + 5]], 1)
                prevs.append(out)

            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more_layers - idx - 2]
                up_ = l(up_, prevs[self.more_layers - idx - 2])
        else:
            up_ = down4

        up4 = self.up4(up_, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, in64)

        output = self.final(up1)
        
        # 如果需要，计算TV Loss（作为正则化项）
        if self.use_tv_loss and self.training:
            tv_reg = self.tv_loss(output)
            # 注意：这里只是返回TV损失值，实际使用需要在训练循环中加到总损失中
            return output, tv_reg
        
        return output
    
    def compute_tv_loss(self, x):
        """计算TV损失（外部调用）"""
        return self.tv_loss(x) if self.use_tv_loss else torch.tensor(0.0).to(x.device)



class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetConv2, self).__init__()

        if norm_layer is not None:
            self.conv1 = nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(inplace=True),)  # 添加inplace=True节省内存
            self.conv2 = nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(inplace=True),)
        
        # 添加残差连接稳定训练
        self.residual = (in_size == out_size)
        if not self.residual and in_size != 0:
            self.shortcut = conv(in_size, out_size, 1, bias=need_bias, pad=pad)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        
        # 残差连接
        if self.residual:
            return outputs + inputs
        else:
            return outputs + self.shortcut(inputs)


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetDown, self).__init__()
        self.conv = unetConv2(in_size, out_size, norm_layer, need_bias, pad)
        self.down = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs = self.down(inputs)
        outputs = self.conv(outputs)
        return outputs


class SafeUnetUp(nn.Module):
    """安全的上采样模块，避免棋盘格伪影"""
    def __init__(self, out_size, upsample_mode, need_bias, pad, same_num_filt=False):
        super(SafeUnetUp, self).__init__()

        num_filt = out_size if same_num_filt else out_size * 2
        
        # ========== 关键修改：替换反卷积为安全上采样 ==========
        if upsample_mode == 'deconv':
            # 即使指定deconv，也使用bilinear避免伪影
            print(f"注意: 'deconv'模式已自动转换为'bilinear'以避免棋盘格伪影")
            upsample_mode = 'bilinear'
            
        # 使用插值上采样 + 卷积
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=upsample_mode, 
                       align_corners=False if upsample_mode == 'bilinear' else None),
            conv(num_filt, out_size, 3, bias=need_bias, pad=pad, downsample_mode='stride')
        )
        
        # 特征融合卷积
        self.conv = nn.Sequential(
            conv(out_size * 2, out_size, 3, bias=need_bias, pad=pad),
            nn.ReLU(inplace=True),
            conv(out_size, out_size, 3, bias=need_bias, pad=pad),
            nn.ReLU(inplace=True)
        )
        
        # 添加批归一化稳定训练
        self.norm = nn.BatchNorm2d(out_size)

    def forward(self, inputs1, inputs2):
        # 上采样
        in1_up = self.up(inputs1)
        
        # 尺寸对齐
        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diffY = inputs2.size(2) - in1_up.size(2)
            diffX = inputs2.size(3) - in1_up.size(3)
            
            # 使用更稳定的填充方式
            in1_up = F.pad(in1_up, [
                diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2
            ])
        
        # 拼接特征并卷积
        output = self.conv(torch.cat([in1_up, inputs2], 1))
        output = self.norm(output)
        
        return output


# ========== 向后兼容的包装器 ==========
class FullUNet(UNet):
    """向后兼容的FullUNet类"""
    def __init__(self, n_channels=1, n_classes=1, **kwargs):
        super().__init__(num_input_channels=n_channels, 
                        num_output_channels=n_classes, 
                        **kwargs)

class MediumUNet(UNet):
    """向后兼容的MediumUNet类"""
    def __init__(self, n_channels=1, n_classes=1, **kwargs):
        super().__init__(num_input_channels=n_channels, 
                        num_output_channels=n_classes,
                        feature_scale=2,  # 中等规模
                        **kwargs)

class OneLayerUNet(UNet):
    """向后兼容的OneLayerUNet类"""
    def __init__(self, n_channels=1, n_classes=1, **kwargs):
        super().__init__(num_input_channels=n_channels, 
                        num_output_channels=n_classes,
                        feature_scale=8,  # 更小规模
                        **kwargs)

class ExtraDeepUNet(UNet):
    """向后兼容的ExtraDeepUNet类"""
    def __init__(self, n_channels=1, n_classes=1, **kwargs):
        super().__init__(num_input_channels=n_channels, 
                        num_output_channels=n_classes,
                        more_layers=2,  # 额外两层
                        **kwargs)

class SuperDeepUNet(UNet):
    """向后兼容的SuperDeepUNet类"""
    def __init__(self, n_channels=1, n_classes=1, **kwargs):
        super().__init__(num_input_channels=n_channels, 
                        num_output_channels=n_classes,
                        more_layers=4,  # 更多层
                        **kwargs)


# ========== 辅助函数 ==========
def check_artifact_reduction(model):
    """检查模型是否已修复棋盘格伪影"""
    print("=== 检查棋盘格伪影修复 ===")
    
    # 检查是否有反卷积层
    has_deconv = any(isinstance(m, nn.ConvTranspose2d) for m in model.modules())
    print(f"1. 模型中是否包含反卷积层: {'❌ 存在（有问题）' if has_deconv else '✓ 不存在（正确）'}")
    
    # 检查上采样模块类型
    has_safe_up = any(isinstance(m, SafeUnetUp) for m in model.modules())
    print(f"2. 是否使用安全上采样模块: {'✓ 是' if has_safe_up else '❌ 否'}")
    
    # 测试前向传播
    test_input = torch.randn(1, 1, 128, 128)
    try:
        output = model(test_input)
        if isinstance(output, tuple):
            output = output[0]  # 如果返回了TV损失
        print(f"3. 前向传播测试: {'✓ 通过' if output.shape == (1, 1, 128, 128) else '❌ 失败'}")
        
        # 检查输出范围
        output_min, output_max = output.min().item(), output.max().item()
        print(f"4. 输出范围检查: [{output_min:.3f}, {output_max:.3f}]")
        print(f"   {'✓ 在合理范围内' if 0 <= output_min <= output_max <= 1 else '⚠ 范围异常'}")
    except Exception as e:
        print(f"3. 前向传播测试: ❌ 失败 ({e})")
    
    print("=== 检查完成 ===")
    return not has_deconv and has_safe_up


if __name__ == "__main__":
    # 测试网络
    print("测试标准UNet...")
    model = UNet(num_input_channels=1, num_output_channels=1)
    check_artifact_reduction(model)
    
    print("\n测试FullUNet（向后兼容）...")
    model2 = FullUNet(n_channels=1, n_classes=1)
    check_artifact_reduction(model2)
    
    print("\n所有测试完成！")