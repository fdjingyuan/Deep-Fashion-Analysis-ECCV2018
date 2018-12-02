import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torchvision
from src import const


class CustomUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, last_act='sigmoid'):
        super(CustomUnetGenerator, self).__init__()

        # construct unet structure
        innermost_nc = 2 ** num_downs
        unet_block = UnetSkipConnectionBlock(ngf * innermost_nc, ngf * innermost_nc, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, keep_size=True)
        for i in range(num_downs):
            k = num_downs - i
            unet_block = UnetSkipConnectionBlock(ngf * (2 ** (k - 1)), ngf * (2 ** k), input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, last_act=last_act, keep_size=True)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, last_act='sigmoid', keep_size=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        if keep_size:
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=1,
                             stride=1, padding=0, bias=use_bias)
        else:
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                 stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            if keep_size:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=1, stride=1,
                                            padding=0)
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
            down = [downconv]
            if last_act == 'tanh':
                up = [uprelu, upconv, nn.Tanh()]
            elif last_act == 'sigmoid':
                up = [uprelu, upconv, nn.Sigmoid()]
            else:
                raise NotImplementedError
            model = down + [submodule] + up
        elif innermost:
            if keep_size:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=1, stride=1,
                                            padding=0)
            else:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            if keep_size:
                raise Exception("can not keep size")
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
            

class ModuleWithAttr(nn.Module):

    # 只能是数字，默认注册为0

    def __init__(self, extra_info=['step']):
        super(ModuleWithAttr, self).__init__()
        for key in extra_info:
            self.set_buffer(key, 0)

    def set_buffer(self, key, value):
        if not(hasattr(self, '__' + key)):
            self.register_buffer('__' + key, torch.tensor(value))
        setattr(self, '__' + key, torch.tensor(value))

    def get_buffer(self, key):
        if not(hasattr(self, '__' + key)):
            raise Exception('no such key!')
        return getattr(self, '__' + key).item()

class BaseLoss(ModuleWithAttr):

    def __init__(self):
        super(BaseLoss, self).__init__()
        self.category_loss_func = torch.nn.CrossEntropyLoss()
        self.attr_loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([const.WEIGHT_ATTR_NEG, const.WEIGHT_ATTR_POS]).to(const.device))
        self.lm_vis_loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([const.WEIGHT_LANDMARK_VIS_NEG, const.WEIGHT_LANDMARK_VIS_POS]).to(const.device))
        self.lm_pos_loss_func = torch.nn.MSELoss()

    def cal_loss(self, sample, output):
        category_loss = self.category_loss_func(output['category_output'], sample['category_label'])
        # 所有的都是2类，每个2类有1000个，都分别计算交叉熵，计算之后的交叉熵加入weight，之后再全部取平均，与我们的期望符合
        attr_loss = self.attr_loss_func(output['attr_output'], sample['attr'])
        lm_vis_loss = self.lm_vis_loss_func(output['lm_vis_output'], sample['landmark_vis'])
        landmark_vis_float = torch.unsqueeze(sample['landmark_vis'].float(), dim=2)
        landmark_vis_float = torch.cat([landmark_vis_float, landmark_vis_float], dim=2)  # 用真实值当mask，只计算vis=1时的损失
        lm_pos_loss = self.lm_pos_loss_func(
            landmark_vis_float * output['lm_pos_output'],
            landmark_vis_float * sample['landmark_pos_normalized']
        )
        all_loss = const.WEIGHT_LOSS_CATEGORY * category_loss + \
            const.WEIGHT_LOSS_ATTR * attr_loss + \
            const.WEIGHT_LOSS_LM_VIS * lm_vis_loss + \
            const.WEIGHT_LOSS_LM_POS * lm_pos_loss

        loss = {
            'all': all_loss,
            'category_loss': category_loss.item(),
            'attr_loss': attr_loss.item(),
            'lm_vis_loss': lm_vis_loss.item(),
            'lm_pos_loss': lm_pos_loss.item(),
            'weighted_category_loss': const.WEIGHT_LOSS_CATEGORY * category_loss.item(),
            'weighted_attr_loss': const.WEIGHT_LOSS_ATTR * attr_loss.item(),
            'weighted_lm_vis_loss': const.WEIGHT_LOSS_LM_VIS * lm_vis_loss.item(),
            'weighted_lm_pos_loss': const.WEIGHT_LOSS_LM_POS * lm_pos_loss.item(),
        }
        return loss


class VGG16Extractor(nn.Module):
    def __init__(self):
        super(VGG16Extractor, self).__init__()
        self.select = {
            '1': 'conv1_1',  # [batch_size, 64, 224, 224]
            '3': 'conv1_2',  # [batch_size, 64, 224, 224]
            '4': 'pooled_1',  # [batch_size, 64, 112, 112]
            '6': 'conv2_1',  # [batch_size, 128, 112, 112]
            '8': 'conv2_2',  # [batch_size, 128, 112, 112]
            '9': 'pooled_2',  # [batch_size, 128, 56, 56]
            '11': 'conv3_1',  # [batch_size, 256, 56, 56]
            '13': 'conv3_2',  # [batch_size, 256, 56, 56]
            '15': 'conv3_3',  # [batch_size, 256, 56, 56]
            '16': 'pooled_3',  # [batch_size, 256, 28, 28]
            '18': 'conv4_1',  # [batch_size, 512, 28, 28]
            '20': 'conv4_2',  # [batch_size, 512, 28, 28]
            '22': 'conv4_3',  # [batch_size, 512, 28, 28]
            '23': 'pooled_4',  # [batch_size, 512, 14, 14]
            '25': 'conv5_1',  # [batch_size, 512, 14, 14]
            '27': 'conv5_2',  # [batch_size, 512, 14, 14]
            '29': 'conv5_3',  # [batch_size, 512, 14, 14]
            '30': 'pooled_5',  # [batch_size , 512, 7, 7]
        }
        self.vgg = torchvision.models.vgg16(pretrained=True).features

    def forward(self, x):
        ret = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                ret[self.select[name]] = x
#                 print(self.select[name], x.size())
#                 print(name, layer)
        return ret

