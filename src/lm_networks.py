import torch
from torch import nn
from torch.nn import functional as F
from src import const
from src.base_networks import ModuleWithAttr, VGG16Extractor
import numpy as np


class LandmarkBranchUpsample(nn.Module):

    def __init__(self, in_channel=256):
        super(LandmarkBranchUpsample, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, 1, 1, 0)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.conv7 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv8 = nn.Conv2d(32, 32, 3, 1, 1)
        self.upconv3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.conv9 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv10 = nn.Conv2d(16, 8, 1, 1, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.upconv1(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.conv9(x))
        x = self.conv10(x)
        # lm_pos_map = F.sigmoid(x)
        lm_pos_map = x
        batch_size, _, pred_h, pred_w = lm_pos_map.size()
        lm_pos_reshaped = lm_pos_map.reshape(batch_size, 8, -1)
        # y是高上的坐标，x是宽上的坐标
        lm_pos_y, lm_pos_x = np.unravel_index(torch.argmax(lm_pos_reshaped, dim=2), (pred_h, pred_w))
        lm_pos_output = np.stack([lm_pos_x / (pred_w - 1), lm_pos_y / (pred_h - 1)], axis=2)

        return lm_pos_map, lm_pos_output


class LandmarkExpNetwork(ModuleWithAttr):

    def __init__(self):
        super(LandmarkExpNetwork, self).__init__()
        self.vgg16_extractor = VGG16Extractor()
        self.lm_branch = const.LM_BRANCH(const.LM_SELECT_VGG_CHANNEL)

    def forward(self, sample):
        batch_size, channel_num, image_h, image_w = sample['image'].size()
        vgg16_output = self.vgg16_extractor(sample['image'])
        vgg16_for_lm = vgg16_output[const.LM_SELECT_VGG]
        lm_pos_map, lm_pos_output = self.lm_branch(vgg16_for_lm)
        return {
            'lm_pos_output': lm_pos_output,
            'lm_pos_map': lm_pos_map,
        }

    def cal_loss(self, sample, output):
        batch_size, _, _, _ = sample['image'].size()

        lm_size = int(output['lm_pos_map'].shape[2])
        if hasattr(const, 'LM_TRAIN_USE') and const.LM_TRAIN_USE == 'in_pic':
            mask = sample['landmark_in_pic'].reshape(batch_size * 8, -1)
        else:
            mask = sample['landmark_vis'].reshape(batch_size * 8, -1)
        mask = torch.cat([mask] * lm_size * lm_size, dim=1).float()
        map_sample = sample['landmark_map%d' % lm_size].reshape(batch_size * 8, -1)
        map_output = output['lm_pos_map'].reshape(batch_size * 8, -1)
        lm_pos_loss = torch.pow(mask * (map_output - map_sample), 2).mean()

        all_loss = \
            const.WEIGHT_LOSS_LM_POS * lm_pos_loss
        loss = {
            'all': all_loss,
            'lm_pos_loss': lm_pos_loss.item(),
            'weighted_lm_pos_loss': const.WEIGHT_LOSS_LM_POS * lm_pos_loss.item(),
        }

        return loss
