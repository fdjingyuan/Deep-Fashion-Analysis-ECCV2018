import torch
import torch.utils.data
from src.const import base_path
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage import io, transform
from src import const


def gaussian_map(image_w, image_h, center_x, center_y, R):
    Gauss_map = np.zeros((image_h, image_w))

    mask_x = np.matlib.repmat(center_x, image_h, image_w)
    mask_y = np.matlib.repmat(center_y, image_h, image_w)
    x1 = np.arange(image_w)
    x_map = np.matlib.repmat(x1, image_h, 1)
    y1 = np.arange(image_h)
    y_map = np.matlib.repmat(y1, image_w, 1)
    y_map = np.transpose(y_map)
    Gauss_map = np.sqrt((x_map - mask_x)**2 + (y_map - mask_y)**2)
    Gauss_map = np.exp(-0.5 * Gauss_map / R)
    return Gauss_map


def gen_landmark_map(image_w, image_h, landmark_in_pic, landmark_pos, R):
    ret = []
    # 修改为不在图片里的时候才为0，其他都给出
    for i in range(landmark_in_pic.shape[0]):
        if landmark_in_pic[i] == 0:
            ret.append(np.zeros((image_w, image_h)))
        else:
            channel_map = gaussian_map(image_w, image_h, landmark_pos[i][0], landmark_pos[i][1], R)
            ret.append(channel_map.reshape((image_w, image_h)))
    return np.stack(ret, axis=0).astype(np.float32)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, landmarks):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), mode='constant')

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return img, landmarks


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, landmarks):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return image, landmarks


class CenterCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, landmarks):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = int((h - new_h) / 2)
        left = int((w - new_w) / 2)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return image, landmarks


class RandomFlip(object):

    def __call__(self, image, landmarks):
        h, w = image.shape[:2]
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            landmarks[:, 0] = w - landmarks[:, 0]

        return image, landmarks


class BBoxCrop(object):

    def __call__(self, image, landmarks, x_1, y_1, x_2, y_2):
        h, w = image.shape[:2]

        top = y_1
        left = x_1
        new_h = y_2 - y_1
        new_w = x_2 - x_1

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return image, landmarks


class CheckLandmarks(object):

    def __call__(self, image, landmark_vis, landmark_in_pic, landmark_pos):
        h, w = image.shape[:2]
        landmark_vis = landmark_vis.copy()
        landmark_in_pic = landmark_in_pic.copy()
        landmark_pos = landmark_pos.copy()
        for i, vis in enumerate(landmark_vis):
            if (landmark_pos[i, 0] < 0) or (landmark_pos[i, 0] >= w) or (landmark_pos[i, 1] < 0) or (landmark_pos[i, 1] >= h):
                landmark_vis[i] = 0
                landmark_in_pic[i] = 0
        for i, in_pic in enumerate(landmark_in_pic):
            if in_pic == 0:
                landmark_pos[i, :] = 0
        return landmark_vis, landmark_in_pic, landmark_pos


class LandmarksNormalize(object):

    def __call__(self, image, landmark_pos):
        h, w = image.shape[:2]
        landmark_pos = landmark_pos / [float(w), float(h)]
        return landmark_pos


class LandmarksUnNormalize(object):

    def __call__(self, image, landmark_pos):
        h, w = image.shape[:2]
        landmark_pos = landmark_pos * [float(w), float(h)]
        return landmark_pos


class DeepFashionCAPDataset(torch.utils.data.Dataset):

    def __init__(self, df, mode, base_path=base_path):
        self.df = df
        self.base_path = base_path
        self.to_tensor = transforms.ToTensor()  # pytorch使用c x h x w的格式转换
        self.rescale = Rescale(256)
        self.rescale_largest_center = Rescale(224)
        self.rescale224square = Rescale((224, 224))
        self.bbox_crop = BBoxCrop()
        self.center_crop = CenterCrop(224)
        self.random_crop = RandomCrop(224)
        # self.random_flip = RandomFlip()
        self.check_landmarks = CheckLandmarks()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.landmarks_normalize = LandmarksNormalize()
        self.landmarks_unnormalize = LandmarksUnNormalize()
        self.mode = mode
        assert self.mode in ['RANDOM', 'CENTER', 'LARGESTCENTER', 'BBOXRESIZE']

        # for vis
        self.unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        self.to_pil = transforms.ToPILImage()

    def plot_sample(self, i):
        sample = self[i]
        if isinstance(sample['image'], torch.Tensor):
            image = self.unnormalize(sample['image'])
            image = self.to_pil(image.float())
            image = np.array(image)
        plt.figure(dpi=200)
        plt.imshow(image)
        for i, in_pic in enumerate(sample['landmark_in_pic']):
            if (in_pic == 1):
                plt.scatter([sample['landmark_pos'][i, 0]], [sample['landmark_pos'][i, 1]], s=20, marker='.', c='g')
            else:
                plt.scatter([sample['landmark_pos'][i, 0]], [sample['landmark_pos'][i, 1]], s=20, marker='x', c='r')

    def plot_landmark_map(self, i):
        sample = self[i]
        landmark_map = sample['landmark_map']
        print(landmark_map.shape)
        landmark_map = np.max(landmark_map, axis=0)
        print(landmark_map.shape)
        plt.imshow(landmark_map)

    def __getitem__(self, i):
        sample = self.df.iloc[i]
        image = io.imread(base_path + sample['image_name'])
        category_label = sample['category_label']
        landmark_vis = sample.filter(regex='lm.*vis').astype(np.int64).values
        landmark_in_pic = sample.filter(regex='lm.*in_pic').astype(np.int64).values
        landmark_pos_x = sample.filter(regex='lm.*x').astype(np.int64).values.reshape(-1, 1)
        landmark_pos_y = sample.filter(regex='lm.*y').astype(np.int64).values.reshape(-1, 1)
        landmark_pos = np.concatenate([landmark_pos_x, landmark_pos_y], axis=1)
        attr = sample.filter(regex='attr.*').astype(np.int64).values
        category_type = sample['category_type']

        if self.mode == 'RANDOM':
            image, landmark_pos = self.rescale(image, landmark_pos)
            image, landmark_pos = self.random_crop(image, landmark_pos)
        elif self.mode == 'CENTER':
            image, landmark_pos = self.rescale(image, landmark_pos)
            image, landmark_pos = self.center_crop(image, landmark_pos)
        elif self.mode == 'LARGESTCENTER':
            image, landmark_pos = self.rescale_largest_center(image, landmark_pos)
            image, landmark_pos = self.center_crop(image, landmark_pos)
        elif self.mode == 'BBOXRESIZE':
            image, landmark_pos = self.bbox_crop(image, landmark_pos, sample.x_1, sample.y_1, sample.x_2, sample.y_2)
            image, landmark_pos = self.rescale224square(image, landmark_pos)
        else:
            raise NotImplementedError
        landmark_vis, landmark_in_pic, landmark_pos = self.check_landmarks(image, landmark_vis, landmark_in_pic, landmark_pos)

        landmark_pos = landmark_pos.astype(np.float32)
        landmark_pos_normalized = self.landmarks_normalize(image, landmark_pos).astype(np.float32)

        image = image.copy()

        image = self.to_tensor(image)
        image = self.normalize(image)
        image = image.float()

        ret = {}
        ret['image'] = image
        ret['category_type'] = category_type
        ret['category_label'] = category_label
        ret['landmark_vis'] = landmark_vis
        ret['landmark_in_pic'] = landmark_in_pic
        ret['landmark_pos'] = landmark_pos
        ret['landmark_pos_normalized'] = landmark_pos_normalized
        ret['attr'] = attr
        image_h, image_w = image.size()[1:]
        if hasattr(const, 'gaussian_R'):
            R = const.gaussian_R
        else:
            R = 16
        ret['landmark_map'] = gen_landmark_map(image_w, image_h, landmark_in_pic, landmark_pos, R)
        ret['landmark_map28'] = gen_landmark_map(int(image_w / 8), int(image_h / 8), landmark_in_pic, landmark_pos / 8, R / 8)
        ret['landmark_map56'] = gen_landmark_map(int(image_w / 4), int(image_h / 4), landmark_in_pic, landmark_pos / 4, R / 4)
        ret['landmark_map112'] = gen_landmark_map(int(image_w / 2), int(image_h / 2), landmark_in_pic, landmark_pos / 2, R / 2)
        ret['landmark_map224'] = gen_landmark_map(image_w, image_h, landmark_in_pic, landmark_pos, R)
        return ret


    def __len__(self):
        return len(self.df)
