"""
* https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd
* https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/3
* https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import cv2
import glob
import numpy as np
import random
import torch
from torchvision import transforms


def process_file(npz_filepath):
    video_file = np.load(npz_filepath)

    color_imgs = video_file['colorImages']
    color_imgs = color_imgs.transpose((3, 0, 1, 2))

    bounding_box = video_file['boundingBox']
    bounding_box = bounding_box.transpose((2, 0, 1))
    top_left = bounding_box[:, 0]
    bottom_right = bounding_box[:, 3]
    hw = bottom_right - top_left
    bounding_box = np.concatenate([top_left, hw], 1)

    landmarks2D = video_file['landmarks2D']
    landmarks2D = landmarks2D.transpose((2, 0, 1))
    landmarks3D = video_file['landmarks3D']
    landmarks3D = landmarks3D.transpose((2, 0, 1))

    headPose = video_file['headPose'].transpose()
    gazeVector = video_file['gazeVector'].transpose()

    return color_imgs, bounding_box, landmarks2D, landmarks3D, headPose, gazeVector


def hflip_ldmks(ldmks, img_width):
    pairs = [
        # Jaw
        (0, 16),
        (1, 15),
        (2, 14),
        (3, 13),
        (4, 12),
        (5, 11),
        (6, 10),
        (7, 9),
        # Brows
        (17, 26),
        (18, 25),
        (19, 24),
        (20, 23),
        (21, 22),
        # Eyes
        (36, 45),
        (37, 44),
        (38, 43),
        (39, 42),
        (40, 47),
        (41, 46),
        # Nose
        (31, 35),
        (32, 34),
        # Mouth
        (50, 52),
        (49, 53),
        (48, 54),
        (60, 64),
        (61, 63),
        (59, 55),
        (58, 56),
        (67, 65),
        ]
    # symm_pts = [8, 27, 28, 29, 30, 33, 51, 57, 62, 66]

    ldmks[:, 0] = img_width - ldmks[:, 0]
    for pair in pairs:
        ldmks[pair, :] = ldmks[[pair[1], pair[0]], :]

    return ldmks

# Resize, affine (rot + shift)
# gaussian blur, normalize, noise, invert?
class Resize(torch.nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape

    def forward(self, x):
        new_shape = (0.7 + 0.3 * np.random.rand(1)) * self.output_shape
        new_shape = np.int(new_shape)

        h, w = x['img'].shape[1:]
        if isinstance(new_shape, int):
            if h > w:
                new_h, new_w = new_shape * h / w, new_shape
            else:
                new_h, new_w = new_shape, new_shape * w / h
        else:
            new_h, new_w = new_shape

        new_h, new_w = int(new_h), int(new_w)

        x['img'] = transforms.functional.resize(x['img'], (new_h, new_w))

        pad_total = self.output_shape - new_shape 
        if pad_total > 0:
            x['img'] = transforms.functional.pad(x['img'], (0, 0, pad_total, pad_total))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        x['ldmks2D'] = x['ldmks2D'] * [new_w / w, new_h / h]
        x['ldmks3D'][:, :2] = x['ldmks3D'][:, :2] * [new_w / w, new_h / h]

        return x


class BBoxAugCrop(torch.nn.Module):
    def __init__(self, buffer_max=0.2):
        super().__init__()
        self.buffer_max = buffer_max

    def forward(self, x):
        bbox = x['bbox'].astype(np.int)

        # Add random buffer of [0, 0.2] width and height to bounding box.        
        buffer_pct = np.random.rand(1) * self.buffer_max
        buffer_pct = 0.2
        min_low = np.min(bbox[:2] // 2).astype(np.int)
        min_high = np.min((x['img'].shape[1:] - bbox[2:]) // 2).astype(np.int)
        buf_pix = np.min([min_low, min_high])  # Protect against extending beyond bounds of frame
        buf_pix = np.int(buffer_pct * buf_pix)
        bbox += np.array([-buf_pix, -buf_pix, 2 * buf_pix, 2 * buf_pix])
        x['bbox'] = bbox

        x['img'] = x['img'][:, bbox[1]: bbox[1] + bbox[3],
                               bbox[0]: bbox[0] + bbox[2]]
        x['ldmks2D'][:, :] -= bbox[:2]
        x['ldmks3D'][:, :2] -= bbox[:2]

        return x


class RandomAug(torch.nn.Module):
    def __init__(self, output_shape=(128, 128),
                 prob_flip=0.5, degrees=(-15, 15),
                 prob_gauss=0.5,
                 interpolation=transforms.functional.InterpolationMode.NEAREST):
        self.prob_flip = prob_flip
        self.prob_gauss = prob_gauss
        self.degrees = degrees
        self.interpolation = interpolation

    @staticmethod
    def get_params(degrees, translate):
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())

        if translate is not None:
            tx = int(round(torch.empty(1).uniform_(-translate[0], translate[0]).item()))
            ty = int(round(torch.empty(1).uniform_(-translate[1], translate[1]).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        scale = 1.0
        shear = (0.0, 0.0)

        return angle, translations, scale, shear

    def __call__(self, x):
        # Apply horizontal flip.
        hflip_bool = torch.rand(1) < self.prob_flip
        if hflip_bool:
            x['img'] = transforms.functional.hflip(x['img'])
            _, h, w = x['img'].shape
            x['ldmks2D'] = hflip_ldmks(x['ldmks2D'], w)
            x['ldmks3D'][:, :2] = hflip_ldmks(x['ldmks3D'][:, :2], w)
            x['headPose'][0] *= -1
            x['headPose'][2] *= -1
            x['gaze'][0] *= -1

        # Apply affine transforms.
        fill = 0
        if isinstance(x['img'], torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * transforms.functional._get_image_num_channels(x['img'])
            else:
                fill = [float(f) for f in fill]
        x_tsl, y_tsl = x['img'].shape[1:]
        translate = (x_tsl // 8, y_tsl // 8)
        params = self.get_params(self.degrees, translate)
        x['img'] = transforms.functional.affine(x['img'], *params, interpolation=self.interpolation, fill=fill)

        # Rotate and shift.
        cen_x, cen_y = x['img'].shape[1:]
        cen_x, cen_y = cen_x // 2, cen_y // 2
        center = np.array([cen_x, cen_y])
        shift = params[1]
        angle = params[0]
        angle_rad = angle * np.pi / 180
        rot_mat = np.array([
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)],
            ])
        ldmk_x = x['ldmks2D'][:, 0]
        ldmk_y = x['ldmks2D'][:, 1]
        x['ldmks2D'] = np.dot(x['ldmks2D'] - center, rot_mat) + center + shift
        x['ldmks3D'][:, :2] = np.dot(x['ldmks3D'][:, :2] - center, rot_mat) + center + shift
        x['headPose'][2] += angle
        x['gaze'][:2] = np.dot(x['gaze'][:2], rot_mat)

        return x


def render(sample):
    img = sample['img']
    # print(img)
    img = (img * 255.).detach().numpy().astype(np.uint8).transpose((1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    ldmks2D = sample['ldmks2D']
    for idx, ldmk in enumerate(ldmks2D):
        ldmk = [int(i) for i in ldmk]
        if idx not in [8, 27, 28, 29, 30, 33, 51, 57, 62, 66]:
            color = (255, 0, 255)
        else:
            color = (0, 255, 255)
        img = cv2.circle(img, (ldmk[0], ldmk[1]), 1, color, -1)

    size = img.shape
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    headpose = sample['headPose']
    yaw = headpose[0] * np.pi / 180.
    pitch = headpose[1] * np.pi / 180.
    roll = headpose[2] * np.pi / 180.

    sinY = np.sin(yaw)
    sinP = np.sin(pitch)
    sinR = np.sin(roll)

    cosY = np.cos(yaw)
    cosP = np.cos(pitch)
    cosR = np.cos(roll)

    axis_length = 0.4 * img.shape[1]

    nose = tuple(int(x) for x in ldmks2D[30, :])
    # nose = (img.shape[0] // 2, img.shape[1] // 2)
    ept = nose + np.array([axis_length * (cosR * cosY + sinY * sinP * sinR), axis_length * cosP * sinR])
    ept = tuple(int(x) for x in ept)
    cv2.line(img, nose, ept, (0,255,0), 2) #GREEN

    ept = nose + np.array([axis_length * (cosR * sinY * sinP + cosY * sinR), -axis_length * cosP * cosR])
    ept = tuple(int(x) for x in ept)
    cv2.line(img, nose, ept, (255,0,0), 2) #BLUE

    ept = nose + np.array([axis_length * sinY * cosP, axis_length * sinP])
    ept = tuple(int(x) for x in ept)
    cv2.line(img, nose, ept, (0,0,255), 2) #RED

    gaze = sample['gaze']
    eyes = (ldmks2D[36, :] + ldmks2D[45, :]) / 2
    eyes = tuple(int(x) for x in eyes)
    gaze_arrow = np.array([int(axis_length * gaze[0]), -int(axis_length * gaze[1])])
    gaze_arrow += eyes
    cv2.arrowedLine(img, eyes, tuple(gaze_arrow), (255, 255, 0), 1)

    cv2.imwrite('test.png', img)


class FaceIterDataset(torch.utils.data.IterableDataset):
    def __init__(self, paths, img_shape=(128, 128),
                 img_dim=[], shuffle=True,
                 num_workers=4, pin_memory=True):
        self.filepaths = paths
        self.img_shape = img_shape
        self.preaug_tsfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(),
            transforms.Grayscale(),
            # transforms.Normalize([127], [255]),
            ])
        self.aug_tsfms = transforms.Compose([
            BBoxAugCrop(0.2),
            RandomAug(self.img_shape, 0.5),
            Resize(128),
            ])
        self.keys = ['img', 'bbox', 'ldmks2D', 'ldmks3D', 'headPose', 'gaze']

    def tsfm(self, data):
        data['img'] = self.preaug_tsfms(data['img'])
        data = self.aug_tsfms(data)
        return data

    def __iter__(self):
        for filepath in self.filepaths:
            print(filepath)
            data = process_file(filepath)
            for sample in zip(*data):
                sample = {k: v for k, v in zip(self.keys, sample)}
                sample = self.tsfm(sample)
                # render(sample)
                yield sample


class MultiStreamDataLoader():
    def __init__(self, datasets):
        self.datasets = datasets
        self.batch_size = batch_size

    def get_stream_loaders(self):
        return zip(*[torch.utils.data.DataLoader(dataset, num_workers=1)
                   for dataset in self.datasets])

    def __iter__(self):
        for batches in self.get_stream_loaders():
            yield list(chain(*batch_parts))


def main():
    # filepaths = ['/mnt/f/face_data/joint_task_data/youtube_faces_with_keypoints_full_1/Aaron_Eckhart_0.npz',
    #              '/mnt/f/face_data/joint_task_data/youtube_faces_with_keypoints_full_1/Aaron_Eckhart_1.npz',
    #              '/mnt/f/face_data/joint_task_data/youtube_faces_with_keypoints_full_1/Abba_Eban_0.npz']
    filepaths = glob.glob('/mnt/f/face_data/joint_task_data/*/*.npz')

    fd = FaceIterDataset(filepaths)
    print(fd)
    dataset_loader = torch.utils.data.DataLoader(dataset=fd,
                                                 batch_size=16,
                                                 num_workers=1,
                                                 shuffle=False)
    for i, sample in enumerate(dataset_loader):
        print(sample['img'].shape)
        # print(item)
        # print([(data[0], data[1].shape) for data in sample.items()])
        # if i == 1:
        #     break
    # for i, batch in enumerate(MultiStreamDataLoader()):



if __name__ == '__main__':
    main()