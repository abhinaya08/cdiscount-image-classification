import cv2
import torch
import numpy as np
import random
import math
from sklearn.preprocessing import MinMaxScaler


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', 'jpg', '.jpeg'])


def calc_ndwi(image):
    """
    calculate normalized difference water index
    input image is of the format(NIR, R, G)
    """
    return (image[:, :, 2] - image[:, :, 0]) / (image[:, :, 2] + image[:, :, 0] + 1e-8)


def scale(img):
    rescaleimg = np.reshape(img, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 255))
    rescaleimg = scaler.fit_transform(rescaleimg)  # .astype(np.float32)
    img_scaled = (np.reshape(rescaleimg, img.shape))
    return img_scaled


def image_to_tensor(image):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    tensor = torch.from_numpy(image).float().div(255)

    tensor[0] = (tensor[0] - mean[0]) / std[0]
    tensor[1] = (tensor[1] - mean[1]) / std[1]
    tensor[2] = (tensor[2] - mean[2]) / std[2]
    return tensor


def tensor_to_image(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    tensor[0] = tensor[0]*std[0] + mean[0]
    tensor[1] = tensor[1]*std[1] + mean[1]
    tensor[2] = tensor[2]*std[2] + mean[2]

    image = tensor.numpy()*255
    image = np.transpose(image, (1, 2, 0))
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def fix_multi_crop(image, roi_size=(160, 160)):
    height, width = image.shape[0:2]
    h, w = roi_size
    dy = height - h
    dx = width - w

    images = []
    rois = [(dx//2, dy//2, width-dx//2, height-dy//2),
            (0, 0, w, h), (dx, 0, width, h),
            (0, dy, w, height), (dx, dy, width, height), ]
    image = cv2.flip(image, 1)

    for roi in rois:
        x0, y0, x1, y1 = roi
        i = np.ascontiguousarray(image[y0:y1, x0:x1, :])
        images.append(i)
    i = cv2.resize(image, roi_size)
    images.append(i)
    return images


def random_resize(image, scale_x_limits=(0.9, 1.1), scale_y_limits=(0.9, 1.1), u=0.5):
    if random.random() < u:
        height, width = image.shape[0:2]

        scale_x = random.uniform(scale_x_limits[0], scale_x_limits[1])
        if scale_y_limits is not None:
            scale_y = random.uniform(scale_y_limits[0], scale_y_limits[1])
        else:
            scale_y = scale_x

        w = int(scale_x * width)
        h = int(scale_y * height)

        image = cv2.resize(image, (w, h))
    return image


def random_crop(image, size=(160, 160), u=0.5):
    height, width = image.shape[0:2]
    w, h = size

    if random.random() < u:
        x0 = np.random.choice(width - w)
        y0 = np.random.choice(height - h)
    else:
        x0 = (width - w) // 2
        y0 = (height - h) // 2

    x1 = x0 + w
    y1 = y0 + h
    image = image[y0:y1, x0:x1]
    return image


def fix_center_crop(image, size=(160, 160)):
    height, width = image.shape[0:2]
    w, h = size

    x0 = (width - w) // 2
    y0 = (height - h) // 2
    x1 = x0 + w
    y1 = y0 + h
    image = image[y0:y1, x0:x1]
    return image


def random_crop_scale(image, scale_limit=(1/1.2, 1.2), size=[-1, -1], u=0.5):
    if random.random() < u:
        image = image.copy()

        height, width, channel = image.shape
        sw, sh = size
        if sw == -1:
            sw = width
        if sh == -1:
            sh = height
        box0 = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh], ])

        scale = random.uniform(scale_limit[0], scale_limit[1])
        w = int(scale * sw)
        h = int(scale * sh)

        if w > width and h > height:
            x0 = random.randint(width - w, 0)
            y0 = random.randint(height - h, 0)
            x1 = x0 + w
            y1 = y0 + h
        elif w < width and h < height:
            x0 = random.randint(0, width - w)
            y0 = random.randint(0, height - h)
            x1 = x0 + w
            y1 = y0 + h
        elif w == width and h == height:
            return image
        else:
            print(w, h)
            raise NotImplementedError

        box1 = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], ])
        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box1,box0)
        image = cv2.warpPerspective(image, mat, (sw, sh),flags=cv2.INTER_LINEAR,  # cv2.BORDER_REFLECT_101
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0,))
        # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  # cv2.BORDER_REFLECT_10
    return image


def random_shift_scale_rotate(image, shift_limit=(-0.0625, 0.0625), scale_limit=(1/1.2, 1.2),
                              rotate_limit=(-15, 15), aspect_limit=(1, 1),  size=[-1,-1],
                              borderMode=cv2.BORDER_REFLECT_101, u=0.5):
    if random.random() < u:
        height, width, channel = image.shape
        if size[0] == -1:
            size[0] = width
        if size[1] == -1:
            size[1] = height

        angle = random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = random.uniform(scale_limit[0], scale_limit[1])
        aspect = random.uniform(aspect_limit[0], aspect_limit[1])
        sx = scale * aspect / (aspect**0.5)
        sy = scale / (aspect**0.5)
        dx = round(random.uniform(shift_limit[0], shift_limit[1])*width)
        dy = round(random.uniform(shift_limit[0], shift_limit[1])*height)

        cc = math.cos(angle/180*math.pi)*sx
        ss = math.sin(angle/180*math.pi)*sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width/2, height/2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width/2+dx, height/2+dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        image = cv2.warpPerspective(image, mat, (size[0], size[1]),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode,
                                    borderValue=(0, 0, 0, ))
    return image


def fix_crop(image, roi=(0, 0, 256, 256)):
    x0, y0, x1, y1 = roi
    image = image[y0:y1, x0:x1, :]
    return image


def fix_resize(image, w, h):
    image = cv2.resize(image, (w, h))
    return image


def random_horizontal_flip(image, u=0.5):
    if random.random() < u:
        image = cv2.flip(image, 1)
    return image


def train_augment(image):  # used for training
    image = fix_resize(image, 224, 224)
    # image = random_resize(image, scale_x_limits=[0.9, 1.1], scale_y_limits=[0.9, 1.1], u=0.5)
    # image = random_crop(image, size=(160, 160), u=0.5)
    image = random_shift_scale_rotate(image)
    image = random_horizontal_flip(image, u=0.5)
    tensor = image_to_tensor(image)
    return tensor


def valid_augment(image):  # used for validation
    # image = fix_center_crop(image, size=(160, 160))
    tensor = image_to_tensor(image)
    return tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread('data/lufei.jpg')

    plt.figure(0)
    plt.title("Image")
    plt.imshow(random_shift_scale_rotate(img))
    plt.show()