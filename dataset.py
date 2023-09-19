from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import InterpolationMode


# 数据加载环节

# 1. 构建数据集，data=‘./data/train’
def build_data_set(dest_image_size, data):
    transform = build_transform(dest_image_size)

    # ImageFolder 会自动的将同一文件夹内的数据打上一个标签，也
    # 就是说 logo 文件夹的数据，ImageFolder 会认为是来自同一类别，
    # others 文件夹的数据，ImageFolder 会认为是来自另外一个类别。
    dataset = datasets.ImageFolder(data, transform=transform, target_transform=None)
    return dataset


def build_transform(dest_image_size):
    normalize = transforms.Lambda(_norm_advprop)
    if not isinstance(dest_image_size, tuple):
        dest_image_size = (dest_image_size, dest_image_size)
    else:
        dest_image_size = dest_image_size

    transform = transforms.Compose([
        transforms.Resize(dest_image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize
    ])

    return transform


def _norm_advprop(img):
    return img * 2.0 - 1.0
