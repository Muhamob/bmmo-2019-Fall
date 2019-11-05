import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ImageOps import grayscale


def generate_images(fg, bg, num=10, normalize=True):
    h, w = fg.shape
    H, W = bg.shape
    assert h < H and w < W, "Foreground has to be less than background"

    imgs = np.empty((H, W, num))
    i_s = np.random.randint(0, H-h+1, size=num)
    j_s = np.random.randint(0, W-w+1, size=num)

    for img_num, (i, j) in enumerate(zip(i_s, j_s)):
        img = np.copy(bg)
        img[i:i+h, j:j+w] = np.where(fg>10, fg, img[i:i+h, j:j+w])
        imgs[..., img_num] = img

    return np.asarray(imgs) / 255.


def generate_imgs_from_src(fg_path, bg_path, num=10):
    fg = grayscale(Image.open(fg_path))
    fg = fg.resize((32, 32))
    fg.load()
    fg = np.asarray(fg, dtype="int32")

    bg = grayscale(Image.open(bg_path))
    bg.load()
    bg = np.asarray(bg, dtype="int32")

    return generate_images(fg, bg, num)


def generate_imgs_from_src_and_noise(fg_path, num=10):
    fg = Image.open(fg_path).convert('LA')
    fg = fg.resize((64, 64))
    fg.load()
    fg = np.asarray(fg, dtype="int32")


def add_noise(img, s):
    return np.clip(img + np.random.normal(0, s, img.shape), 0, 255)