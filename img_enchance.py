#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/28 15:09
# @Author  : apple233
import random
from PIL import Image,ImageEnhance,ImageFilter
import numpy as np
import cv2

#添加背景图片
def src_blend_background(img_src,base_img):
    """
    Args:
        img_src: PIL
        base_img: PIL 背景图
    Returns:
    """
    base_img = base_img.convert('RGBA')
    img_src = img_src.convert('RGBA')
    num = random.randint(1, 3)
    nhc = ImageEnhance.Color(base_img)
    nhb = ImageEnhance.Brightness(base_img)

    enh_con = ImageEnhance.Contrast(base_img)
    ratio1 = random.uniform(1, 4)  # 减弱和增强两个系数
    ratio2 = random.uniform(1, 3)

    base_img = enh_con.enhance(ratio2)
    base_img=base_img.resize(img_src.size)
    # img_copy = Image.blend(img_src, base_img.resize(img_src.size), 0.1)  # uniform(0.4,0.5)
    base_img.resize(img_src.size)
    img_src=cv2.cvtColor(np.asarray(img_src), cv2.COLOR_RGBA2BGRA)
    base_img=cv2.cvtColor(np.asarray(base_img), cv2.COLOR_RGBA2BGRA)

    img_copy = cv2.addWeighted(img_src, 0.7, base_img,0.3, -50)
    # img_copy=2*img_src
    # img_copy=np.array(img_copy).astype(np.uint8)
    # base_img=base_img*0.4
    # img_copy = img_src+base_img
    img_copy=Image.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGRA2RGBA))

    return img_copy


#有损压缩达到模糊效果
def compress(img):
    """

    Args:
        img: PIL

    Returns:

    """
    #pil2cv
    img=cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGR)
    # param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(8,30)]
    param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(8,20)]
    img_encode = cv2.imencode('.jpeg', img, param)
    img_decode = cv2.imdecode(img_encode[1], cv2.IMREAD_COLOR)
    # cv2pil
    imgarr = Image.fromarray(np.uint8(img_decode))
    return imgarr

#高斯模糊
def gaussian_blur(img):
    """
    
    Args:
        img: PIL

    Returns:

    """
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGR)
    sigmaX=0.1
    img = cv2.GaussianBlur(img, (0, 0), sigmaX=sigmaX)
    imgarr = Image.fromarray(np.uint8(img))

    return imgarr



#先放大后缩小 模拟花屏
def blurred_aug(img_src):
    """

    Args:
        img_src: PIL

    Returns:

    """
    w,h=img_src.size
    # ratio=random.randint(2,10)
    ratio=random.choice([0.35,0.4,0.45,0.5,0.6])
    h_reisze,w_resize=h*ratio,w*ratio
    h_reisze=int(h_reisze)
    w_resize=int(w_resize)
    img_src=img_src.resize((w_resize,h_reisze)) #缩小
    img_src=img_src.resize((w,h),Image.NEAREST)#放大

    return img_src

#模拟印泥缺失
def texture_aug(img,img_texture):
    """

    Args:
        img_src: PIL
        texture: PIL

    Returns:

    """
    w,h=img.size
    pos_random = (random.randint(0, 200), random.randint(0, 100))
    box = (pos_random[0], pos_random[1], pos_random[0] + 300, pos_random[1] + 300)
    img_wl_random = img_texture  # .crop(box).rotate(randint(0, 360))
    # 重新设置im2的大小，并进行一次高斯模糊
    # img_wl_random = img_wl_random.resize(img.size, Image.Resampling.BICUBIC)
    img_wl_random = img_wl_random.resize(img.size)
    img_wl_random = img_wl_random.resize(img.size).convert('L').filter(ImageFilter.GaussianBlur(1))
    # 将纹理图的灰度映射到原图的透明度，由于纹理图片自带灰度，映射后会有透明效果，所以fill的透明度不能太低

    img_s = img.copy()
    imgarr = np.array(img_s)
    img_wl_random_imgarr = np.array(img_wl_random)

    imgarr[:, :, 3] = (img_wl_random_imgarr / 255 * imgarr[:, :, 3]).astype(int)
    imgarr = Image.fromarray(np.uint8(imgarr))

    img = imgarr.filter(ImageFilter.GaussianBlur(0.6))
    return img


#边界扩充
def boundary_expand(img):
    """

    Args:
        img: PIL

    Returns:PIL

    """
    # pil2cv
    w,h=img.size
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGRA)
    #常数填充：
    # img = cv2.copyMakeBorder(img, random.randint(5,h//4),  random.randint(5,h//4), random.randint(5,h//4), random.randint(5,h//4), cv2.BORDER_CONSTANT,None, value=(0,0,0))
    img = cv2.copyMakeBorder(img, random.randint(5,h//4),  random.randint(5,h//4), random.randint(5,h//4), random.randint(5,h//4), cv2.BORDER_REPLICATE)
    # cv2pil
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    return img

#随机resize
def resize_aug(img):
    """

    Args:
        img: PIL

    Returns:PIL

    """
    #
    w,h=img.size
    ratio=random.choice([0.5,0.6,0.7,1.1,1.2])
    h_reisze, w_resize = h * ratio, w * ratio
    img = img.resize((int(w_resize), int(h_reisze)), Image.BILINEAR)
    return img

