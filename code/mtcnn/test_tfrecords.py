# -*- coding: utf-8 -*-  
""" 
Created on Thu Jan  1 16:09:32 2015 

@author: crw 
"""
# 参数含义：
#  CropFace(image, eye_left, eye_right, offset_pct, dest_sz)
# eye_left is the position of the left eye
# eye_right is the position of the right eye
# 比例的含义为：要保留的图像靠近眼镜的百分比，
# offset_pct is the percent of the image you want to keep next to the eyes (horizontal, vertical direction)
# 最后保留的图像的大小。
# dest_sz is the size of the output image
#
import sys, math
from PIL import Image


# 计算两个坐标的距离
def Distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)

    # 根据参数，求仿射变换矩阵和变换后的图像。


def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)
    # 根据所给的人脸图像，眼睛坐标位置，偏移比例，输出的大小，来进行裁剪。


def CropFace(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.3, 0.3), dest_sz=(96, 112)):
    # calculate offsets in original image 计算在原始图像上的偏移。
    offset_h = math.floor(float(offset_pct[0]) * dest_sz[0])
    offset_v = math.floor(float(offset_pct[1]) * dest_sz[1])
    # get the direction  计算眼睛的方向。
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians  计算旋转的方向弧度。
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them  # 计算两眼之间的距离。
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width    计算最后输出的图像两只眼睛之间的距离。
    reference = dest_sz[0] - 2.0 * offset_h
    # scale factor   # 计算尺度因子。
    scale = float(dist) / float(reference)
    # rotate original around the left eye  # 原图像绕着左眼的坐标旋转。
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    # crop the rotated image  # 剪切
    crop_xy = (eye_left[0] - scale * offset_h, eye_left[1] - 1.4*scale * offset_v)  # 起点
    crop_size = (dest_sz[0] * scale, dest_sz[1] * scale)  # 大小
    image = image.crop(
        (int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0] + crop_size[0]), int(crop_xy[1] + crop_size[1])))
    # resize it 重置大小
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image


def creat_face(points, imagee):
    image = Image.open(imagee)
    leftx = points[0][0]
    lefty = points[0][1]
    rightx = points[0][2]
    righty = points[0][3]

    CropFace(image, eye_left=(leftx, lefty), eye_right=(rightx, righty), offset_pct=(0.3, 0.3),
             dest_sz=(96, 112)).save(imagee)