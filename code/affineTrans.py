import cv2, numpy
import face_detection

imgSize = [112, 112]

normalizationPoint = [[0.31556875000000000, 0.4615741071428571],
                      [0.68262291666666670, 0.4615741071428571],
                      [0.50026249999999990, 0.6405053571428571],
                      [0.34947187500000004, 0.8246919642857142],
                      [0.65343645833333330, 0.8246919642857142]]

coord5point = numpy.array(normalizationPoint) * 112

# 最终的人脸对齐图像尺寸分为两种：112x96和112x112，并分别对应结果图像中的两组仿射变换目标点,如下所示
imgSize1 = [112, 96]
imgSize2 = [112, 112]

coord5point1 = [[30.2946, 51.6963],  # 112x96图像的目标点
                [65.5318, 51.6963],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.3655]]

coord5point2 = [[30.2946 + 8.0000, 51.6963],  # 112x112图像的目标点,在112*96标准点基础上所有x坐标向右平移8px
                [65.5318 + 8.0000, 51.6963],
                [48.0252 + 8.0000, 71.7366],
                [33.5493 + 8.0000, 92.3655],
                [62.7299 + 8.0000, 92.3655]]


def transformation_from_points(points1, points2):
    # 变量类型转换
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    # mean()函数功能：求取均值
    # 经常操作的参数为axis，以m * n矩阵举例：
    #
    # axis 不设置值，对 m*n 个数求均值，返回一个实数
    # axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
    # axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    # print(c1)
    points2 -= c2
    # 计算标准差
    # axis=0计算每一列的标准差
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    # numpy.linalg模块包含线性代数的函数，可以计算逆矩阵、求特征值、解线性方程组以及求解行列式等
    # svd为奇异值分解
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), numpy.matrix([0., 0., 1.])])


def warp_im(img_im, orgi_landmarks, tar_landmarks):
    pts1 = numpy.float64(numpy.mat([[point[0], point[1]] for point in orgi_landmarks]))
    # print(pts1.shape)
    pts2 = numpy.float64(numpy.mat([[point[0], point[1]] for point in tar_landmarks]))
    # 求仿射变换矩阵（2行3列）
    M = transformation_from_points(pts1, pts2)
    # 第三个参数为变换后的图像大小（采用二元元祖（宽，高）），第二个参数为变换矩阵
    dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
    # print(M)
    # print(M[:2])
    return dst


def get_cropImage(face_landmarks, img_path="images/face_test.jpg", mode=1):
    pic_path = img_path
    img_im = cv2.imread(pic_path)
    coord5 = coord5point
    img_size = imgSize
    # 获取112*96图片
    if mode == 2:
        coord5 = coord5point1
        img_size = imgSize1
    # 仿射变换
    dst = warp_im(img_im, face_landmarks, coord5)
    # 截取112*112
    crop_im = dst[0:img_size[0], 0:img_size[1]]
    return crop_im


if __name__ == '__main__':
    path = './images/person7.jpg'
    img = cv2.imread(path)
    features_points, rects = face_detection.get_landmarkAndrect(path)
    crop_img = get_cropImage(features_points[3], path)
    cv2.imshow("img1", crop_img)
    cv2.imshow('1', img)
    # cv2.imwrite('./images/affine.jpg', crop_img)
    cv2.waitKey(0)
