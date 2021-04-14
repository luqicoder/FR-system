import time
import tensorflow as tf
import cv2
import numpy as np
from mtcnn.mtcnn import PNet, RNet, ONet
from mtcnn.tools import detect_face, get_model_filenames

# 设置利用cpu
# import os
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class face_recognition:
    # 静态属性
    model_path = './models/all_in_one/'
    is_first = True
    p1 = 'null'
    p2 = 'null'
    p3 = 'null'

    images = 'null'
    dropout_rate = 'null'
    output_tensor = 'null'
    sess_recog = 'null'

    def __init__(self):
        self.imgSize = [112, 112]
        self.normalizationPoint = [[0.31556875000000000, 0.4615741071428571],
                                   [0.68262291666666670, 0.4615741071428571],
                                   [0.50026249999999990, 0.6405053571428571],
                                   [0.34947187500000004, 0.8246919642857142],
                                   [0.65343645833333330, 0.8246919642857142]]
        self.coord5point = np.array(self.normalizationPoint) * 112

        # 初始化
        face_recognition.init_session()

    # 获取5个特征点和人脸检测框
    def get_landmarkAndrect(self, image_path):
        minsize = 20
        threshold = [0.8, 0.8, 0.8]
        factor = 0.7
        img = cv2.imread(image_path)
        start_time = time.time()
        rectangles, points = detect_face(img, minsize,
                                         face_recognition.p1, face_recognition.p2, face_recognition.p3,
                                         threshold, factor)
        duration = time.time() - start_time
        print("检测时间：", duration)
        points = np.transpose(points)
        num = points.shape[0]
        if num == 0:
            return [], []

        split_points = np.vsplit(points, num)
        k = 0
        for i in split_points:
            split_points[k] = i.reshape(5, 2)
            k += 1
        return split_points, rectangles

    def transformation_from_points(self, points1, points2):
        # 变量类型转换
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)
        # mean()函数功能：求取均值
        # 经常操作的参数为axis，以m * n矩阵举例：
        #
        # axis 不设置值，对 m*n 个数求均值，返回一个实数
        # axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
        # axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        # print(c1)
        points2 -= c2
        # 计算标准差
        # axis=0计算每一列的标准差
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2
        # numpy.linalg模块包含线性代数的函数，可以计算逆矩阵、求特征值、解线性方程组以及求解行列式等
        # svd为奇异值分解
        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T
        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

    def warp_im(self, img_im, orgi_landmarks, tar_landmarks):
        pts1 = np.float64(np.mat([[point[0], point[1]] for point in orgi_landmarks]))
        # print(pts1.shape)
        pts2 = np.float64(np.mat([[point[0], point[1]] for point in tar_landmarks]))
        # 求仿射变换矩阵（2行3列）
        M = self.transformation_from_points(pts1, pts2)
        # 第三个参数为变换后的图像大小（采用二元元祖（宽，高）），第二个参数为变换矩阵
        dst = cv2.warpAffine(img_im, M[:2], (img_im.shape[1], img_im.shape[0]))
        # print(M)
        # print(M[:2])
        return dst

    # 获取仿射变换后的图片
    def get_cropImage(self, face_landmarks, img_path, mode=1):
        pic_path = img_path
        img_im = cv2.imread(pic_path)
        # 仿射变换
        dst = self.warp_im(img_im, face_landmarks, self.coord5point)
        # 截取112*112
        crop_im = dst[0:self.imgSize[0], 0:self.imgSize[1]]
        return crop_im

    # 获取size为112*112的512维特征向量
    def get_512features(self, img_path):
        img = img_path
        if type(img_path) == str:
            img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
            img = np.array(img, dtype=np.float32)
            # 图片尺寸更改为112 * 112
            if img.shape[0] != 112 or img.shape[1] != 112:
                print("需要更改图片尺寸")
                img = cv2.resize(img, (112, 112))
        else:
            if img.shape == (112, 96, 3):
                z = np.zeros((112, 8, 3), dtype=np.float32)
                img = np.append(img, z, axis=1)
                img = np.append(z, img, axis=1)
                # img = np.array(img, dtype=np.uint8)
                # cv.imshow("img",img)
                # cv.waitKey(0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img, dtype=np.float32)
        img = img - 127.5
        img = img * 0.0078125
        img = np.expand_dims(img, axis=0)

        prediction0 = face_recognition.sess_recog.run(face_recognition.output_tensor, feed_dict={face_recognition.images: img, face_recognition.dropout_rate: 1})
        return prediction0

    # 获取任意图片大小的特征向量
    def get_single_feature_vector(self, img_path):
        features_points, _ = self.get_landmarkAndrect(img_path)
        crop_img = self.get_cropImage(features_points[0], img_path)
        flip_img = cv2.flip(crop_img, 1, dst=None)  # 水平镜像
        vector = self.get_512features(crop_img)  # 提取的512维特征向量
        vector_flip = self.get_512features(flip_img)
        # 向量求和，利用求和后的向量做相似度对比
        sum_vector = []
        for i in range(512):
            sum_vector.append(vector[0][i] + vector_flip[0][i])
        return sum_vector

    def cos_sim(self, vector_a, vector_b):
        """
        计算两个向量之间的余弦相似度
        :param vector_a: 向量 a
        :param vector_b: 向量 b
        :return: sim
        """
        if type(vector_a) == list:
            vector_a = np.mat(vector_a)
        if type(vector_b) == list:
            vector_b = np.mat(vector_b)
        num = np.dot(vector_a, vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        sim = num / denom
        return sim

    @staticmethod
    def init_session():
        if face_recognition.is_first:
            face_recognition.init_recog_session()
            face_recognition.is_first = False
            model_dir = face_recognition.model_path
            gragh = tf.get_default_graph()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
            config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
            sess = tf.Session(config=config)
            file_paths = get_model_filenames(model_dir)
            if len(file_paths) == 3:
                print(1)
                image_pnet = tf.placeholder(
                    tf.float32, [None, None, None, 3])
                pnet = PNet({'data': image_pnet}, mode='test')
                out_tensor_pnet = pnet.get_all_output()

                image_rnet = tf.placeholder(tf.float32, [None, 24, 24, 3])
                rnet = RNet({'data': image_rnet}, mode='test')
                out_tensor_rnet = rnet.get_all_output()

                image_onet = tf.placeholder(tf.float32, [None, 48, 48, 3])
                onet = ONet({'data': image_onet}, mode='test')
                out_tensor_onet = onet.get_all_output()

                saver_pnet = tf.train.Saver(
                    [v for v in tf.global_variables()
                     if v.name[0:5] == "pnet/"])
                saver_rnet = tf.train.Saver(
                    [v for v in tf.global_variables()
                     if v.name[0:5] == "rnet/"])
                saver_onet = tf.train.Saver(
                    [v for v in tf.global_variables()
                     if v.name[0:5] == "onet/"])

                saver_pnet.restore(sess, file_paths[0])

                def pnet_fun(img):
                    return sess.run(
                        out_tensor_pnet, feed_dict={image_pnet: img})

                saver_rnet.restore(sess, file_paths[1])

                def rnet_fun(img):
                    return sess.run(
                        out_tensor_rnet, feed_dict={image_rnet: img})

                saver_onet.restore(sess, file_paths[2])

                def onet_fun(img):
                    return sess.run(
                        out_tensor_onet, feed_dict={image_onet: img})

            else:
                saver = tf.train.import_meta_graph(file_paths[0])
                saver.restore(sess, file_paths[1])

                def pnet_fun(img):
                    return sess.run(
                        ('softmax/Reshape_1:0',
                         'pnet/conv4-2/BiasAdd:0'),
                        feed_dict={
                            'Placeholder:0': img})

                def rnet_fun(img):
                    return sess.run(
                        ('softmax_1/softmax:0',
                         'rnet/conv5-2/rnet/conv5-2:0'),
                        feed_dict={
                            'Placeholder_1:0': img})

                def onet_fun(img):
                    return sess.run(
                        ('softmax_2/softmax:0',
                         'onet/conv6-2/onet/conv6-2:0',
                         'onet/conv6-3/onet/conv6-3:0'),
                        feed_dict={
                            'Placeholder_2:0': img})

                face_recognition.p1 = pnet_fun
                face_recognition.p2 = rnet_fun
                face_recognition.p3 = onet_fun

    @staticmethod
    def init_recog_session():
        if face_recognition.is_first:
            model_path = "models/ckpt_model_d"
            # saver = tf.train.import_meta_graph(model_path + '/InsightFace_iter_best_710000.ckpt.meta')  # 加载图结构
            # 恢复tensorflow图，也就是读取神经网络的结构，从而无需再次构建网络
            saver = tf.train.import_meta_graph(model_path + '/InsightFace_iter_best_710000.ckpt.meta')
            # 获取当前图，为了后续训练时恢复变量
            gragh = tf.get_default_graph()
            images = gragh.get_tensor_by_name('img_inputs:0')
            dropout_rate = gragh.get_tensor_by_name('dropout_rate:0')
            output_tensor = gragh.get_tensor_by_name('resnet_v1_50/E_BN2/Identity:0')
            output_tensor1 = gragh.get_tensor_by_name('arcface_loss/norm_embedding:0')
            sess = tf.Session()
            # 如果没有checkpoint文件
            saver.restore(sess, model_path + '/InsightFace_iter_best_710000.ckpt')  # 重点，将地址写到.ckpt

            face_recognition.sess_recog = sess
            face_recognition.images = images
            face_recognition.dropout_rate = dropout_rate
            face_recognition.output_tensor = output_tensor1


