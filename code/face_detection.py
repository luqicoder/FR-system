import time
import tensorflow as tf
import cv2
import numpy as np
from mtcnn.mtcnn import PNet, RNet, ONet
from mtcnn.tools import detect_face, get_model_filenames


# 获取5个特征点和人脸检测框
def get_landmarkAndrect(image_path='./images/0001.png'):
    model_dir = './models/all_in_one/'
    minsize = 20
    threshold = [0.8, 0.8, 0.8]
    factor = 0.7

    img = cv2.imread(image_path)
    # cv2.imshow("1", img)
    # cv2.waitKey(0)
    # tools.py
    file_paths = get_model_filenames(model_dir)
    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
            config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
            # config = tf.ConfigProto(allow_soft_placement=True)
            with tf.Session(config=config) as sess:
                if len(file_paths) == 3:
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

                start_time = time.time()
                rectangles, points = detect_face(img, minsize,
                                                 pnet_fun, rnet_fun, onet_fun,
                                                 threshold, factor)
                duration = time.time() - start_time

                # print("检测时间：", duration)
                # print(type(rectangles))
                points = np.transpose(points)
                # print("边界框：", rectangles[0])
                # for i in range(len(rectangles)):
                #     print("边界框：", rectangles[i])
                # print("特征点", type(points), points.reshape(5, 2))#待修改
                # 如果有多个人的特征点
                num = points.shape[0]
                if num == 0:
                    return [], []

                split_points = np.vsplit(points, num)
                k = 0
                for i in split_points:
                    split_points[k] = i.reshape(5, 2)
                    k += 1
                return split_points, rectangles


if __name__ == '__main__':
    path = './images/person7.jpg'
    img = cv2.imread(path)
    img_shape = img.shape
    fea_points, rectangles = get_landmarkAndrect(path)
    res_size = len(fea_points)
    for i in range(res_size):
        rect = rectangles[i]
        points = fea_points[i]
        top_left = (max(0, int(rect[0])), max(0, int(rect[1])))
        right_bottom = (min(img_shape[1], int(rect[2])), min(img_shape[0], int(rect[3])))
        cv2.rectangle(img, top_left, right_bottom, (0, 0, 255), 1)
        for i in range(5):
            cv2.circle(img, (points[i][0], points[i][1]), 1, (0, 0, 255), 4)
    cv2.imshow('1', img)
    cv2.waitKey(0)
