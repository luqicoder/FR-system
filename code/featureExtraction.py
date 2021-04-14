import tensorflow as tf
import cv2 as cv
import numpy as np

model_path = "models/ckpt_model_d"
# saver = tf.train.import_meta_graph(model_path + '/InsightFace_iter_best_710000.ckpt.meta')  # 加载图结构
# 恢复tensorflow图，也就是读取神经网络的结构，从而无需再次构建网络
saver = tf.train.import_meta_graph(model_path + '/InsightFace_iter_best_710000.ckpt.meta')
gragh = tf.get_default_graph()  # 获取当前图，为了后续训练时恢复变量
images = gragh.get_tensor_by_name('img_inputs:0')
dropout_rate = gragh.get_tensor_by_name('dropout_rate:0')
output_tensor = gragh.get_tensor_by_name('resnet_v1_50/E_BN2/Identity:0')
output_tensor1 = gragh.get_tensor_by_name('arcface_loss/norm_embedding:0')
sess = tf.Session()
# 如果没有checkpoint文件
saver.restore(sess, model_path + '/InsightFace_iter_best_710000.ckpt')  # 重点，将地址写到.ckpt


# 如果有checkpoint文件
# saver.restore(sess, tf.train.latest_checkpoint(model_path))# 加载变量值


def get_512features(img_path='images/112_112.jpg'):
    img = img_path
    if type(img_path) == str:
        img = cv.imread(img_path, cv.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float32)
        # 图片尺寸更改为112 * 112
        if img.shape[0] != 112 or img.shape[1] != 112:
            print("需要更改图片尺寸")
            img = cv.resize(img, (112, 112))
    else:
        if img.shape == (112, 96, 3):
            z = np.zeros((112, 8, 3), dtype=np.float32)
            img = np.append(img, z, axis=1)
            img = np.append(z, img, axis=1)
            # img = np.array(img, dtype=np.uint8)
            # cv.imshow("img",img)
            # cv.waitKey(0)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.array(img, dtype=np.float32)
    img = img - 127.5
    img = img * 0.0078125
    img = np.expand_dims(img, axis=0)

    prediction0 = sess.run(output_tensor1, feed_dict={images: img, dropout_rate: 1})
    return prediction0


if __name__ == '__main__':
    fea = get_512features('images/affine.jpg')
    print(fea)
