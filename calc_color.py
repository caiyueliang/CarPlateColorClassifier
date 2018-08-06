# encoding:utf-8
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
import common


def test(name, path):
    img = cv2.imread(path)
    h = np.zeros((256, 256, 3))  # 创建用于绘制直方图的全0图像

    bins = np.arange(256).reshape(256, 1)  # 直方图中各bin的顶点位置
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR三种颜色
    for ch, col in enumerate(color):
        originHist = cv2.calcHist([img], [ch], None, [256], [0, 256])
        cv2.normalize(originHist, originHist, 0, 255 * 0.9, cv2.NORM_MINMAX)
        hist = np.int32(np.around(originHist))
        pts = np.column_stack((bins, hist))
        # print(ch, col, pts)
        cv2.polylines(h, [pts], False, col)

    h = np.flipud(h)

    cv2.imshow(name, h)


def opencv_hist():
    test('blue_1', '/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/0/480467_闽D77V56_0.png')
    test('blue_2', '/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/0/485280_闽D9H686_0.png')
    test('blue_3', '/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/0/485360_闽C990AN_0.png')
    test('blue_4', '/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/0/480466_闽DF3N37_0.png')

    test('green_1', '/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/1/547941_粤YXV317_0.png')
    test('green_2', '/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/1/680889_粤BF0638_0.png')
    test('green_3', '/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/1/649950_粤ADD862_0.png')
    test('green_4', '/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/1/696070_粤BF0638_0.png')

    test('yellow_1', '/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/2/628135_粤Y37765_0.png')
    test('yellow_2', '/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/2/549292_粤Y37765_0.png')
    test('yellow_3', '/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/2/556084_粤Y22839_0.png')
    test('yellow_4', '/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/2/665752_粤Y37765_0.png')
    cv2.waitKey(0)


def hist():
    plt.figure(num='hist', figsize=(10, 10))  # 创建一个名为astronaut的窗口,并设置大小

    plt.subplot(2, 3, 1)  # 将窗口分为两行两列四个子图，则可显示四幅图片
    plt.title('blue_1')  # 第一幅图片标题
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/0/480467_闽D77V56_0.png', 0)
    plt.hist(img.ravel(), 256, [0, 256])

    plt.subplot(2, 3, 2)  # 第二个子图
    plt.title('blue_2')  # 第二幅图片标题
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/0/485280_闽D9H686_0.png', 0)
    plt.hist(img.ravel(), 256, [0, 256])

    plt.subplot(2, 3, 3)  # 第三个子图
    plt.title('green_1')  # 第三幅图片标题
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/1/547941_粤YXV317_0.png', 0)
    plt.hist(img.ravel(), 256, [0, 256])

    plt.subplot(2, 3, 4)  # 第四个子图
    plt.title('green_2')  # 第四幅图片标题
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/1/680889_粤BF0638_0.png', 0)
    plt.hist(img.ravel(), 256, [0, 256])

    plt.subplot(2, 3, 5)
    plt.title('yellow_1')
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/2/628135_粤Y37765_0.png', 0)
    plt.hist(img.ravel(), 256, [0, 256])

    plt.subplot(2, 3, 6)
    plt.title('yellow_2')
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/2/549292_粤Y37765_0.png', 0)
    plt.hist(img.ravel(), 256, [0, 256])

    plt.show()  # 显示窗口


def color_hist():
    plt.figure(num='color_hist', figsize=(10, 10))  # 创建一个名为astronaut的窗口,并设置大小
    color = ('b', 'g', 'r')

    plt.subplot(2, 3, 1)  # 将窗口分为两行两列四个子图，则可显示四幅图片
    plt.title('blue_1')  # 第一幅图片标题
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/0/480467_闽D77V56_0.png')
    for i, col in enumerate(color):
        print i
        histr = cv2.calcHist(img, [i], None, [256], [0.0, 255.0])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.subplot(2, 3, 2)  # 第二个子图
    plt.title('blue_2')  # 第二幅图片标题
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/0/485280_闽D9H686_0.png')
    for i, col in enumerate(color):
        histr = cv2.calcHist(img, [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.subplot(2, 3, 3)  # 第三个子图
    plt.title('green_1')  # 第三幅图片标题
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/1/547941_粤YXV317_0.png')
    for i, col in enumerate(color):
        histr = cv2.calcHist(img, [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.subplot(2, 3, 4)  # 第四个子图
    plt.title('green_2')  # 第四幅图片标题
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/1/680889_粤BF0638_0.png')
    for i, col in enumerate(color):
        histr = cv2.calcHist(img, [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.subplot(2, 3, 5)
    plt.title('yellow_1')
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/2/628135_粤Y37765_0.png')
    for i, col in enumerate(color):
        histr = cv2.calcHist(img, [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.subplot(2, 3, 6)
    plt.title('yellow_2')
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/2/549292_粤Y37765_0.png')
    for i, col in enumerate(color):
        histr = cv2.calcHist(img, [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.show()  # 显示窗口


def image_open():
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/0/480467_闽D77V56_0.png')
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/0/485280_闽D9H686_0.png')
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/1/547941_粤YXV317_0.png')
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/1/680889_粤BF0638_0.png')
    # img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/2/549292_粤Y37765_0.png')
    img = cv2.imread('/home/caiyueliang/deeplearning/CarPlateColorClassifier/Data/color_test/2/549292_粤Y37765_0.png')
    cv2.imshow('img', img)
    img_open = common.image_open(img, (10, 10))
    cv2.imshow('img_open', img_open)
    img_close = common.image_close(img, (10, 10))
    cv2.imshow('img_close', img_close)
    cv2.waitKey(0)


if __name__ == '__main__':
    opencv_hist()
    # hist()
    # color_hist()
    # image_open()
