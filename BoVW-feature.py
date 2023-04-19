# import argparse as ap            # argparse是python用于解析命令行参数和选项的标准模块
import cv2
import numpy as np
import os
# scikit-learn是机器学习库，它具有各种分类：回归和聚类算法，包括支持向量机，随机森林，梯度提升，k均值和DBSCAN，并且旨在与Python数值科学库NumPy和SciPy联合使用。
# from sklearn.externals import joblib  # scikit-learn库,但是后续版本已经取消了joblib模块
import joblib
# scipy是一个科学运算库
from scipy.cluster.vq import *    # 提供了k-means的实现,后续可以直接使用kmean函数实现k均值聚类
from sklearn import preprocessing

# # 通过parse使得可以通过终端向里面传递参数---，终端获取参数的一种方法吧
# parser = ap.ArgumentParser()
# parser.add_argument("-t", "--trainingSet", help="Path to Training Set", )
# # parser.add_argument("-t", "--trainingSet", help="Path to Training Set", )  # required="True"
# args = vars(parser.parse_args())

# 获取训练数据，终端获取或者直接给出路径
# train_path = args["trainingSet"]           # python BoVW.py -t E:\DataSet\tiger\train\test
train_path = "E:\DataSet/tiger/train/test"
training_names = os.listdir(train_path)  # 访问路径下文件或文件夹，返回名称

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
for training_name in training_names:
    image_path = os.path.join(train_path, training_name)
    image_paths += [image_path]

# 聚类数量，也是最后生成的word数量，因为聚类就是把相似的聚类成一个word
numWords = 1200

# Create feature extraction and keypoint detector objects
# List where all the descriptors are stored
sift = cv2.SIFT_create()
des_list = []

# 特征提取
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，返回数据下标和数据。
for i, image_path in enumerate(image_paths):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print("Extract SIFT of {} image, NO.{}/{}".format(training_names[i], i, len(image_paths)))
    kpts, des = sift.detectAndCompute(gray, None)
    des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
# downsampling = 1
# descriptors = des_list[0][1][::downsampling,:]
# for image_path, descriptor in des_list[1:]:
#    descriptors = np.vstack((descriptors, descriptor[::downsampling,:]))

# Stack all the descriptors vertically in a numpy array
# 将所有描述符都叠放在一个numpy数组中，vstack沿着行向下堆叠
descriptors = des_list[0][1]   # 初始化列表，先填入第一个
for image_path, descriptor in des_list[1:]:   # 第二行开始向下堆叠
    descriptors = np.vstack((descriptors, descriptor))

# Perform k-means clustering
# 返回的是聚类后的中心向量组成的数组（字典） 和 平均欧式距离？
# 因此，voc是一个（1000,128）数组，因为SIFT描述符就是128的向量，聚类的时候就是选择一个作为质心，然后不断迭代至最佳
print("Start k-means: {} words, {} key points".format(numWords, descriptors.shape[0]))
voc, variance = kmeans(descriptors, numWords, 1)     # 1表示自动迭代值收敛,也可以取其他具体数值

# Calculate the histogram of features
# vq函数是如何检索比对的，可能就是使用的K叉树？
im_features = np.zeros((len(image_paths), numWords), "float32")    # 生成一个数组存储每幅图生成直方图
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)         # 判断第i副图像的描述符与哪个单词相近，返回给words，而distance则是对应的距离
    for w in words:
        im_features[i][w] += 1                        # 第i副图的直方图生成完毕


# Perform Tf-Idf vectorization
# 下面这个只乘了IDF？
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)  # 求每一列的和
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Perform L2 normalization
im_features = im_features*idf
im_features = preprocessing.normalize(im_features, norm='l2')   # 归一化

joblib.dump((im_features, image_paths, idf, numWords, voc), "bof.pkl", compress=3)  # 保存模型,名称为bof.pkl
