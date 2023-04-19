# import argparse as ap
import cv2
import imutils
import os
import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
from pylab import *
from PIL import Image

# # Get the path of the training set
# 终端获取查询集
# parser = ap.ArgumentParser()
# # parser.add_argument("-i", "--image", help="Path to query image", required="True")
# parser.add_argument("-i", "--image", help="Path to query image")
# args = vars(parser.parse_args())

# # Get query image path
# image_path = args["image"]
image_path = "E:\DataSet/tiger/train/test/009.jpg"

# Load the classifier, class names, scaler, number of clusters and vocabulary
# 使用joblb加载之前训练好的数据集
im_features, image_paths, idf, numWords, voc = joblib.load("./bof.pkl")

# 对待查询的图像进行特征提取和直方图绘制
sift = cv2.SIFT_create()
des_list = []
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
kpts, des = sift.detectAndCompute(img, None)
des_list.append((image_path, des))

descriptors = des_list[0][1]  # 特征描述符向量

test_features = np.zeros((1, numWords), "float32")
words, distance = vq(descriptors, voc)
for w in words:                 # 单词直方图绘制
	test_features[0][w] += 1

# Perform Tf-Idf vectorization and L2 normalization
test_features = test_features*idf
test_features = preprocessing.normalize(test_features, norm='l2')

# 使用乘积量化图像间的相似关系
score = np.dot(test_features, im_features.T)
rank_ID = np.argsort(-score)   # 返回排序的下标

# 打印前5张的评分
score_result = []
for j in rank_ID[0][0:]:
	score_result.append(score[0][j])
	if j == rank_ID[0][4]:
		break
print("相似度评分最高的5张")
print(score_result)

# Visualize the results 前5张
vs0 = cv2.imread(image_path)    # 原图像
cv2.imshow("test_image", vs0)
a = rank_ID[0][0]
vs1 = cv2.imread(image_paths[a])
for i in rank_ID[0][1:]:
	img = cv2.imread(image_paths[i])
	vs1 = np.hstack((vs1, img))
	if i == rank_ID[0][4]:
		break
cv2.imshow("search_result",vs1)
cv2.waitKey(0)
cv2.destroyAllWindows()
# vs1 = np.hstack((img, invert))  # 水平堆叠
# vs2 = np.hstack((gaussianBlur, flip))  # 水平堆叠
# result = np.vstack((vs1, vs2))  # 竖直堆叠

#  原方案看不懂
# figure()
# gray()
# subplot(5, 4, 1)
# imshow(img[:, :, ::-1])
# axis('off')
# for i, ID in enumerate(rank_ID[0][0:16]):
# 	img2 = Image.open(image_paths[ID])
# 	gray()
# 	subplot(5, 4, i+5)
# 	imshow(img2)
# 	axis('off')
#
# show()
