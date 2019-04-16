# -*- coding=utf-8 -*-
from PIL import Image
import cv2
from skimage.feature import hog
import numpy as np
import joblib
from sklearn.svm import LinearSVC
# from sklearn.svm import SVC

size = 170

# 变成灰度图片
def rgb2gray(im):
    # gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    gray = im[:, :, 0]
    return gray

def get_feat(image):
	image = np.reshape(image, (size, size, -1))
	gray = rgb2gray(image) / 255.0
	fd = hog(gray, orientations=12,block_norm='L1', pixels_per_cell=[8, 8], cells_per_block=[4, 4], visualize=False,
                 transform_sqrt=True)
	return fd

def predict(fd):
	clf = LinearSVC()
	# clf = SVC(probability=True)
	clf = joblib.load('model')
	data_feat = fd.reshape((1, -1)).astype(np.float64)
	# result = clf.predict(data_feat)
	# print(dir(clf))
	result = clf._predict_proba_lr(data_feat)
	print(result)
	return result

# temp = Image.open(r"D:\活体检测V1\real\1110.jpg")
# image = temp.copy()
# temp.close()

disp = cv2.imread("./real/1095.jpg")
image = Image.fromarray(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB))
print(disp.shape)
# cv2.cvtColor(disp,cv2.COLOR_BGR2RGB)
# cv2.imshow()
predict(get_feat(image))
# cv2.imshow("OpenCV",disp)
# image.show()


