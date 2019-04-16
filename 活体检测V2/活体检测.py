# -*- coding:utf-8 -*-
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog
import joblib
from sklearn.svm import LinearSVC

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
sign = cap1.isOpened() and cap2.isOpened()
if (sign == False): 
  print("相机打开失败！")
  exit(1)

color = (0,255,0)  #人脸识别边框的颜色

s = False  
size = 170   
value = 0.55       #阈值

list1 = []
list2 = []

classfier = cv2.CascadeClassifier("D:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml")  #Opencv人脸识别分类器

window_size = 8
min_disp = 16
num_disp = 192 - min_disp
bolckSize = window_size
uniquenessRatio = 0
speckleRange = 13
speckleWindowSize = 0
disp12MaxDiff = 200
P1 = 600
P2 = 2400

cv2.namedWindow('left')
cv2.namedWindow('right')
cv2.namedWindow('disparity')

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
	clf = joblib.load('model')
	data_feat = fd.reshape((1, -1)).astype(np.float64)
	# result = clf.predict(data_feat)
	result = clf._predict_proba_lr(data_feat)
	# print(result)
	return result

while(sign):
	ret1,imgL = cap1.read()     #普通相机读取视频帧
	ret2,imgR = cap2.read()	    #红外相机读取视频帧	
	if not (ret1 and ret2):
		break


	stereo = cv2.StereoSGBM_create(
		minDisparity = min_disp,
		numDisparities = num_disp,
		blockSize = window_size,
		uniquenessRatio = uniquenessRatio,
		speckleRange = speckleRange,
		speckleWindowSize = speckleWindowSize,
		disp12MaxDiff = disp12MaxDiff,
		P1 = P1,
		P2 = P2 
	)


	# try:
	# imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
	# imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
	# disp = stereo.compute(imgL_gray,imgR_gray).astype(np.float32) / 16.0
	# disp = (disp - min_disp)/num_disp

	disp = stereo.compute(imgL,imgR).astype(np.float32) / 16.0
	disp = (disp - min_disp)/num_disp

	faceRectsL = classfier.detectMultiScale(imgL, scaleFactor = 1.1, minNeighbors = 6, minSize = (30, 30)) #人脸检测
	faceRectsR = classfier.detectMultiScale(imgR, scaleFactor = 1.1, minNeighbors = 6, minSize = (30, 30)) #人脸检测

	if len(faceRectsR) > 0:
		s = True
		for faceRect2 in faceRectsR:    #框出每一张脸
			x2,y2,w2,h2 = faceRect2
			cv2.rectangle(imgR,(x2 ,y2 - 10),(x2 + w2  ,y2 + h2 ), color, 2)
	else:
		s = False

	if len(faceRectsL) > 0 and s:
		for faceRect1 in faceRectsL:    #框出每一张脸
			x1,y1,w1,h1 = faceRect1
			try:
				image = np.uint8(disp*255.0)
				image = image[y1-10:y1+h1,x1:x1+w1] 
				image = cv2.resize(image,(170,170))	
				# print(image.shape)		
				imgae = Image.fromarray(image)

				result = predict(get_feat(image))

				print(round(result[0][0],5))
				if (round(result[0][0],5) > value):
					# print("real")
					color = (0,255,0)
				else:
					# print("fake")
					color = (0,0,255)
			except:
				print("预测出错")
			cv2.rectangle(imgL,(x1 ,y1 - 10),(x1 + w1  ,y1 + h1 ), color, 2)
			# cv2.rectangle(disp,(x1 ,y1 - 10),(x1 + w1  ,y1 + h1 ), color, 2)	


	cv2.imshow('left',imgL)
	cv2.imshow('right',imgR)
	cv2.imshow('disparity',disp) 
	# except:
	# 	print("error")

	key = cv2.waitKey(15)

	if key & 0xff == ord('q'):
		break

cap1.release()
cap2.release()
cv2.destroyAllWindows()




