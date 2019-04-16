# -*- coding:utf-8 -*-
import cv2
import numpy as np

def update(val = 0):
	global disp 
	stereo.setBlockSize(cv2.getTrackbarPos('window_size','disparity'))
	stereo.setUniquenessRatio(cv2.getTrackbarPos('uniquenessRatio','disparity'))
	stereo.setSpeckleWindowSize(cv2.getTrackbarPos('speckleWindowSize','disparity'))
	stereo.setSpeckleRange(cv2.getTrackbarPos('speckleRange','disparity'))
	stereo.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff','disparity'))

	disp = stereo.compute(imgL,imgR).astype(np.float32) / 16.0

window_size = 8
min_disp = 16
num_disp = 192- min_disp
bolckSize = window_size
uniquenessRatio = 0
speckleRange = 13
speckleWindowSize = 0
disp12MaxDiff = 200
P1 = 600
P2 = 2400

cv2.namedWindow('disparity')
cv2.createTrackbar('speckleRange','disparity',speckleRange,50,update)
cv2.createTrackbar('window_size','disparity',window_size,21,update)
cv2.createTrackbar('speckleWindowSize','disparity',speckleWindowSize,200,update)
cv2.createTrackbar('uniquenessRatio','disparity',uniquenessRatio,50,update)
cv2.createTrackbar('disp12MaxDiff','disparity',disp12MaxDiff,250,update)


cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
sign = cap1.isOpened() and cap2.isOpened()
if (sign == False):
  print("相机打开失败！")


while(sign):
	ret1,imgL = cap1.read()     #普通相机读取视频帧
	ret2,imgR = cap2.read()	    #红外相机读取视频帧	
	if not (ret1 and ret2):
		break
	
	# imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

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
	update()

	# cv2.imshow('normal',imgl)
	cv2.imshow('left',imgL)
	cv2.imshow('right',imgR)
	cv2.imshow('disparity',(disp - min_disp)/num_disp)

	if cv2.waitKey(15) & 0xff == ord('q'):
		break

cap1.release()
cap2.release()
cv2.destroyAllWindows()



