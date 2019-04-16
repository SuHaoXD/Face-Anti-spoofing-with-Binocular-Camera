# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
sign = cap1.isOpened() and cap2.isOpened()
if (sign == False): 
  print("相机打开失败！")
  exit(1)

color = (0,255,0)  #人脸识别边框的颜色

classfier = cv2.CascadeClassifier("D:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt2.xml")  #Opencv人脸识别分类器


cv2.namedWindow('left')
cv2.namedWindow('right')

if __name__ == '__main__':

	while(sign):
		ret1,imgL = cap1.read()     #普通相机读取视频帧
		ret2,imgR = cap2.read()	    #红外相机读取视频帧	
		if not (ret1 and ret2):
			break

		# imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
		# imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
		# disp = stereo.compute(imgL_gray,imgR_gray).astype(np.float32) / 16.0
		# disp = (disp - min_disp)/num_disp

		# disp = stereo.compute(imgL,imgR).astype(np.float32) / 16.0
		# disp = (disp - min_disp)/num_disp

		img_hsv = cv2.cvtColor(imgR,cv2.COLOR_BGR2HSV)
		img_ycrcb = cv2.cvtColor(imgR,cv2.COLOR_BGR2YCrCb)

		Rgray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

		# surf = cv2.xfeatures2d.SURF_create(20000)

		# keypoints,descriptor = surf.detectAndCompute(Rgray,None)
		# # print(type(keypoints),len(keypoints),keypoints[0])
		# # print(descriptor.shape)

		# imgR = cv2.drawKeypoints(image=imgR,keypoints = keypoints,outImage=imgR,color=(255,0,255),
		# 	flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  

		faceRectsL = classfier.detectMultiScale(imgL, scaleFactor = 1.1, minNeighbors = 6, minSize = (30, 30)) #人脸检测
		faceRectsR = classfier.detectMultiScale(imgR, scaleFactor = 1.1, minNeighbors = 6, minSize = (30, 30)) #人脸检测
		if len(faceRectsR) > 0:
			for faceRect1 in faceRectsR:    #框出每一张脸
				x1,y1,w1,h1 = faceRect1
				# cv2.rectangle(imgL,(x - 145,y - 10 + 30),(x + w - 145  ,y + h + 30), color, 2)
				cv2.rectangle(imgR,(x1 ,y1 - 10),(x1 + w1  ,y1 + h1 ), color, 2)
				# cv2.rectangle(disp,(x - 150,y - 10),(x + w-150 ,y + h), color, 2)
			for faceRect2 in faceRectsL:    
				x2,y2,w2,h2 = faceRect2
				cv2.rectangle(imgL,(x2 ,y2 - 10),(x2 + w2  ,y2 + h2 ), color, 2)
				# cv2.rectangle(disp,(x2 ,y2 - 10),(x2 + w2  ,y2 + h2 ), color, 2)	

		cv2.imshow('left',imgL)
		cv2.imshow('right',imgR)
		cv2.imshow('img_hsv',img_hsv)
		cv2.imshow('img_ycrcb',img_ycrcb)

		key = cv2.waitKey(15)

		if key & 0xff == ord('q'):
			break

		

	cap1.release()
	cap2.release()
	cv2.destroyAllWindows()



