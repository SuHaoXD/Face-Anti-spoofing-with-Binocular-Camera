# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import multiprocessing

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
sign = cap1.isOpened() and cap2.isOpened()
if (sign == False): 
  print("相机打开失败！")
  exit(1)

color = (0,255,0)  #人脸识别边框的颜色

i = 2001
# i = sorted([int(k[:-4]) for k in os.listdir('./HOG_SVM-master/imageTest/')])[-1] + 1
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


def com(list1,list2):
	while True:
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

		# print(len(list1))
		# print(len(list2))

		if list1 and list2:
			L = list1.pop(0)
			R = list2.pop(0)
			disp = stereo.compute(L,R).astype(np.float32) / 16.0
			disp = (disp - min_disp)/num_disp
			cv2.imshow('disparity',disp)

if __name__ == '__main__':

	# def callbackFunc(e, x, y, f, p):
	#     if e == cv2.EVENT_LBUTTONDOWN:        
	#         print(x,y)

	# cv2.setMouseCallback("left", callbackFunc, None)
	# cv2.setMouseCallback("right", callbackFunc, None)
	# cv2.setMouseCallback("disparity", callbackFunc, None)
	multiprocessing.freeze_support()

	list1 = multiprocessing.Manager().list()
	list2 = multiprocessing.Manager().list()

	p = multiprocessing.Process(target = com,args = (list1,list2,))
	p.start()

	while(sign):
		ret1,imgL = cap1.read()     #普通相机读取视频帧
		ret2,imgR = cap2.read()	    #红外相机读取视频帧	
		if not (ret1 and ret2):
			break

		# rgray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
		# imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

		list1.append(imgL)
		list2.append(imgR)

		# stereo = cv2.StereoSGBM_create(
		# 	minDisparity = min_disp,
		# 	numDisparities = num_disp,
		# 	blockSize = window_size,
		# 	uniquenessRatio = uniquenessRatio,
		# 	speckleRange = speckleRange,
		# 	speckleWindowSize = speckleWindowSize,
		# 	disp12MaxDiff = disp12MaxDiff,
		# 	P1 = P1,
		# 	P2 = P2 
		# )	

		try:
			# imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
			# imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
			# disp = stereo.compute(imgL_gray,imgR_gray).astype(np.float32) / 16.0
			# disp = (disp - min_disp)/num_disp

			# disp = stereo.compute(imgL,imgR).astype(np.float32) / 16.0
			# disp = (disp - min_disp)/num_disp

			faceRectsL = classfier.detectMultiScale(imgL, scaleFactor = 1.1, minNeighbors = 6, minSize = (30, 30)) #人脸检测
			faceRectsR = classfier.detectMultiScale(imgR, scaleFactor = 1.1, minNeighbors = 6, minSize = (30, 30)) #人脸检测
			if len(faceRectsR) > 0:
				for faceRect1 in faceRectsR:    #框出每一张脸
					x1,y1,w1,h1 = faceRect1
					# cv2.rectangle(imgL,(x - 145,y - 10 + 30),(x + w - 145  ,y + h + 30), color, 2)
					cv2.rectangle(imgR,(x1 ,y1 - 10),(x1 + w1  ,y1 + h1 ), color, 2)
					# cv2.rectangle(disp,(x - 150,y - 10),(x + w-150 ,y + h), color, 2)
				for faceRect2 in faceRectsL:    #框出每一张脸
					x2,y2,w2,h2 = faceRect2
					cv2.rectangle(imgL,(x2 ,y2 - 10),(x2 + w2  ,y2 + h2 ), color, 2)
					# cv2.rectangle(disp,(x2 ,y2 - 10),(x2 + w2  ,y2 + h2 ), color, 2)	

			cv2.imshow('left',imgL[0:480,192:640])
			cv2.imshow('right',imgR[0:480,192:640])
 
		except:
			print("error")

		key = cv2.waitKey(15)

		if key & 0xff == ord('q'):
			break

		# elif key & 0xff == ord("d"):
		# 	print("左边坐标:",faceRectsL)
		# 	print("右边坐标:",faceRectsR)

		elif key & 0xff == ord("s"):
			try:			
				# print(disp.shape)
				# print(imgL.shape)
				# print(faceRect)
				disp = np.uint8(disp*255.0)
				disp = disp[y2-10:y2+h2,x2:x2+w2] 
				disp = cv2.resize(disp,(170,170))
				print(disp.shape)
				# disp = disp[:,:,np.newaxis]
				# disp = np.concatenate([disp,disp,disp],axis = 2)
				# print(disp.shape)
				# disp = np.uint8(disp*255.0)
				# with open("111.txt",'w') as fp:
				# 	fp.write(str(disp))
				cv2.imwrite("./" + str(i) + ".jpg",disp)
				i += 1
			except:
				pass

	cap1.release()
	cap2.release()
	cv2.destroyAllWindows()



