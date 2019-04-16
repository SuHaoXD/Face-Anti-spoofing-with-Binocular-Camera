import os
import random
import cv2

# path=r"D:\活体检测V1\HOG_SVM-master\image\\"       

# #获取该目录下所有文件，存入列表中
# f=os.listdir(path)

# n=0
# for i in f:
    
#     #设置旧文件名（就是路径+文件名）
#     oldname=path+f[n]
    
#     #设置新文件名
#     newname=path+str(2000+n+1)+'.jpg'   
#     #用os模块中的rename方法对文件改名
#     os.rename(oldname,newname)
#     print(oldname,'======>',newname)
#     n+=1





# dir = os.getcwd()
# dir = os.chdir(r'D:\活体检测V1\HOG_SVM-master\imageTest')
# f = open(r'D:\活体检测V1\HOG_SVM-master\imageTest\train.txt','a')

# dir = os.chdir(r'E:\HOG_SVM-master\imageTest')
# f = open(r'E:\HOG_SVM-master\imageTest\train.txt','a')

# j = 0
# for file in os.listdir(dir):
#      #os.listdir('.')遍历文件夹内的每个文件名，并返回一个包含文件名的list
#     if file[-3: ] == 'txt':
#         continue   #过滤掉改名的.py文件
#     if file[0] == '1':
#     	m = 1
#     else:
#     	m = 2
#     # os.rename(file,'bg'+str(j)+'.jpg')
#     f.write(file + " " + str(m) + '\n')
#     j+=1 
#     print (file)

# img = cv2.imread(r"D:\NormalizedFace\ImposterNormalized\0008\0008_00_00_01_10.bmp")

# print(img.shape)
