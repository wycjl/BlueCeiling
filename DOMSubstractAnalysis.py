# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:48:46 2018
@author: WangYan
Change:2018/8/29
"""
import io
import os
import sys
import cv2
import site
import subprocess
import numpy as np
from osgeo import gdal, gdalconst, osr

bigfile='E:/landingpeng/luoxingall.tif'

# gdal脚本路径
dirs=site.getsitepackages()
gdaldir=dirs[0]+'/Scripts'
gdalcalcpy_file=gdaldir+'/gdal_calc.py'
mergepy_file=gdaldir+'/gdal_merge.py'
# 注册驱动
#gdal.AllRegister()
#打开图像数据集
#img = gdal.Open(bigfile)


spitdir = os.path.dirname(bigfile)
if(os.path.exists('tile')==False):
	os.makedirs('tile')
gdalretile_file=gdaldir+'/gdal_retile.py'
gdal_retile ='python '+gdalretile_file+' -ps 5000 5000 -co "ALPHA=YES" -r bilinear -targetDir '+'tile'+' '+bigfile
os.system(gdal_retile)
print('----------裁剪完成---------------')

print('----------开始检测---------------')

if (os.path.exists('temp')==False):
	os.makedirs('temp')
if (os.path.exists('out')==False):
	os.makedirs('out')
if (os.path.exists('coord')==False):
	os.makedirs('coord')
pwd = os.getcwd()
pwd_temp = os.path.join(pwd,'temp')
pwd_out = os.path.join(pwd,'out')

rootdir = 'tile'
list = os.listdir(rootdir)
for i in range(0,len(list)):
	if os.path.splitext(list[i])[1] == '.tif':
		infile=os.path.join(rootdir,list[i])
		tempfile=os.path.join(pwd_temp,os.path.splitext(list[i])[0]+'-temp.tif')
		outfile=os.path.join(pwd_out,os.path.splitext(list[i])[0]+'-out.tif')
		file = os.path.join(pwd_out,os.path.splitext(list[i])[0]+'coord.txt')
		f = open(file,'w')
		img = cv2.imread(infile)
#cv2.imshow('img',img)

#kernel_2 = np.ones((2,2),np.uint8)#2x2的卷积核
#kernel_3 = np.ones((3,3),np.uint8)#3x3的卷积核
		kernel_4 = np.ones((4,4),np.uint8)#4x4的卷积核

		hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		low_range = np.array([209/2, 45*255/100, 85*255/100])
		high_range = np.array([212/2, 58*255/100, 95*255/100])
		mask = cv2.inRange(hue_image, low_range, high_range)

# =============================================================================
# erosion = cv2.erode(mask,kernel_4,iterations = 1)
# 
# erosion = cv2.erode(erosion,kernel_4,iterations = 1)
# 
# dilation = cv2.dilate(erosion,kernel_4,iterations = 1)
# 
		dilation = cv2.dilate(mask,kernel_4,iterations = 1)
		# =============================================================================
		cv2.imwrite(tempfile,mask)
		mask=cv2.imread(tempfile)
		#target是把原图中的非目标颜色区域去掉剩下的图像
		target = cv2.bitwise_and(img, mask)

		ret, binary = cv2.threshold(dilation,127,255,cv2.THRESH_BINARY) 

		#在binary中发现轮廓，轮廓按照面积从小到大排列
		_, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		p=0
		for i in contours:#遍历所有的轮廓
		   x,y,w,h = cv2.boundingRect(i)#将轮廓分解为识别对象的左上角坐标和宽、高
		   #在图像上画上矩形（图片、左上角坐标、右下角坐标、颜色、线条宽度）
		   if w<25 and h<25:
		      continue
		   else:
		      cv2.rectangle(img,(x,y),(x+w,y+h),(0,100,255),8)
		   #给识别对象写上标号
		      font=cv2.FONT_HERSHEY_SIMPLEX
		      x0=(x+w)/2
		      y0=(y+h)/2
		   #str2='('+str(x0)+','+str(y0)+')'
		      str2 = str(p+1)
		      f.writelines(str2+'  '+str(x)+'  '+str(y)+'  '+str(w)+'  '+str(h)+'\n')
		      cv2.putText(img,str2,(x,y-5), font, 1,(100,100,255),2)#加减10是调整字符位置
		   p+=1
		f.close()
		# cv2.imshow('target', target)
		# cv2.imshow('Mask', mask)
		# cv2.imshow("prod", dilation)
		# cv2.namedWindow("img", cv2.WINDOW_NORMAL)
		# cv2.imshow('img', img)
		cv2.imwrite(outfile, img)#将画上矩形的图形保存到当前目录  

		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

		#设置坐标参考系统
		#注册驱动
		gdal.AllRegister()
		#打开Dom影像
		ods1=gdal.Open(infile,gdal.GA_ReadOnly)
		#更新影像
		ods2=gdal.Open(outfile,gdal.GA_Update)
		# 设置投影
		srs=ods1.GetProjectionRef()
		ods2.SetProjection(srs)
		#设置六参数
		geotransform=ods1.GetGeoTransform()
		ods2.SetGeoTransform(geotransform)
		# 使用FlushCache将数据写入文件
		ods2.FlushCache()
		print(i)


print('-----合并所有的outfile------')

def listdir(rootdir):
	filenames=[]
	for filename in os.listdir(rootdir):
		pathname = os.path.join(rootdir,filename)
		if (os.path.isfile(filename)):
			print(pathname)
			filenames.append(pathname)
	return filenames

def getfilelist(filepaths):
    files=''
    for item in filepaths:
        files+=' '+item
    return files

rootdir = 'out'
files = ''
for filename in os.listdir(rootdir):
	print(filename)
	files+=' '+rootdir+'/'+filename
mergeImg = bigfile.replace('.tif','_merge.tif')

gdal_merge ='python '+mergepy_file+'  -o '+mergeImg_file+files
os.system(gdal_merge)

print('---------合并完成---------')



