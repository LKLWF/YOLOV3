# 批量修改图片名，按照递增方式，不满4位补0
import os 
def rename():
	#原始图片路径
	path = r'F:\AI\MaskRecognition\yolo3\yolov3_keras\VOCdevkit\VOC2007\label_nomask'
	#获取该路径下所有图片
	filelist = os.listdir(path)
	a = 1028
	for files in filelist:
		#原始路径 
		Olddir = os.path.join(path,files)
		
		#if os.path.isdir(Olddir):
		#	continue
		#将图片名切片,比如 xxx.bmp 切成xxx和.bmp
		#xxx
		filename = os.path.splitext(files)[0]
		#.bmp
		filetype = os.path.splitext(files)[1]
		#需要存储的路径 a 是需要定义修改的文件名
		Newdir=os.path.join(path,str(a).zfill(4)+filetype)
		os.rename(Olddir,Newdir)
		a += 1
rename()