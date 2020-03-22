# 批量修改xml

import os
import os.path
import xml.dom.minidom
path="F:/AI/MaskRecognition/yolo3/yolov3_keras/VOCdevkit/VOC2007/Annotations"
files=os.listdir(path)  #得到文件夹下所有文件名称
s=[]
for xmlFile in files: #遍历文件夹
    if not os.path.isdir(xmlFile): #判断是否是文件夹,不是文件夹才打开
        dom=xml.dom.minidom.parse(os.path.join(path,xmlFile))  #最核心的部分,路径拼接,输入的是具体路径
        root=dom.documentElement
        #获取标签对name/pose之间的值
        name=root.getElementsByTagName('folder')
        pose=root.getElementsByTagName('path')
        #原始信息
        # print ('原始信息')
        n0=name[0]
        # print( n0.firstChild.data)
 
        p0=pose[0]
        # print( p0.firstChild.data)
	
	#修改
        n0.firstChild.data='JPEGImages'
        part=xmlFile[0:4]
        part1=part+'.jpg'
        p0.firstChild.data='F:/AI/MaskRecognition/yolo3/yolov3_keras/VOCdevkit/VOC2007/JPEGImages/'+part1
	#打印输出
        # print('修改后的 name')
        # print( n0.firstChild.data)
 
        # print( '修改后的 path')
        # print( p0.firstChild.data)
        # print( '~~~~~')
        with open(os.path.join(path,xmlFile),'w') as fh:
            dom.writexml(fh)
        #     print('name/path OK!')