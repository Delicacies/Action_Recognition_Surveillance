from os import name
import os
import cv2
import sys
import numpy as np
from cv2 import pointPolygonTest

def readRegionTXT(path='Utils/region.txt'):
    region_points = []
    for line in open(path):
        str_list = line.split(" ")
        point = [int(str_list[0]), int(str_list[1])]
        region_points.append(point)
    return region_points

def readRegion(path='Utils/region.txt'):
    region_points = []
    for line in open(path):
        str_list = line.split(" ")
        point = [int(str_list[0]), int(str_list[1])]
        region_points.append(point)
    return region_points


def getRegionCamera(count, camera=0, regionpath='Utils/region.txt'): 
    region = []
    fw = open(regionpath, 'w+')    # 将要输出保存的文件地址
    # if os.path.exists(regionpath):
    #     fw = open(regionpath, 'w+')    # 将要输出保存的文件地址
    # else:
    #     os.makedirs(regionpath)
    #     fw = open(regionpath, 'w+')
    
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        # if event == cv2.EVENT_RBUTTONDOWN:
        #     finish = True
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            cv2.circle(frame, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(frame, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", frame)
            fw.write(f"{x} {y}\n")
            # for line in open(regionpath, "w+"):    # 读取的文件
            #     fw.write(f"[{x},{y}]")    # 将字符串写入文件中
            #     # line.rstrip("\n")为去除行尾换行符
            #     fw.write("\n")    # 换行
            region.append([x,y])
            print(x,y)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cap = cv2.VideoCapture(camera)
    while(cap.isOpened()):
        res, frame = cap.read(0)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if len(region)==count:
            drawregion(frame, region=region)
            fw.close()
            break
            # return region

    return region

def getRegionFrame(count, frame=None, regionpath='Utils/region.txt'): 
    region = []
    fw = open(regionpath, 'w+')    # 将要输出保存的文件地址  
    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            cv2.circle(frame, (x, y), 1, (0, 0, 255), thickness=-1)
            cv2.putText(frame, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("frame", frame)
            fw.write(f"{x} {y}\n")
            region.append([x,y])
            print(x,y)

    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", on_EVENT_LBUTTONDOWN)
    while(True):
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if len(region)==count:
            drawregion(frame, region=region)
            fw.close()
            break
    cv2.destroyWindow("frame")
    return region


def drawregion(image, region=[[10,10],[480,10],[480,480],[10,480]], danger=False):
    point_color = (255, 0, 0) if danger else (0, 255, 0)
    count = len(region)
    thickness = 2 
    lineType = 4
    for i in range(count-1):
        cv2.line(image, (int(region[i][0]),int(region[i][1])), (int(region[i+1][0]),int(region[i+1][1])), point_color, thickness, lineType)
    if count > 1:
        cv2.line(image, (int(region[count-1][0]),int(region[count-1][1])), (int(region[0][0]),int(region[0][1])), point_color, thickness, lineType)
    return image
    # cv2.namedWindow("image")
    # while True:
    #     cv2.imshow('image', image)
    #     if cv2.waitKey(1) & 0xFF==ord('q'):
    #         break
    # # cv2.waitKey (10000) # 显示 10000 ms 即 10s 后消失
    # cv2.destroyAllWindows()


def judgeInArea(region=[[10,10],[480,10],[480,480],[10,480]], det=None):
    rect = np.array(region, dtype=np.int32)
    danger_person = []
    num = len(det)
    for i in range(num):
        dist = cv2.pointPolygonTest(rect,((det[i][0] + det[i][2])/2,det[i][3]),0)
        # print(dist)
        if dist>=0:
            danger_person.append(i)
            # return True
    # print(danger_person)
    return danger_person

def judgeInAreaOnePerson(region=[[10,10],[480,10],[480,480],[10,480]], det=None):
    rect = np.array(region, dtype=np.int32)
    dist = cv2.pointPolygonTest(rect, (int((det[0] + det[2])/2),int(det[3])), 0)
    return dist >= 0

if __name__ == "__main__":
    # image = '000002.jpg'
    count = 4
    # region = positionget(conut, image)
    region = getRegionCamera(count, 0)
    # drawregion(image, region)
    # judgeInArea(region, [[100,100, 20200]])




 
