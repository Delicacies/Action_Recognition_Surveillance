#import ctypes
import threading
import time
import os
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  #root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from HikVision_Python_Header import NET_DVR_DEVICEINFO_V30,NET_DVR_PREVIEWINFO,FRAME_INFO
import ctypes
from ctypes import *
import cv2
import numpy as np
import cupy as cp
from matplotlib import pyplot as plt

# 初始化全局变量
PLAYCTRL_PORT0 = c_int(-1)
PLAYCTRL_PORT1 = c_int(-1)
PLAYCTRL_PORT2 = c_int(-1)
PLAYCTRL_PORT3 = c_int(-1)
q0=None
q1=None
q2=None
q3=None


# 码流回调数据类型
NET_DVR_SYSHEAD = 1
NET_DVR_STREAMDATA = 2
NET_DVR_AUDIOSTREAMDATA = 3
NET_DVR_PRIVATE_DATA = 112
# 码流回调函数
#REALDATACALLBACK = CFUNCTYPE(None, c_long, c_ulong, POINTER(c_ubyte), c_ulong, c_void_p)
REALDATACALLBACK = CFUNCTYPE(None, c_long, c_ulong, POINTER(c_byte), c_ulong, c_void_p)
# 解码回调函数
# DECCBFUN = CFUNCTYPE(None, c_long, POINTER(c_ubyte), c_long, POINTER(FRAME_INFO), c_long, c_long)
DECCBFUN = CFUNCTYPE(None, c_int, POINTER(c_ubyte), c_int, POINTER(FRAME_INFO), c_int, c_int)

# HCNetSDK库文件路径
#HCNetSDK_Path = "/home/zmhh/Action_Recognition/HikVisionCamera/lib1/libhcnetsdk.so"
HCNetSDK_Path = "libhcnetsdk.so"
HCNetSDK_Path = str(ROOT)+'/lib/'+HCNetSDK_Path

# PlatCtrlSDK库文件路径
#PlayCtrlSDK_Path = "/home/zmhh/Action_Recognition/HikVisionCamera/lib1/libPlayCtrl.so"
PlayCtrlSDK_Path = "libPlayCtrl.so"
PlayCtrlSDK_Path = str(ROOT)+'/lib/'+PlayCtrlSDK_Path
#PlayCtrlSDK_Path = r'./lib/libPlayCtrl.so'

# ld_library_path = os.getenv("LD_LIBRARY_PATH")
# so_library_path = str(ROOT)+'/lib'
# if so_library_path not in ld_library_path:
#     os.environ['LD_LIBRARY_PATH']+=':'+so_library_path
#     os.execv(sys.argv[0], sys.argv)

# # 加载动态链接库
# cwd_path = os.getcwd()
# #so_path = str(cwd_path)+'/HiKCamSDK/lib'
# so_path = str(cwd_path)+'/lib'
# os.chdir(so_path)
# pdll = ctypes.CDLL(PlayCtrlSDK_Path)
# dll = ctypes.CDLL(HCNetSDK_Path)
# os.chdir(cwd_path)

pdll = ctypes.cdll.LoadLibrary(PlayCtrlSDK_Path)
dll = ctypes.cdll.LoadLibrary(HCNetSDK_Path)

#四个摄像头对应图像
img_dict= {
    0:None,
    1:None,
    2:None,
    3:None
}
# 初始化HCNetSDK库
def InitHCNetSDK(dll):
    # 初始化DLL
    dll.NET_DVR_Init()
    # 设置设备超时时间
    dll.NET_DVR_SetConnectTime(int(5000), 4) #超时时间5s, 最大重连次数4次
    dll.NET_DVR_SetReconnect(10000,True)     #重连间隔10s

# 定义解码回调函数
def DecodeCallback0(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2):
    global q0
    # print("Frame information:")
    # print("pFrameInfo.contents.nType",pFrameInfo.contents.nType)
    # print("Frame Width",pFrameInfo.contents.nWidth)
    # print("Frame Height",pFrameInfo.contents.nHeight)
    if pFrameInfo.contents.nType == 3:

        t = time.time()

        img = np.frombuffer(bytes(pBuf[0:nSize]), cp.uint8)
        img.resize(((pFrameInfo.contents.nHeight * 3) // 2, pFrameInfo.contents.nWidth))

        q0 = img
        #print("frombuffer:", time.time() - t)

# 定义解码回调函数
def DecodeCallback1(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2):
    global q1
    # print("Frame information:")
    # print("pFrameInfo.contents.nType",pFrameInfo.contents.nType)
    # print("Frame Width",pFrameInfo.contents.nWidth)
    # print("Frame Height",pFrameInfo.contents.nHeight)
    if pFrameInfo.contents.nType == 3:

        t = time.time()

        img = np.frombuffer(bytes(pBuf[0:nSize]), cp.uint8)
        img.resize(((pFrameInfo.contents.nHeight * 3) // 2, pFrameInfo.contents.nWidth))

        q1 = img
        # print("frombuffer:", time.time() - t)
# 定义解码回调函数
def DecodeCallback2(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2):
    global q2
    #print("pBuf",pBuf)
    # print("Frame information:")
    # print("pFrameInfo.contents.nType",pFrameInfo.contents.nType)
    # print("Frame Width",pFrameInfo.contents.nWidth)
    # print("Frame Height",pFrameInfo.contents.nHeight)
    if pFrameInfo.contents.nType == 3:

        t = time.time()

        img = np.frombuffer(bytes(pBuf[0:nSize]), cp.uint8)
        img.resize(((pFrameInfo.contents.nHeight * 3) // 2, pFrameInfo.contents.nWidth))

        q2 = img
        # print("frombuffer:", time.time() - t)
# 定义解码回调函数
def DecodeCallback3(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2):
    global q3
    #print("pBuf",pBuf)
    # print("Frame information:")
    # print("pFrameInfo.contents.nType",pFrameInfo.contents.nType)
    # print("Frame Width",pFrameInfo.contents.nWidth)
    # print("Frame Height",pFrameInfo.contents.nHeight)
    if pFrameInfo.contents.nType == 3:

        t = time.time()

        img = np.frombuffer(bytes(pBuf[0:nSize]), cp.uint8)
        img.resize(((pFrameInfo.contents.nHeight * 3) // 2, pFrameInfo.contents.nWidth))

        q3 = img
        # print("frombuffer:", time.time() - t)

# 定义码流回调函数
def RealDataCallBack0(lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
    if dwDataType == NET_DVR_SYSHEAD:
        # 设置流播放模式
        pdll.PlayM4_SetStreamOpenMode(PLAYCTRL_PORT0, 0)
        # 打开流播放
        if pdll.PlayM4_OpenStream(PLAYCTRL_PORT0, pBuffer, dwBufSize, 1024 * 4096):
            # pdll.PlayM4_SetDecodeEngineEx(PLAYCTRL_PORT, 1)

            # 注册解码回调函数
            global decodeCallback
            decodeCallback = DECCBFUN(DecodeCallback0)
            
            # 解码回调
            pdll.PlayM4_SetDecCallBack(PLAYCTRL_PORT0, decodeCallback)
            # 开始播放
            pdll.PlayM4_Play(PLAYCTRL_PORT0, None)

        else:
            print("Open Stream Failed!", pdll.PlayM4_GetLastError(PLAYCTRL_PORT0))

    elif dwDataType == NET_DVR_STREAMDATA:
        pdll.PlayM4_InputData(PLAYCTRL_PORT0, pBuffer, dwBufSize)

def RealDataCallBack1(lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
    if dwDataType == NET_DVR_SYSHEAD:
        # 设置流播放模式
        pdll.PlayM4_SetStreamOpenMode(PLAYCTRL_PORT1, 0)
        # 打开流播放
        if pdll.PlayM4_OpenStream(PLAYCTRL_PORT1, pBuffer, dwBufSize, 1024 * 4096):
            # pdll.PlayM4_SetDecodeEngineEx(PLAYCTRL_PORT, 1)

            # 注册解码回调函数
            global decodeCallback
            decodeCallback = DECCBFUN(DecodeCallback1)
            
            # 解码回调
            pdll.PlayM4_SetDecCallBack(PLAYCTRL_PORT1, decodeCallback)
            # 开始播放
            pdll.PlayM4_Play(PLAYCTRL_PORT1, None)

        else:
            print("Open Stream Failed!", pdll.PlayM4_GetLastError(PLAYCTRL_PORT1))

    elif dwDataType == NET_DVR_STREAMDATA:
        pdll.PlayM4_InputData(PLAYCTRL_PORT1, pBuffer, dwBufSize)

def RealDataCallBack2(lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
    if dwDataType == NET_DVR_SYSHEAD:
        # 设置流播放模式
        pdll.PlayM4_SetStreamOpenMode(PLAYCTRL_PORT2, 0)
        # 打开流播放
        if pdll.PlayM4_OpenStream(PLAYCTRL_PORT2, pBuffer, dwBufSize, 1024 * 4096):
            # pdll.PlayM4_SetDecodeEngineEx(PLAYCTRL_PORT, 1)

            # 注册解码回调函数
            global decodeCallback
            decodeCallback = DECCBFUN(DecodeCallback2)
            
            # 解码回调
            pdll.PlayM4_SetDecCallBack(PLAYCTRL_PORT2, decodeCallback)
            # 开始播放
            pdll.PlayM4_Play(PLAYCTRL_PORT2, None)

        else:
            print("Open Stream Failed!", pdll.PlayM4_GetLastError(PLAYCTRL_PORT2))
            
    elif dwDataType == NET_DVR_STREAMDATA:
        pdll.PlayM4_InputData(PLAYCTRL_PORT2, pBuffer, dwBufSize)

def RealDataCallBack3(lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
    if dwDataType == NET_DVR_SYSHEAD:
        # 设置流播放模式
        pdll.PlayM4_SetStreamOpenMode(PLAYCTRL_PORT3, 0)
        # 打开流播放
        if pdll.PlayM4_OpenStream(PLAYCTRL_PORT3, pBuffer, dwBufSize, 1024 * 4096):
            # pdll.PlayM4_SetDecodeEngineEx(PLAYCTRL_PORT, 1)

            # 注册解码回调函数
            global decodeCallback
            decodeCallback = DECCBFUN(DecodeCallback3)
            
            # 解码回调
            pdll.PlayM4_SetDecCallBack(PLAYCTRL_PORT3, decodeCallback)
            # 开始播放
            pdll.PlayM4_Play(PLAYCTRL_PORT3, None)

        else:
            print("Open Stream Failed!", pdll.PlayM4_GetLastError(PLAYCTRL_PORT3))
            
    elif dwDataType == NET_DVR_STREAMDATA:
        pdll.PlayM4_InputData(PLAYCTRL_PORT3, pBuffer, dwBufSize)

port_dict= {
    0:PLAYCTRL_PORT0,
    1:PLAYCTRL_PORT1,
    2:PLAYCTRL_PORT2,
    3:PLAYCTRL_PORT3
}

rdcb_dict= {
    0:RealDataCallBack0,
    1:RealDataCallBack1,
    2:RealDataCallBack2,
    3:RealDataCallBack3
}



class Hik_Camera(object):
    def __init__(self,
        DeviceIp = '192.168.1.65',
        DevicePort = 8000,
        DeviceUserName = 'admin',
        DevicePassword = 'hk888888',
        cam_id = 0
        ):
    
        self.DeviceIp = DeviceIp    # 设备连接信息
        self.DevicePort = DevicePort
        self.DeviceUserName = DeviceUserName
        self.DevicePassword = DevicePassword

        self.realDataCallback = REALDATACALLBACK(rdcb_dict.get(cam_id,RealDataCallBack0))

        self.PLAYCTRL_PORT = port_dict.get(cam_id,PLAYCTRL_PORT0)
        self.cam_id = cam_id
        self.img_dict = img_dict
        self.img=None

    def grabbed(self):
        self.img_dict['0']=q0
        self.img_dict['1']=q1
        self.img_dict['2']=q2
        self.img_dict['3']=q3
        return True if self.img_dict.get(str(self.cam_id),q0) is not None else False

    def getitem(self):
        if self.grabbed():
            self.img = self.img_dict.get(str(self.cam_id),q0)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_YUV2BGR_YV12)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        return self.img


    # 通过PlatCtrl获取未使用的通道号
    def GetPort(self):
        return pdll.PlayM4_GetPort(byref(self.PLAYCTRL_PORT))


    # 设备登录
    def LoginDevice(self):
        DeviceInfo = NET_DVR_DEVICEINFO_V30()
        self.lUserId = dll.NET_DVR_Login_V30(bytes(self.DeviceIp, 'utf-8'), self.DevicePort, bytes(self.DeviceUserName, 'utf-8')
                                        , bytes(self.DevicePassword, 'utf-8'), byref(DeviceInfo))

        return self.lUserId, DeviceInfo


    # 调用摄像头
    def Camera(self):
        # 预览参数定义
        previewInfo = NET_DVR_PREVIEWINFO()
        previewInfo.hPlayWnd = 0
        previewInfo.lChannel = 1  # 通道号
        previewInfo.dwStreamType = 0  # 主码流
        previewInfo.dwLinkMode = 0  # TCP
        previewInfo.bBlocked = 1  # 阻塞取流
        # import sys
        # from PyQt5.QtWidgets import QApplication,QWidget
        # app = QApplication(sys.argv)
        # w = QWidget()
        # w.resize(500, 300)
        handle = dll.NET_DVR_RealPlay_V40(self.lUserId, byref(previewInfo), self.realDataCallback, None)
        #handle = dll.NET_DVR_RealPlay_V40(lUserId, byref(previewInfo), None, None)
        if handle == -1:
            print("real play handle error!",dll.NET_DVR_GetLastError())

        return handle


    # SDK获取图像部分
    def Get_Stream(self):

        self.GetPort()
        print("PLAYCTRL_PORT: ",self.PLAYCTRL_PORT)
        # 初始化HCNetSDK库
        InitHCNetSDK(dll)

        # 登陆设备
        (lUserId, deviceInfo) = self.LoginDevice()
        print("UserId:",lUserId)
        if lUserId < 0:
            print('Login device fail, error code is:', dll.NET_DVR_GetLastError())
            dll.NET_DVR_Cleanup()
            exit()

        handle = self.Camera()
        # while True:
        #     print("aaaa")
        #     time.sleep(1)
        # res = dll.NET_DVR_SetRealDataCallBack(handle,realDataCallback,lUserId)
        # print("res",res)
        sys.stdin = os.fdopen(0)
        input()

    def start(self):
        self.thread = threading.Thread(target=self.Get_Stream,daemon=True)
        self.thread.start()
        return self

    def Free_Resources(self):
        # 登出设备
        dll.NET_DVR_Logout(self.lUserId)

        # 释放资源
        dll.NET_DVR_Cleanup()

# 图像转码
def Play():
    while True:
        if q0 is not None:
            img = q0

            t = time.time()

            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YV12)

            #img = cv2.undistort(img, mtx, dist, None, mtx)

            #img=cv2.resize(img,(0,0),fx=5.0,fy=5.0)

            #print("cvtColor:", time.time() - t)

            cv2.namedWindow("video", cv2.WINDOW_NORMAL)
            cv2.imshow("video", img)

            # 按下Esc中断串流
            key = cv2.waitKey(1) & 0xff
            if key == 27:  # 27 is the Esc Key
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    p1 = threading.Thread(target=Start)
    p2 = threading.Thread(target=Play)
    p1.start()
    p2.start()
