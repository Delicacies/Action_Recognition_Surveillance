import ctypes
import threading
import time

from HikVision_Python_Header import *
from ctypes import *
import cv2
import numpy as np
import cupy as cp
from matplotlib import pyplot as plt



# 设备连接信息
DeviceIp = '192.168.1.65'
DevicePort = 8000
DeviceUserName = 'admin'
DevicePassword = 'hk888888'

# HCNetSDK库文件路径
#HCNetSDK_Path = "/home/zmhh/Action_Recognition/HikVisionCamera/lib1/libhcnetsdk.so"
HCNetSDK_Path = "./libhcnetsdk.so"

# PlatCtrlSDK库文件路径
#PlayCtrlSDK_Path = "/home/zmhh/Action_Recognition/HikVisionCamera/lib1/libPlayCtrl.so"
PlayCtrlSDK_Path = "./libPlayCtrl.so"
#PlayCtrlSDK_Path = r'./lib/libPlayCtrl.so'

# 码流回调数据类型
NET_DVR_SYSHEAD = 1
NET_DVR_STREAMDATA = 2
NET_DVR_AUDIOSTREAMDATA = 3
NET_DVR_PRIVATE_DATA = 112

# 码流回调函数
#REALDATACALLBACK = CFUNCTYPE(None, c_long, c_ulong, POINTER(c_ubyte), c_ulong, c_void_p)
REALDATACALLBACK = CFUNCTYPE(None, c_long, c_ulong, POINTER(c_byte), c_ulong, c_void_p)
# 解码回调函数
#DECCBFUN = CFUNCTYPE(None, c_long, POINTER(c_ubyte), c_long, POINTER(FRAME_INFO), c_long, c_long)
#DECCBFUN = CFUNCTYPE(None, c_int, c_char_p, c_int, POINTER(FRAME_INFO), c_int, c_int)
DECCBFUN = CFUNCTYPE(None, c_int, POINTER(c_ubyte), c_int, POINTER(FRAME_INFO), c_int, c_int)

# 初始化全局变量
PLAYCTRL_PORT = c_int(-1)
#PLAYCTRL_PORT = c_long(-1)


pdll = None
dll = None
decodeCallback = None
q = None

# 加载HCNetSDK库
def LoadHCNetSDK():
    global dll
    #dll = WinDLL(HCNetSDK_Path)
    dll = ctypes.CDLL(HCNetSDK_Path)
    return dll


# 加载PlayCtrl库
def LoadPlayCtrlSDK():
    global pdll
    #pdll = WinDLL(PlayCtrlSDK_Path)
    pdll = ctypes.CDLL(PlayCtrlSDK_Path)
    return pdll


# 初始化HCNetSDK库
def InitHCNetSDK(dll):
    # 初始化DLL
    dll.NET_DVR_Init()
    # 设置设备超时时间
    dll.NET_DVR_SetConnectTime(int(5000), 4) #超时时间5s, 最大重连次数4次
    dll.NET_DVR_SetReconnect(10000,True)     #重连间隔10s


# 通过PlatCtrl获取未使用的通道号
def GetPort(pdll):
    return pdll.PlayM4_GetPort(byref(PLAYCTRL_PORT))


# 设备登录
def LoginDevice(dll):
    DeviceInfo = NET_DVR_DEVICEINFO_V30()
    lUserId = dll.NET_DVR_Login_V30(bytes(DeviceIp, 'utf-8'), DevicePort, bytes(DeviceUserName, 'utf-8')
                                    , bytes(DevicePassword, 'utf-8'), byref(DeviceInfo))

    return lUserId, DeviceInfo


# 定义解码回调函数
def DecodeCallback(nPort, pBuf, nSize, pFrameInfo, nReserved1, nReserved2):
    global q
    #print("pBuf",pBuf)
    print("Frame information:")
    print("pFrameInfo.contents.nType",pFrameInfo.contents.nType)
    print("Frame Width",pFrameInfo.contents.nWidth)
    print("Frame Height",pFrameInfo.contents.nHeight)
    if pFrameInfo.contents.nType == 3:

        t = time.time()

        img = np.frombuffer(bytes(pBuf[0:nSize]), cp.uint8)
        img.resize(((pFrameInfo.contents.nHeight * 3) // 2, pFrameInfo.contents.nWidth))

        q = img
        print("frombuffer:", time.time() - t)


# 定义码流回调函数
def RealDataCallBack(lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
    if dwDataType == NET_DVR_SYSHEAD:
        # 设置流播放模式
        pdll.PlayM4_SetStreamOpenMode(PLAYCTRL_PORT, 0)
        # 打开流播放
        if pdll.PlayM4_OpenStream(PLAYCTRL_PORT, pBuffer, dwBufSize, 1024 * 4096):
            # pdll.PlayM4_SetDecodeEngineEx(PLAYCTRL_PORT, 1)

            # 注册解码回调函数
            global decodeCallback
            decodeCallback = DECCBFUN(DecodeCallback)
            

            # 解码回调
            pdll.PlayM4_SetDecCallBack(PLAYCTRL_PORT, decodeCallback)
            # 开始播放
            pdll.PlayM4_Play(PLAYCTRL_PORT, None)

        else:
            print("Open Stream Failed!", pdll.PlayM4_GetLastError(PLAYCTRL_PORT))

    elif dwDataType == NET_DVR_STREAMDATA:
        pdll.PlayM4_InputData(PLAYCTRL_PORT, pBuffer, dwBufSize)


# 调用摄像头
def Camera(lUserId, dll, realDataCallback):
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
    handle = dll.NET_DVR_RealPlay_V40(lUserId, byref(previewInfo), realDataCallback, None)
    #handle = dll.NET_DVR_RealPlay_V40(lUserId, byref(previewInfo), None, None)
   
    if not handle:
        print("real play handle error!",dll.NET_DVR_GetLastError())

    return handle


# SDK获取图像部分
def Start():


    # 加载库
    pdll = LoadPlayCtrlSDK()
    dll = LoadHCNetSDK()

    # 获取端口号
    GetPort(pdll)
    print("PLAYCTRL_PORT: ",PLAYCTRL_PORT)

    # 初始化HCNetSDK库
    InitHCNetSDK(dll)

    # 登陆设备
    (lUserId, deviceInfo) = LoginDevice(dll)
    print("UserId:",lUserId)
    if lUserId < 0:
        print('Login device fail, error code is:', dll.NET_DVR_GetLastError())
        dll.NET_DVR_Cleanup()
        exit()
    # 注册码流回调函数
    realDataCallback = REALDATACALLBACK(RealDataCallBack)
    
    handle = Camera(lUserId, dll, realDataCallback)
    # res = dll.NET_DVR_SetRealDataCallBack(handle,realDataCallback,lUserId)
    # print("res",res)
    #time.sleep(10)
    input()
    
    # 登出设备
    dll.NET_DVR_Logout(lUserId)

    # 释放资源
    dll.NET_DVR_Cleanup()


# 图像转码与畸变修正
def Play():
    while True:
        if q is not None:
            img = q

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
