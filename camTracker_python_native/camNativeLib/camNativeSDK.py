import logging
import os
import platform
from ctypes import CDLL

import camNativeLib.camPTZ
from camNativeLib.NET_DVR_DEVICEINFO_V40 import *
from camNativeLib.NET_DVR_PTZPOS import *
from camNativeLib.NET_DVR_USER_LOGIN_INFO import *
from camNativeLib.camEnum import *
from camSettings import *


class HKSdkApi:
    def __init__(self):
        self.path = nativeSdkPath
        self.dll_list = []
        # 登陆设备信息
        self.dvrDeviceInfo = None
        # 登陆用户信息
        self.dvrLoginInfo = None
        self.userid = None
        pass

    # 遍历动态链接库目录
    def add_dll(self):
        files = os.listdir(self.path)
        if platform.system() == "Windows":
            for file in files:
                if not os.path.isdir(self.path + file):
                    if file.endswith(".dll"):  # linux 替换成.so
                        self.dll_list.append(self.path + "/" + file)
                else:
                    self.add_dll(self.path + file + "/", self.dll_list)
        else:
            for file in files:
                if not os.path.isdir(self.path + file):
                    if file.endswith(".so"):  # linux 替换成.so
                        self.dll_list.append(self.path + "/" + file)
                else:
                    self.add_dll(self.path + file + "/", self.dll_list)

    # 载入动态链接库
    def callCpp(self, func_name, *args):
        for win_lib in self.dll_list:
            try:
                lib = ctypes.cdll.LoadLibrary(win_lib)
                try:
                    value = eval("lib.%s" % func_name)(*args)
                    print("调用的库：" + win_lib)
                    print("执行成功,返回值：" + str(value))
                    return value
                except:
                    continue
            except:
                print("库文件载入失败：" + win_lib)
                continue
        print("没有找到接口！")
        return False

    # 摄像头用户登陆
    def NET_DVR_Login_V40(self, sDVRIP, wDVRPort, sUserName, sPassword):
        set_overtime = self.callCpp("NET_DVR_SetConnectTime", 5000, 4)  # 设置超时
        if set_overtime:
            logging.info(sDVRIP + ", 设置超时时间成功")
        else:
            error_info = self.callCpp("NET_DVR_GetLastError")
            logging.error(sDVRIP + ", 设置超时错误信息：" + str(error_info))
            return False

        self.dvrLoginInfo = NET_DVR_USER_LOGIN_INFO()
        print(self.dvrLoginInfo)
        # c++传递进去的是byte型数据，需要转成byte型传进去，否则会乱码
        self.dvrLoginInfo.sDeviceAddress = (ctypes.c_byte * 129)(*[ctypes.c_byte(ord(c)) for c in sDVRIP])
        self.dvrLoginInfo.sUserName = (ctypes.c_byte * 64)(*[ctypes.c_byte(ord(c)) for c in sUserName])
        self.dvrLoginInfo.sPassword = (ctypes.c_byte * 64)(*[ctypes.c_byte(ord(c)) for c in sPassword])
        self.dvrLoginInfo.wPort = wDVRPort
        self.dvrLoginInfo.bUseAsynLogin = 0
        self.dvrDeviceInfo = NET_DVR_DEVICEINFO_V40()
        dvrUserLoginRef = ctypes.byref(self.dvrLoginInfo)
        dvrDeviceInfoRef = ctypes.byref(self.dvrDeviceInfo)
        lUserID = self.callCpp("NET_DVR_Login_V40", dvrUserLoginRef, dvrDeviceInfoRef)
        print(sDVRIP + ", 登录结果：" + str(lUserID))
        print(self.dvrDeviceInfo.byRes1)
        # dvrDeviceInfoRef.struDeviceV30.byStartChan
        if lUserID == -1:  # -1表示失败，其他值表示返回的用户ID值。
            error_info = self.callCpp("NET_DVR_GetLastError")
            print(sDVRIP + ", 登录错误信息：" + str(error_info))
        self.userid = lUserID
        return lUserID

    def NET_DVR_Login_V30(self, sDVRIP="192.168.1.64", wDVRPort=8000, sUserName="admin", sPassword="hk888888"):
        set_overtime = self.callCpp("NET_DVR_SetConnectTime", 5000, 4)  # 设置超时
        if set_overtime:
            logging.info(sDVRIP + ", 设置超时时间成功")
        else:
            error_info = self.callCpp("NET_DVR_GetLastError")
            logging.error(sDVRIP + ", 设置超时错误信息：" + str(error_info))
            return False

        # c++传递进去的是byte型数据，需要转成byte型传进去，否则会乱码
        sDVRIP_bytes = bytes(sDVRIP, "ascii")
        sUserName = bytes(sUserName, "ascii")
        sPassword = bytes(sPassword, "ascii")
        DeviceInfo = LPNET_DVR_DEVICEINFO_V30()
        DeviceInfoRef = ctypes.byref(DeviceInfo)
        lUserID = self.callCpp("NET_DVR_Login_V30", sDVRIP_bytes, wDVRPort, sUserName, sPassword, DeviceInfoRef)
        logging.info(sDVRIP + ", 登录结果：" + str(lUserID))
        if lUserID == -1:  # -1表示失败，其他值表示返回的用户ID值。
            error_info = self.callCpp("NET_DVR_GetLastError")
            logging.error(sDVRIP + ", 登录错误信息：" + str(error_info))
        self.userid = lUserID
        print(self.userid)
        return lUserID

    # 登出摄像头
    def NET_DVR_Logout(self):
        res = self.callCpp("NET_DVR_Logout", self.userid)
        if res == -1:  # -1表示失败，其他值表示返回的用户ID值。
            error_info = self.callCpp("NET_DVR_GetLastError")
            logging.error(self.userid + ", 登出错误信息：" + str(error_info))
        print(f"{self.userid} 登出")
        return res

    def NET_DVR_GetDVRConfig(self, lUserID, dwCommand, IChannel, IpOutBuffer, IpBytesReturned):
        IpOutBufferRef = ctypes.byref(IpOutBuffer)
        print(ctypes.sizeof(IpOutBuffer))
        buffSize = ctypes.sizeof(IpOutBuffer)
        IpBytesReturnedRef = ctypes.byref(IpBytesReturned)
        res = self.callCpp("NET_DVR_GetDVRConfig", lUserID, dwCommand, IChannel, IpOutBufferRef, buffSize,
                           IpBytesReturnedRef)
        if res == -1:  # -1表示失败
            error_info = self.callCpp("NET_DVR_GetLastError")
            logging.error(dwCommand + ", 接口调用错误信息：" + str(error_info))
        return res

    # 计算16进制值转10进制值
    def dex2hex(self, hexValue):
        res = hex(hexValue)
        res = int(str(res).replace('0x', ''))
        return res

    def hex2dec(self, decValue):
        res = int('0x' + str(decValue), 16)
        print(res)
        return res

    def control(self, ptzCommand, ptzTrigger, ptzspeed):
        # res = self.callCpp("NET_DVR_PTZControl_Other", self.userid,
        #                    self.dvrDeviceInfo.struDeviceV30.byStartChan, ptzCommand, ptzTrigger)
        res = self.callCpp("NET_DVR_PTZControlWithSpeed_Other", self.userid,
                           self.dvrDeviceInfo.struDeviceV30.byStartChan, ptzCommand, ptzTrigger, ptzspeed)
        print(res)
        if res == -1:  # -1表示失败
            error_info = self.callCpp("NET_DVR_GetLastError")
            logging.error(ptzCommand + "控制云台PTZ错误信息：" + str(error_info))
        return res

    # 获取摄像头PTZ
    def getPTZ(self):
        ptzpos = NET_DVR_PTZPOS()
        resBuff = ctypes.c_buffer(255)
        res = self.NET_DVR_GetDVRConfig(self.userid, NET_DVR_GET_PTZPOS, self.dvrDeviceInfo.struDeviceV30.byStartChan,
                                        ptzpos, resBuff)
        # if res:
        #     print("控制成功")
        # else:
        #     print("控制失败: " + str(self.callCpp("NET_DVR_GetLastError")))
        if res == 0:  # -1表示失败
            error_info = self.callCpp("NET_DVR_GetLastError")
            logging.error("获取PTZ错误信息：" + str(error_info))
            return res
        print(self.dvrDeviceInfo.struDeviceV30.byStartChan)
        print(ptzpos.wPanPos)
        ptz = camNativeLib.camPTZ.CamPTZ()
        ptz.action = self.dex2hex(ptzpos.wAction)
        ptz.pan = self.dex2hex(ptzpos.wPanPos)
        ptz.tilt = self.dex2hex(ptzpos.wTiltPos)
        ptz.zoom = self.dex2hex(ptzpos.wZoomPos)
        return ptz

    # 云台控制操作
    def NET_DVR_PTZControl_Other(self, dwPTZCommand, dwStop, dwSpeed):
        res = self.callCpp("NET_DVR_PTZControlWithSpeed_Other", self.userid,
                           self.dvrDeviceInfo.struDeviceV30.byStartChan, dwPTZCommand, dwStop, dwSpeed)
        if res:
            print("控制成功")
        else:
            print("控制失败: " + str(self.callCpp("NET_DVR_GetLastError")))

    def NET_DVR_Init(self):
        res = self.callCpp("NET_DVR_Init")
        if res:
            print(res)
        else:
            print("Fail" + str(self.callCpp("NET_DVR_GetLastError")))
        return res
