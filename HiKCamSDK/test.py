from Camera import *

if __name__ == '__main__':
    # 加载库
    pdll = LoadPlayCtrlSDK()
    dll = LoadHCNetSDK()

    # 获取端口号
    GetPort(pdll)

    # 初始化HCNetSDK库
    InitHCNetSDK(dll)

    # 登陆设备
    (lUserId, deviceInfo) = LoginDevice(dll)

    if lUserId < 0:
        print('Login device fail, error code is:', dll.NET_DVR_GetLastError())
        dll.NET_DVR_Cleanup()
        exit()

    # 注册码流回调函数
    realDataCallback = REALDATACALLBACK(RealDataCallBack)

    handle = Camera(lUserId, dll, realDataCallback)


    input()

    # 登出设备
    dll.NET_DVR_Logout(lUserId)

    # 释放资源
    dll.NET_DVR_Cleanup()