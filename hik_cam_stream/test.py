import time

from camNativeLib.camNativeSDK import *
from camNativeLib.camEnum import *

sdk = HKSdkApi()
sdk.add_dll()
sdk.NET_DVR_Init()
sdk.NET_DVR_Login_V40()
# sdk.getPTZ()
# sdk.control(PAN_LEFT, PTZ_CONTROL_START)
# time.sleep(5)
# sdk.control(PAN_LEFT, PTZ_CONTROL_STOP)
sdk.NET_DVR_PTZControl_Other(UP_LEFT, PTZ_CONTROL_START)
time.sleep(0.5)
sdk.NET_DVR_PTZControl_Other(UP_LEFT, PTZ_CONTROL_STOP)
# sdk.NET_DVR_PTZControl_Other(ZOOM_IN, PTZ_CONTROL_START)
# time.sleep(1.5)
# sdk.NET_DVR_PTZControl_Other(PAN_AUTO, PTZ_CONTROL_STOP)
sdk.NET_DVR_Logout()
