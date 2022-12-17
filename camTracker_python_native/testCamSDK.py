import logging
from camNativeLib.camNativeSDK import *

sdk = HKSdkApi()
sdk.add_dll()
sdk.NET_DVR_Init()
# sdk.NET_DVR_Login_V30()
sdk.NET_DVR_Login_V40()
sdk.getPTZ()
sdk.control(PAN_LEFT, PTZ_CONTROL_STOP, 1)
sdk.control(PAN_LEFT, PTZ_CONTROL_STOP, 1)
# sdk.NET_DVR_Logout()
