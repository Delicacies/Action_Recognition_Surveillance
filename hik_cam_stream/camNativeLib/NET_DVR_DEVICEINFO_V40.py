from camNativeLib.LPNET_DVR_DEVICEINFO_V30 import *


class NET_DVR_DEVICEINFO_V40(ctypes.Structure):
    _fields_ = [
        ("struDeviceV30", LPNET_DVR_DEVICEINFO_V30),
        ("bySupportLock", ctypes.c_byte),
        ("byRetryLoginTime", ctypes.c_byte),
        ("byPasswordLevel", ctypes.c_byte),
        ("byRes1", ctypes.c_byte),
        ("dwSurplusLockTime", ctypes.c_uint16),
        ("byRes2", ctypes.c_byte * 256)
    ]
