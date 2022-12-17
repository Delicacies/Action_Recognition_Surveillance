import ctypes


class NET_DVR_USER_LOGIN_INFO(ctypes.Structure):
    _fields_ = [
        ("sDeviceAddress", ctypes.c_byte * 129),
        ("byUseTransport", ctypes.c_byte),
        ("wPort", ctypes.c_ushort),
        ("sUserName", ctypes.c_byte * 64),
        ("sPassword", ctypes.c_byte * 64),
        ("pUser", ctypes.c_char_p),
        ("bUseAsynLogin", ctypes.c_uint16),
        ("byRes2", ctypes.c_byte * 256)
    ]
