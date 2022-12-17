import ctypes

class NET_DVR_PTZPOS(ctypes.Structure):
    _fields_=[
        ("wAction",ctypes.c_short),
        ("wPanPos",ctypes.c_short),
        ("wTiltPos",ctypes.c_short),
        ("wZoomPos",ctypes.c_short)
    ]