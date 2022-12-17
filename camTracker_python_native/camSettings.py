import os
from pathlib import Path
import platform

FILE = Path(__file__)
ROOT = FILE.parents[0]

# use for call by my_multi_thread.py 
if platform.system() == "Windows":
    nativeSdkPath = str(ROOT) + "/lib"
else:
    nativeSdkPath = str(ROOT) + "/lib_linux"
