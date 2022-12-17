from Camera_Class import *

cam0 = Hik_Camera(
        DeviceIp = '192.168.1.65',
        DevicePort = 8000,
        DeviceUserName = 'admin',
        DevicePassword = 'hk888888',
        cam_id = 0)
#p2 = threading.Thread(target=Play)
cam0 = cam0.start()
while True:
        if cam0.grabbed():
                print(cam0.grabbed())
                #print("aaa",cam0.img_dict.get('0'))
                img = cam0.getitem()
                cv2.imshow('aaa',img)
                key = cv2.waitKey(1)
                
p2.start()

