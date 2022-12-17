import cv2
import configparser
import multiprocessing as mp
from image_input import image_inference
from websocket import websocket_process, websocket_process2

num_camera = 4
list_Q = [[mp.Queue(maxsize=3) for _ in range(3)] for _ in range(num_camera)]
def shower(q, camera_index):
    print("Start to capture and save video of camera {}...".format(camera_index))
    while True:
        img = q.get()[0]  # @yjy 1122
        cv2.namedWindow("camera {}".format(camera_index))
        cv2.imshow("camera {}".format(camera_index), img)
        cv2.waitKey(1)

def consumer(q, img_dict, camera_id):
    # q = list_Q[camera_id]  # camera_id = 0, 1, 2, 3
    while True:
        frame = q.get()
        img_dict[camera_id]['img'] = frame[0]
        img_dict[camera_id]['behavior'] = frame[1]
        img_dict[camera_id]['num_person'] = frame[2]

URLS = [
    "rtsp://admin:hk888888@192.168.1.64/Streaming/Channels/1",
    "rtsp://admin:hk888888@192.168.1.65/Streaming/Channels/1",
    "rtsp://admin:hk888888@192.168.1.66/Streaming/Channels/1",
    "rtsp://admin:hk888888@192.168.1.67/Streaming/Channels/1",
]

def readConfig(path="Utils/camera_config.ini"):
    cf = configparser.ConfigParser()
    cf.read(path)
    open_camera = cf.get("select_camera", "ids").split(",")
    open_camera = list(map(int, open_camera))
    camera_cfgs = []
    for i,id in enumerate([0,1,2,3]):
        cfg = []
        items = cf.items("camera_"+str(id))
        for it in items:
            if it[0] in ["ip", "user", "password", "device"]:
                cfg.append(it[1])
            elif it[0] in ["camera_no", "num_frame", "detection_input_size"]:
                cfg.append(int(it[1]))
            elif it[0] in ["input_down_scale"]:
                cfg.append(float(it[1]))
            elif it[0] in ["save_image", "danger_area", "cross_border", "detect_clothes", "detect_helmet", "detect_smoke", "detect_phone", "fall_down","use_sdk"]:
                if it[1] in ["True", "true"]:
                    b = True
                elif it[1] in ["False", "false"]:
                    b = False
                cfg.append(b)
        camera_cfgs.append(cfg)
    return open_camera, camera_cfgs

def run():
    open_camera, camera_cfgs = readConfig("Utils/camera_config.ini")
    # ['192.168.1.64', 'admin', 'hk888888', 0, 0.7, 5, False, True, False, True, True, True, True, True, 'cuda', 672]
    print(open_camera)
    print(camera_cfgs)
    # id, camera_ip, input_down_scale, num_frame, save_video, danger_area, cross_border
    # camera_cfgs = ['192.168.1.64', 'admin', 'hk888888', 0, 0.7, 5, False, True, False, True, True, True, True, True, 'cuda', 672),
    #                (0,       1, 1.0, 5, False, True, False, True, True, True, True, True, 'cuda', 672)]
    # open_camera = [0,1]  # index for opened camera
    nloops = len(open_camera)
    processes = []
    m = mp.Manager()
    img_dict = [m.dict() for _ in range(4)]
    for i in range(nloops):
        camera_cfg = camera_cfgs[open_camera[i]]
        camera_no = camera_cfg[3]
        q_list = list_Q[camera_no]
        # if i == 0:
        camera_ip = f"rtsp://{camera_cfg[1]}:{camera_cfg[2]}@{camera_cfg[0]}/Streaming/Channels/1"
        # else:
        #     camera_ip = 2  # test for usb camera
        t1 = mp.Process(target=image_inference, args=(q_list,     # q1, q2, q3
                                                  camera_ip,  # camera_ip
                                                  camera_no,  # camera_no
                                                  camera_cfg[4],  # input_down_scale
                                                  camera_cfg[5],  # num_frame
                                                  camera_cfg[6],  # save_image
                                                  camera_cfg[7],  # danger_area
                                                  camera_cfg[8],  # cross_border
                                                  camera_cfg[9],  # detect_clothes
                                                  camera_cfg[10], # detect_helmet
                                                  camera_cfg[11], # detect_smoke
                                                  camera_cfg[12], # detect_phone
                                                  camera_cfg[13], # fall_down
                                                  camera_cfg[14], # device
                                                  camera_cfg[15], # detection_input_size
                                                  camera_cfg[16], # use_sdk
                                                  ))
        s1 = mp.Process(target=consumer, args=(q_list[0], img_dict, camera_no, )) # get queue data
        t1.daemon = True
        s1.daemon = True
        # s1 = mp.Process(target=shower, args=(q_list[0], camera_cfg[1])) # show queue data
        processes.append(t1)
        processes.append(s1)
    show_id = mp.Value('i', camera_cfgs[open_camera[0]][3])
    view_all = mp.Value('i', 1)  # view_all.value = 0: view one camera; 1: view all camera
    w1 = mp.Process(target=websocket_process, args=(img_dict, show_id, view_all))
    w2 = mp.Process(target=websocket_process2, args=(img_dict, list_Q, show_id, camera_cfgs, view_all))
    w1.daemon = True
    w2.daemon = True
    processes.append(w1)
    processes.append(w2)

    [process.start() for process in processes]
    [process.join() for process in processes]

    # pool.map_async(multithread_run, (img_dict, [camera_cfgs[i] for i in open_camera]))  # 异步，(要调用的目标,(传递给目标的参数元组,))
    # pool.close()  # 关闭进程池，关闭后po不再接受新的请求
    # pool.join()  # 等待po中所有子进程执行完成

if __name__ == "__main__":
    run()
