import os
import cv2
import math
import time
import torch
import argparse
import numpy as np
import queue as q
import PIL

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single
from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

from yolov5.yolov5Loader import Yolov5Detector
from yolov5.yolov5PersonLoader import Yolov5PersonDetector
from yolov5.playingCellPhoneUtils import getArmKneeRectsOnePerson, nearestTrackID
from Utils.yolov5Utils import get_person_crops
from Utils.DangerArea import judgeInAreaOnePerson, drawregion, getRegionFrame

from HiKCamSDK.Camera_Class import *

source = "rtsp://admin:hk888888@192.168.1.64/Streaming/Channels/1"
# source = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def kpt2bbox(kpt, ex=10):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

# 图像上显示中文
def addTextPIL(frame, text, position, color=(255,0,0), size=30):
    pilimg = PIL.Image.fromarray(frame) # RGB
    draw = PIL.ImageDraw.Draw(pilimg)
    font = PIL.ImageFont.truetype("Utils/font/simheittf.ttf", size, encoding="utf-8")
    draw.text(position, text, color, font=font)
    return np.array(pilimg)

def image_inference(q_list,
                    camera_ip=source,       # camera_id
                    camera_no=0,
                    input_down_scale=1.0,
                    num_frame=5,         # num of continuous detected frame 
                    save_image=True,
                    danger_area=True,  # 闯入禁区
                    cross_border=False, # 违法越界
                    detect_clothes=True,
                    detect_helmet=True,
                    detect_smoke=True,
                    detect_phone=True,
                    fall_down=True,
                    device='cuda',
                    detection_input_size=672,
                    use_sdk=False):
    print(device)
    pose_input_size = '224x192'
    pose_backbone = 'resnet50'
    save_out = 'video/0.mp4'
    if camera_no in [0, 1]:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    region_points = []
    detect_classes = []
    show_detect_classes = [1, 2, 3]
    countDict = {"No_helmet":0, "Invalid_clothes":0, "Playing_phone":0, \
            "Smoking":0, "Fall_down":0, "Dangerous_zoom":0}  # 连续危险危险帧数
    last = {"No_helmet": -100, "Invalid_clothes": -100, "Playing_phone": -100, 
            "Smoking": -100, "Fall_down": -100, "Dangerous_zoom": -100, "Cross_border":-100} # 上一次危险行为帧数
    if detect_helmet:
        detect_classes.append(1)
    if detect_phone:
        detect_classes.append(2)
    if detect_smoke:
        detect_classes.append(3)
    detect_helmet_phone_smoke = True if len(detect_classes)>0 else False
    # Load Camera
    # print(camera_ip[22:34])
    # print(camera_ip[7:12])
    # print(camera_ip[13:21])
    if use_sdk:
        cam = Hik_Camera(
            DeviceIp = camera_ip[22:34],  # ip
            DevicePort = 8000,
            DeviceUserName = camera_ip[7:12], # admin
            DevicePassword = camera_ip[13:21],  #'hk888888'
            cam_id = camera_no).start() 
        input_size = (1280,720)
    else:
        cam_source = camera_ip
        if type(cam_source) is str and os.path.isfile(cam_source):
            # Use loader thread with Q for video file.
            cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
        else:
            # Use normal thread loader for webcam.
            cam = CamLoader(cam_source, preprocess=preproc).start()
        input_size = cam.frame_size
    print("input_size:", input_size)  # 640*480  1920*1080
    input_size = (int(input_size[0]*input_down_scale), int(input_size[1]*input_down_scale))

    # YOLOV5 Person Detector
    detect_model = Yolov5PersonDetector(weights='weights/yolov5s.pt', 
                                        imgsz=[detection_input_size], 
                                        conf_thres=0.5, 
                                        iou_thres=0.45,
                                        device="cuda", # device
                                        view_img=False, 
                                        half=False, 
                                        view_car=False)

    # YOLOv5 Detector
    if detect_helmet_phone_smoke:
        # yolov5_helmet = Yolov5Detector(weights='output/exp_helmet/weights/best.pt', imgsz=[200], conf_thres=0.7, line_thickness=2, half=True)
        # yolov5_phone = Yolov5Detector(weights='output/exp18_phonesmoke_v5m/weights/best.pt', imgsz=[520], conf_thres=0.25, half=False, phone_smoke_filter=False)
        yolov5_helmet_phone_smoke = Yolov5Detector(weights='/output/exp33/weights/best.pt',
                                                imgsz=[320], 
                                                conf_thres=0.5,
                                                device="cuda", # device
                                                classes=detect_classes, 
                                                hide_conf=True, half=False, 
                                                phone_smoke_filter=False)

    # POSE MODEL.
    inp_pose = pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(pose_backbone, inp_pose[0], inp_pose[1], device="cuda")  # device

    # Tracker.
    max_age = 30  # 30 已经命中的tracker，连续max_age没有命中目标的tracker会被删除
    tracker = Tracker(max_iou_distance=0.7, max_age=max_age, n_init=1)   # max_iou_distance越大越不容易丢失 @yjy # 连续n_init帧命中目标，暂定的tracker才会变为confired

    # Actions Estimate.
    action_model = TSSTG(device="cuda") # device

    # save video
    # i = 0
    # if save_video:
    #     new_save_out = save_out
    #     while os.path.exists(new_save_out):
    #         i += 1
    #         new_save_out = save_out[:-5]+str(i)+'.mp4'
    #     codec = cv2.VideoWriter_fourcc(*'mp4v')
    #     writer = cv2.VideoWriter(new_save_out, codec, 20, (input_size[0], input_size[1]))
    f = 0
    fps_time = 0
    save_duration = 60
    action_index = 0
    last_behavior_text = ""
    while True:
        # print(cam.grabbed())
        # print("aaa",cam.img_dict.get('0'))
        if cam.grabbed():
            f += 1
            frame = cam.getitem()  # RGB HWC
            # 可用于前端交互字典: 摄像头ip地址, 帧数, 是否佩戴安全帽, 非法穿戴工装, 打电话, 吸烟, 跌倒, 闯入危险区域, 人数, 附加信息
            resDict = {"Camera_id":source, "Frame":f, "No_helmet":False, "Invalid_clothes":False, "Playing_phone":False, \
                "Smoking":False, "Fall_down":False, "Dangerous_zoom":False, "Cross_border":False, "Text":"All Safe"}
            num_person = 0
            invalid_id = [] # invalid person id
            dangerArea_id = []
            fallDown_id = []
            work_clothes_id= []
            no_helmet_id = []
            phone_id = []
            smoke_id = []

            # update selected function
            if not q_list[2].empty():
                selected_actions = q_list[2].get()
                danger_area = True if "Dangerous_zoom" in selected_actions else False
                detect_clothes = True if "Invalid_clothes" in selected_actions else False
                detect_helmet = True if "No_helmet" in selected_actions else False
                detect_smoke = True if "Smoking" in selected_actions else False
                detect_phone = True if "Playing_phone" in selected_actions else False
                fall_down = True if "Fall_down" in selected_actions else False
                cross_border = True if "Cross_border" in selected_actions else False
                detect_helmet_phone_smoke = True if detect_helmet or detect_smoke or detect_phone else False
                show_detect_classes = []
                if detect_helmet:
                    show_detect_classes.append(1)
                if detect_phone:
                    show_detect_classes.append(2)
                if detect_smoke:
                    show_detect_classes.append(3)

            if frame.shape != (720, 1280, 3):
                frame = cv2.resize(frame, input_size)
            if input_down_scale != 1.0:
                frame = cv2.resize(frame, input_size) # 480*640*3
            # read dangerous region points
            # if danger_area or cross_border:
            if not q_list[1].empty():
                choose_points = q_list[1].get()
                if choose_points != "clear_region":
                    temp_list = [[choose_points[i]['x'], choose_points[i]['y']] for i in range(len(choose_points))]
                    region_points = [[p[0]*frame.shape[1], p[1]*frame.shape[0]] for p in temp_list]
                else:
                    region_points = []

            # YOLOv5 detect person
            # print("---------yolov5 starts to detect first------------")
            detect_model.frame_process(frame=frame)
            frame, pred = detect_model.detect()  # pred: list of tensor[n*6: xyxy,conf, cls]
            if(len(pred)):
                detected = torch.tensor(pred).to('cpu')
                num_person = len(pred)
            else:
                detected = None
            # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
            tracker.predict()
            # Merge two source of predicted bbox together.
            for track in tracker.tracks:
                # det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32) # for Tiny Yolov3:clas_conf, bbox_conf, label
                det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0]], dtype=torch.float32) # for YOLOv5
                # detected = torch.cat([detected, det], dim=0) if detected is not None else det  # 多人(有非极大值抑制)
                detected = detected if detected is not None else det  # 单人 + 跟踪
            
            # 判断是否玩手机或吸烟
            body_rects = []  # body of persons to draw rectangle
            detections = []  # List of Detections object for tracking.
            if detected is not None:
                # Predict skeleton pose of each bboxs.
                poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4], ignore=0.003)

                # Yolov5 detect
                if(detect_helmet_phone_smoke and len(detected)): 
                    # print("---------yolov5 starts to detect second------------")
                    inps, new_rects = get_person_crops(frame, detected[:, 0:4], 
                                                        ignore_threshold=0.0, 
                                                        expand_ratio=0.05, 
                                                        cut_legs=False)  # RGB HWC
                    if(len(inps)):
                        # dataset = yolov5_helmet.dataOnPerson(imgs=inps, ori_frame=frame, new_rects=new_rects)  # RGB
                        # frame, helmet_res = yolov5_helmet.detectOnPerson(dataset=dataset, frame=frame)  # BGR & annotated
                        # frame, phone_res = yolov5_phone.detectOnPerson(dataset=dataset, frame=frame) # RGB, [[[xyxy,conf,cls]]]
                        # Unify Detector
                        dataset = yolov5_helmet_phone_smoke.dataOnPerson(imgs=inps, 
                                                                        ori_frame=frame, 
                                                                        new_rects=new_rects)  # RGB
                        frame, det_res = yolov5_helmet_phone_smoke.detectOnPerson(dataset=dataset, 
                                                                            frame=frame, 
                                                                            poses=poses, 
                                                                            show_detect_classes=show_detect_classes)
                        if len(det_res):
                            classes = []
                            for det in det_res: # person
                                for d in det: # object
                                    classes.append(d[-1])
                            countDict["No_helmet"] = 0 if 1 not in classes else countDict["No_helmet"] + 1
                            countDict["Playing_phone"] = 0 if 2 not in classes else countDict["Playing_phone"] + 1
                            countDict["Smoking"] = 0 if 3 not in classes else countDict["Smoking"] + 1
                            for i,det in enumerate(det_res):  # 每一个人
                                track_id_new = 0
                                for d in det: # 每一个物体
                                    track_id_new = nearestTrackID(d[:4], tracker) # 查询检测框属于哪个track_id
                                    if d[-1] == 1 and countDict["No_helmet"] > num_frame and detect_helmet:
                                        resDict["No_helmet"] = True
                                        if track_id_new not in no_helmet_id:
                                            no_helmet_id.append(track_id_new)
                                    if d[-1] == 2 and countDict["Playing_phone"] > num_frame and detect_phone:
                                        resDict["Playing_phone"] = True
                                        if track_id_new not in phone_id:
                                            phone_id.append(track_id_new)
                                    if d[-1] == 3 and countDict["Smoking"] > num_frame and detect_smoke:
                                        resDict["Smoking"] = True
                                        if track_id_new not in smoke_id:
                                            smoke_id.append(track_id_new)
                                if resDict["No_helmet"] or resDict["Playing_phone"] or resDict["Smoking"]:
                                    invalid_id.append(track_id_new)
                        else:
                            countDict["No_helmet"] = 0
                            countDict["Playing_phone"] = 0
                            countDict["Smoking"] = 0
                
                # Create Detections object. 对象
                detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                        np.concatenate((ps['keypoints'].numpy(),
                                                        ps['kp_score'].numpy()), axis=1),
                                        ps['kp_score'].mean().numpy()) for ps in poses]

            # Update tracks by matching each track information of current and previous frame or
            # create a new track if no matched.
            tracker.update(detections)

            # Predict Actions of each track.
            has_invalid_clothes = False # 当前帧是否存在短袖
            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                center_bbox = track.get_center().astype(int)

                if fall_down:  # 是否动作识别
                    action = '站立..'
                    clr_text = (0, 255, 0)
                    # Use 30 frames time-steps to prediction.
                    if len(track.keypoints_list) == 20: # 30
                        pts = np.array(track.keypoints_list, dtype=np.float32)
                        out = action_model.predict(pts, frame.shape[:2])
                        action_name = action_model.class_names[out[0].argmax()]
                        action = '{}'.format(action_name)
                        if action_name in ['跌倒'] and out[0].max() > 0.3:  # @yjy
                            clr_text = (255, 0, 0)
                            resDict["Fall_down"] = True
                            fallDown_id.append(track_id)

                # 判断进入危险区域
                if len(region_points) >= 3:
                    if danger_area:  # 闯入禁区
                        if judgeInAreaOnePerson(region=region_points, det=bbox):
                            dangerArea_id.append(track_id)
                    elif cross_border:    # 违法越界
                        if not judgeInAreaOnePerson(region=region_points, det=bbox):
                            dangerArea_id.append(track_id)

                
                # 人体躯干手臂部分，检测工装
                if detect_clothes:
                    body_rects = getArmKneeRectsOnePerson(keypoint_results=track.keypoints_list[-1],
                                                    bbox=bbox, 
                                                    imshape=frame.shape[:2],
                                                    ignore_thresh=0.05, 
                                                    visual_thresh=0.5)
                    if len(body_rects):
                        for br in body_rects:
                            if br[2]>br[0] and br[3]>br[1] and br[0]>0 and br[2]<frame.shape[1] and br[1]>0 and br[3]<frame.shape[0]: #(480,640)
                                skin_area = frame[br[1]:br[3],br[0]:br[2], :]
                                if skin_area is not None:
                                    center = [int((br[3]-br[1])/2), int((br[2]-br[0])/2)]
                                    skin_area_bgr = skin_area[:, :, ::-1]
                                    hsv = cv2.cvtColor(skin_area_bgr, cv2.COLOR_BGR2HSV)
                                    (h,s,v) = cv2.split(hsv)
                                    is_skin = h[center[0],center[1]]>=5 \
                                            and h[center[0],center[1]]<=14 \
                                            and s[center[0],center[1]]>28 \
                                            and v[center[0], center[1]]>50
                                    if is_skin:
                                        has_invalid_clothes = True
                                        color = (255, 0, 0)  # RGB
                                        cv2.circle(skin_area, (center[1],center[0]), 10, (255,0,0))
                                        frame = cv2.rectangle(frame, (br[0],br[1]), (br[2],br[3]), color, 2)
                                        frame = addTextPIL(frame, text="未正确穿戴工服", position=(min(br[0],br[2])+5, min(br[1],br[3])+15),
                                                            color=color, size=30)
                    if countDict["Invalid_clothes"] >= num_frame:  # yjy 1123
                        resDict["Invalid_clothes"] = True
                        if track_id not in work_clothes_id:
                            work_clothes_id.append(track_id)                         
                                                        
                # 人体骨架显示
                if fall_down: # 是否动作识别
                    color_box = (255, 0, 0) if track_id in invalid_id+fallDown_id+dangerArea_id+work_clothes_id else (0, 255, 0)  # Bug RGB
                    if track.time_since_update <= 2:  # 只画出丢失2帧之内的跟踪框 "==0"
                        frame = draw_single(frame, track.keypoints_list[-1])
                        frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_box, 2)  # changed
                        frame = cv2.putText(frame, str(track_id), (center_bbox[0], center_bbox[1]), cv2.FONT_HERSHEY_COMPLEX,
                                            0.8, color_box, 2)
                        frame = addTextPIL(frame, text=action, position=(bbox[0] + 5, bbox[1] + 15), color=clr_text, size=30)

            # 更新连续出现短袖帧数
            if detect_clothes:
                countDict["Invalid_clothes"] = countDict["Invalid_clothes"] + 1 if has_invalid_clothes else 0

            # 绘制危险区域
            if len(region_points):
                frame = drawregion(frame, region=region_points, danger=len(dangerArea_id)>0)
                if len(dangerArea_id)>0:
                    if danger_area:
                        resDict["Dangerous_zoom"] = True
                    elif cross_border:
                        resDict["Cross_border"] = True

            # 危险行为汇总
            resStr = ""
            if detect_helmet and resDict["No_helmet"]:
                resStr += f"未佩戴安全帽,id: {sorted(no_helmet_id)}\n"
            if detect_phone and resDict["Playing_phone"]:
                resStr += f"打电话,id: {sorted(phone_id)}\n"
            if fall_down and resDict["Fall_down"]:
                resStr += f"人员摔倒,id: {sorted(fallDown_id)}\n"
            if detect_smoke and resDict["Smoking"]:
                resStr += f"吸烟,id: {sorted(smoke_id)}\n"
            if danger_area and resDict["Dangerous_zoom"]:
                resStr += f"闯入危险区域,id: {sorted(dangerArea_id)}\n"
            if cross_border and resDict["Cross_border"]:
                resStr += f"违法越界,id: {sorted(dangerArea_id)}\n"
            if detect_clothes and resDict["Invalid_clothes"]:
                resStr += f"未正确穿工服,id: {sorted(work_clothes_id)}\n"

            # 图像上显示危险信息
            frame = addTextPIL(frame, text=resStr, position=(10,50), color=(255,0,0), size=40)

            # Show Frame.
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # RGB to BGR
            # if save_video:
            #     writer.write(frame) # BGR
            # if zoom != 1.0:
            #     frame = cv2.resize(frame, (0, 0), fx=zoom, fy=zoom)
            frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            fps_time = time.time()
            
            # show frame
            # cv2.imshow('frame', frame)
            # key = cv2.waitKey(1)
            # if key & 0xFF == ord('r'):
            #     region_points = getRegionFrame(count=4, frame=frame)
            # elif key & 0xFF == ord('q'):
            #     break

            if resStr != "":
                resStr = f"摄像头 {camera_no}: " + resStr
            resDict["Text"] = resStr.replace("\n", "; ")

            # save image
            if save_image:
                changed = False
                if resDict['No_helmet'] and (f - last['No_helmet']) >= save_duration:
                    last['No_helmet'] = f
                    changed = True
                if resDict['Invalid_clothes'] and (f - last['Invalid_clothes']) >= save_duration:
                    last['Invalid_clothes'] = f
                    changed = True
                if resDict['Playing_phone'] and (f - last['Playing_phone']) >= save_duration:
                    last['Playing_phone'] = f
                    changed = True
                if resDict['Smoking'] and (f - last['Smoking']) >= save_duration:
                    last['Smoking'] = f
                    changed = True
                if resDict['Fall_down'] and (f - last['Fall_down']) >= save_duration:
                    last['Fall_down'] = f
                    changed = True
                if resDict['Dangerous_zoom'] and (f - last['Dangerous_zoom']) >= save_duration:
                    last['Dangerous_zoom'] = f
                    changed = True
                if resDict['Cross_border'] and (f - last['Cross_border']) >= save_duration:
                    last['Cross_border'] = f
                    changed = True

                if changed and resDict['Text'] != "" and resDict['Text'] != last_behavior_text:
                    last_behavior_text = resDict['Text']
                    cv2.imwrite(f"CameraData/camera{camera_no}/事件{action_index}_{resDict['Text']}.jpg", frame)
                    action_index += 1
            
            # 发送数据到前端
            q_list[0].put([frame, resDict, num_person])
            q_list[0].get() if q_list[0].qsize() > 1 else time.sleep(0.01)

    # Clear resource.
    cam.stop() if not use_sdk else cam.Free_Resources()
    # if save_video:
    #     writer.release()
    cv2.destroyAllWindows()

if  __name__ == "__main__":
    my_q = q.Queue()
    image_inference(q=my_q, camera_ip=source)
