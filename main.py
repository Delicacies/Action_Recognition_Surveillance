import os
import cv2
import math
import time
import torch
import argparse
import numpy as np

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

from yolov5.yolov5Loader import Yolov5Detector
from yolov5.yolov5PersonLoader import Yolov5PersonDetector
from yolov5.playingCellPhoneUtils import judge_playing_phone, maybe_playing_phone, getBodyRects, getArmRects, getArmKneeRects
from Utils.yolov5Utils import get_person_crops
from Utils.DangerArea import judgeInArea, drawregion, getRegionFrame, readRegion

source = "rtsp://admin:hk888888@192.168.1.64/Streaming/Channels/1"

def preproc(image):
    """preprocess function for CameraLoader.
    """
    # image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def kpt2bbox(kpt, ex=10):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                        help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=720,  # 576 640
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='192x160',
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=True, action='store_true',
                        help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='video/0.mp4',
                        help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on cpu or cuda.')
    par.add_argument('--detect_clothes', default=True, action='store_true',
                        help='Detect worker clothes on arms.')
    par.add_argument('--danger_area', default=True, action='store_true',
                        help='Detect danger area person.')

    args = par.parse_args()
    device = args.device
    region_points = []
    countDict = {"Camera_id":source, "Frame":None, "No_helmet":0, "Invalid_clothes":0, "Playing_phone":0, \
             "Smoking":0, "Fall_down":False, "Dangerous_zoom":0, "Num_person":0, "Text":"All Safe"}
    num_frame = 10 # continuous frame num
    # continuous_frame = [0, 0, 0, 0, 0] # helmet, phone, smoke, clothes, danger_area
    # Load Camera
    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()
    input_size = cam.frame_size
    print("input_size:", input_size)  # 640*480  704*576 1920*1080

    scale = 1.0  # 0.7  # down scale rate
    input_size = (int(input_size[0]*scale), int(input_size[1]*scale)) #(1440, 810)

    # DETECTION MODEL.
    inp_dets = args.detection_input_size

    # YOLOV5 Person Detector
    detect_model = Yolov5PersonDetector(weights='weights/yolov5s.pt', imgsz=[inp_dets], conf_thres=0.5, iou_thres=0.45, view_img=False, half=False, view_car=False)

    # YOLOv5 Detector
    # yolov5_helmet = Yolov5Detector(weights='output/exp_helmet/weights/best.pt', imgsz=[200], conf_thres=0.7, line_thickness=2, half=True)
    # yolov5_phone = Yolov5Detector(weights='output/exp18_phonesmoke_v5m/weights/best.pt', imgsz=[520], conf_thres=0.25, half=False, phone_smoke_filter=False)
    yolov5_helmet_phone_smoke = Yolov5Detector(weights='/output/exp31/weights/best.pt', imgsz=[400], conf_thres=0.4, classes=[1,2,3], hide_conf=True, half=False, phone_smoke_filter=False)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 20  # 20 已经命中的tracker，连续max_age没有命中目标的tracker会被删除
    tracker = Tracker(max_iou_distance=0.4, max_age=max_age, n_init=1)   # 3 @yjy # 连续n_init帧命中目标，暂定的tracker才会变为confired

    # Actions Estimate.
    action_model = TSSTG()

    # read dangerous region points
    if args.danger_area:
        region_points = readRegion()

    # resize_fn = ResizePadding(inp_dets, inp_dets)  # @yjy 1029
    # save video num
    i = 0
    zoom = 1.0/scale #- 0.15  # 0.8  show scale
    outvid = False  
    if args.save_out != '':
        new_save_out = args.save_out
        while os.path.exists(new_save_out):
            i += 1
            new_save_out = args.save_out[:-5]+str(i)+'.mp4'
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(new_save_out, codec, 20, (input_size[0], input_size[1]))
    # time.sleep(0.5) # added
    fps_time = 0
    f = 0
    while cam.grabbed():
        f += 1
        frame = cam.getitem()  # RGB HWC
        # 可用于前端交互字典: 摄像头ip地址, 帧数据, 是否佩戴安全帽, 非法穿戴工装, 打电话, 吸烟, 跌倒, 闯入危险区域, 人数, 附加信息
        resDict = {"Camera_id":source, "Frame":None, "No_helmet":False, "Invalid_clothes":False, "Playing_phone":False, \
             "Smoking":False, "Fall_down":False, "Dangerous_zoom":False, "Num_person":0, "Text":"All Safe"}
        invalid_id = [] # invalid person id
        dangerArea_id = []
        frame = cv2.resize(frame, input_size) # 480*640*3
        # Detect humans bbox in the frame with detector model. Yolov3-Tiny

        # YOLOv5 detect person
        print("---------yolov5 starts to detect first------------")
        detect_model.frame_process(frame=frame)
        frame, pred = detect_model.detect()  # pred: list of tensor[n*6: xyxy,conf, cls]
        if(len(pred)):
            # pred[:, 5] = 1.0  # conf=1.0
            # pr = []
            # for p in pred:
            #     p[5]=1.0
            #     pr.append(p)
            detected = torch.tensor(pred).to('cpu')
            resDict["Num_person"] = len(pred)
            if len(region_points):
                dangerArea_id = judgeInArea(region=region_points, det=pred[:][:4])
        else:
            detected = None

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            # det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32) # for Tiny Yolov3:clas_conf, bbox_conf, label
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0]], dtype=torch.float32) # for YOLOv5
            # detected = torch.cat([detected, det], dim=0) if detected is not None else det  # 多人(有非极大值抑制)
            detected = detected if detected is not None else det  # 单人+跟踪
            # detected = detected if detected is not None else None  # 单人无跟踪
        
        # 判断是否玩手机或吸烟
        is_PlayingPhone_Or_Smoking = False
        body_rects = []  # body of persons to draw rectangle
        detections = []  # List of Detections object for tracking.

        if f%1 == 0:  # detect helmet ignore 2 frames @yjy 1113
            if detected is not None:
                # Predict skeleton pose of each bboxs.
                poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

                # 人体上半身主干区域
                if args.detect_clothes:
                    # body_rects = getBodyRects(keypoint_results=poses, imshape=frame.shape[:2], ignore_thresh=0.05, visual_thresh=0.2)
                    body_rects = getArmKneeRects(keypoint_results=poses, imshape=frame.shape[:2], ignore_thresh=0.05, visual_thresh=0.3)
                '''
                # 根据人体姿态计算手臂弯曲大于120度，再剪裁人体区域照片，检测手机与香烟
                playingPersonMask = maybe_playing_phone(keypoint_results=poses,
                                        imshape=frame.shape,
                                        visual_thread=0.2)
                print(playingPersonMask)
                if(len(playingPersonMask)):  # 防止有人体检测框但没有识别姿态
                    detected_for_Phone = detected.index_select(torch.tensor(0).to('cpu'), torch.tensor(playingPersonMask).to('cpu'))
                    # detected_for_Phone = detected_for_Phone.unsqueeze(0)
                    # print(detected_for_Phone)
                '''
                # Yolov5 detect
                if(len(detected) and f%1 == 0):  # detect helmet ignore 2 frames @yjy
                    print("---------yolov5 starts to detect second------------")
                    inps, new_rects = get_person_crops(frame, detected[:, 0:4], ignore_threshold=0.0, expand_ratio=0.05, cut_legs=False)  # RGB HWC
                    if(len(inps)):
                        # dataset = yolov5_helmet.dataOnPerson(imgs=inps, ori_frame=frame, new_rects=new_rects)  # RGB
                        # frame, helmet_res = yolov5_helmet.detectOnPerson(dataset=dataset, frame=frame)  # BGR & annotated
                        # frame, phone_res = yolov5_phone.detectOnPerson(dataset=dataset, frame=frame) # RGB, [[[xyxy,conf,cls]]]
                        # Unify Detector
                        dataset = yolov5_helmet_phone_smoke.dataOnPerson(imgs=inps, ori_frame=frame, new_rects=new_rects)  # RGB
                        frame, det_res = yolov5_helmet_phone_smoke.detectOnPerson(dataset=dataset, frame=frame, poses=poses)
                        if len(det_res):
                            classes = []
                            for det in det_res: # person
                                for d in det: # object
                                    classes.append(d[-1])
                            countDict["No_helmet"] = 0 if 1 not in classes else countDict["No_helmet"] + 1
                            countDict["Playing_phone"] = 0 if 2 not in classes else countDict["Playing_phone"] + 1
                            countDict["Smoking"] = 0 if 3 not in classes else countDict["Smoking"] + 1
                            for i,det in enumerate(det_res):
                                for d in det:
                                    if d[-1] == 1 and countDict["No_helmet"]>num_frame:
                                        resDict["No_helmet"]=True
                                    if d[-1] == 2 and countDict["Playing_phone"]>num_frame:
                                        resDict["Playing_phone"]=True
                                        countDict["Playing_phone"] += 1
                                    if d[-1] == 3 and countDict["Smoking"]>num_frame:
                                        resDict["Smoking"]=True
                                        countDict["Smoking"] += 1
                                if resDict["No_helmet"] or resDict["Playing_phone"] or resDict["Smoking"]:
                                    invalid_id.append(i)
                                    # bbox = pred[i][:4]
                                    # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)

                '''
                if(len(playingPersonMask) and len(detected_for_Phone[0])): # 人体检测框+姿态成功识别+手臂弯曲大于120度
                    inps_phone, new_rects_phone = get_person_crops(frame, detected_for_Phone[:, 0:4], ignore_threshold=0.02, expand_ratio=0.1)  # RGB HWC
                    if(len(inps_phone)):
                        dataset_phone = yolov5_helmet.dataOnPerson(imgs=inps_phone, ori_frame=frame, new_rects=new_rects_phone)  # RGB
                        frame, phone_res = yolov5_phone.detectOnPerson(dataset=dataset_phone, frame=frame, poses=poses) # RGB, [[[xyxy,conf,cls]]]
                '''
                # 判断是否玩手机或吸烟
                # if(len(phone_res)):
                #     is_PlayingPhone_Or_Smoking, leftArmAngle, rightArmAngle = judge_playing_phone(
                #             detect_results=phone_res[0],
                #             keypoint_results=poses,  # @yjy
                #             imshape=frame.shape,
                #             phone_threshold=0.3,
                #             visual_thread=0.2)
                # print(is_PlayingPhone_Or_Smoking)
                
                # Create Detections object. 对象
                detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                        np.concatenate((ps['keypoints'].numpy(),
                                                        ps['kp_score'].numpy()), axis=1),
                                        ps['kp_score'].mean().numpy()) for ps in poses]

                # VISUALIZE.
                # if args.show_detected:
                #     for bb in detected[:, 0:5]:
                #         pass
                        # 检测器+tracker
                        # frame = cv2.rectangle(frame, (math.floor(bb[0]), math.floor(bb[1])), (math.floor(bb[2]), math.floor(bb[3])), (0, 0, 255), 1)

            # Update tracks by matching each track information of current and previous frame or
            # create a new track if no matched.
            tracker.update(detections)

        # 人体躯干手臂部分，检测工装
        if args.detect_clothes:
            for br in body_rects:
                if br[2]>br[0] and br[3]>br[1] and br[0]>0 and br[2]<frame.shape[1] and br[1]>0 and br[3]<frame.shape[0]: #(480,640)
                    skin_area = frame[br[1]:br[3],br[0]:br[2], :]
                    if skin_area is not None:
                        center = [int((br[3]-br[1])/2), int((br[2]-br[0])/2)]
                        skin_area = skin_area[:, :, ::-1]
                        # cv2.imshow("skin", skin_area)
                        # if cv2.waitKey(1) & 0xFF==ord('q'):
                        #     break
                        skin_area = cv2.cvtColor(skin_area, cv2.COLOR_BGR2HSV)
                        (h,s,v) = cv2.split(skin_area)
                        is_skin = h[center[0],center[1]]>3 and h[center[0],center[1]]<=25 and s[center[0],center[1]]>28 and v[center[0], center[1]]>50
                        text = "No Work Clothes!!!" if is_skin else "Valid Worker Clothes"
                        if is_skin:
                            resDict["Invalid_clothes"] = True
                            color = (0,0,255)  # RGB
                            cv2.circle(skin_area, (center[1],center[0]), 10, (255,0,0))
                            frame = cv2.rectangle(frame, (br[0],br[1]), (br[2],br[3]), color, 2)
                            frame = cv2.putText(frame, text, (min(br[0],br[2])+5, min(br[1],br[3])+15), cv2.FONT_HERSHEY_COMPLEX,
                                                0.4, color, 1)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'Standing..'
            clr_text = (0, 255, 0)
            clr_box = (0, 0, 255) if i in invalid_id else (0, 255, 0)  # Bug
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30: # 30
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                if out[0][out[0].argmax()] > 0.10:   # @yjy
                    action_name = action_model.class_names[out[0].argmax()]
                    action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                    if action_name in ['Fall Down', 'Lying Down']:
                        clr_text = (255, 0, 0)
                        resDict["Fall_down"] = True
                else:
                    continue

            # 人体骨架显示
            if track.time_since_update <= 2:  # 只画出丢失2帧之内的跟踪框 "==0 "
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), clr_box, 1)  # changed
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr_text, 2)

        # 判断吸烟或玩手机
        '''
        if is_PlayingPhone_Or_Smoking:
            text1 = "PlayingPhone Or Smoking Warning !!"
            cv2.putText(frame, text1, (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
            if leftArmAngle != 181:
                text2 = "angle of Right Arm: {:.2f}".format(leftArmAngle)
            else:
                text2 = "angle of Right Arm: None"
            if rightArmAngle != 181:
                text3 = "angle of Left Arm: {:.2f}".format(rightArmAngle)
            else:
                text3 = "angle of Left Arm: None"
            cv2.putText(frame, text2, (15,40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            cv2.putText(frame, text3, (15,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
        '''
        # draw danger area
        if len(region_points):
            frame = drawregion(frame, region=region_points, danger=len(dangerArea_id)>0)
            if len(dangerArea_id)>0:
                resDict["Dangerous_zoom"] = True
        # Show Frame.
        frame = frame[:, :, ::-1]  # RGB to BGR
        if outvid:
            writer.write(frame)

        frame = cv2.resize(frame, (0, 0), fx=zoom, fy=zoom)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        fps_time = time.time()
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('r'):
            region_points = getRegionFrame(count=4, frame=frame)
        elif key & 0xFF == ord('q'):
            break
            
        # resDict["Frame"] = frame
        if resDict["No_helmet"] or resDict["Smoking"] or resDict["Playing_phone"] or resDict["Fall_down"]:
            resDict["Text"] =  "Dangerous occurred !!!"
        print(resDict)
    
    # Clear resource.
    cam.stop()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
