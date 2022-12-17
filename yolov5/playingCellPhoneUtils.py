import math
import numpy as np

def phoneNearPerson(phone_center, person_box, imshape, padding=0.5) -> bool:
    phoneX, phoneY = phone_center[0], phone_center[1]
    imwidth = imshape[0]   # 640-xmax
    imheight = imshape[1]  # 480-ymax
    [xmin, ymin, xmax, ymax] = person_box
    pcenterX, pcenterY = (xmin + xmax)/2, (ymin + ymax)/2
    pwidth, pheight = xmax-xmin, ymax-ymin   # xmax-xmin, ymax-ymin 
    pwidth, pheight = pwidth*(1+padding), pheight*(1+padding)
    xmin, ymin, xmax, ymax = pcenterX-pwidth/2, pcenterY-pheight/2, pcenterX+pwidth/2, pcenterY+pheight/2
    # xmin, ymin, xmax, ymax = [i*(1 + padding) for i in person_boox]
    # x--640--imwidth, y--480--imheight
    xmin, xmax, ymin, ymax = int(max(min(xmin, imwidth), 0)), int(max(min(xmax, imwidth), 0)), int(max(min(ymin, imheight), 0)), int(max(min(ymax, imheight), 0))

    if phoneX > xmin and phoneX < xmax and phoneY > ymin and phoneY < ymax:
        return True
    return False

def pointInBox(point, bbox) -> bool:
    xmin, ymin, xmax, ymax = bbox
    if point[0]>=xmin and point[1]<=xmax and point[1]>=ymin and point[1]<=ymax:
        return True
    return False

def nearestTrackID(bbox, tracker):
    person_center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    dist_min = 100000000
    track_id_min = 1
    for i, track in enumerate(tracker.tracks):
        if not track.is_confirmed():
                continue
        track_id = track.track_id
        center_bbox = track.get_center().astype(int)
        dist_center = math.sqrt((person_center[0]-center_bbox[0])**2 + (person_center[1]-center_bbox[1])**2)
        if dist_center < dist_min:
            dist_min = dist_center
            track_id_min = track_id
    return track_id_min

def phoneNearHand(phone, hand, radius):
    phoneX, phoneY = phone[0],phone[1]
    # radius = math.sqrt((hand[0]-elbow[0])**2 + (hand[1]-elbow[1])**2)  # r = ||hand - elbow||
    dist = math.sqrt((hand[0]-phoneX)**2 + (hand[1]-phoneY)**2)
    if dist<= radius:
        return True
    return False


def phoneNearHead(phone, head, radius):
    phoneX, phoneY = phone[0],phone[1]
    # radius = math.sqrt((head[0]-shoulder_center[0])**2 + (head[1]-shoulder_center[1])**2)
    dist = math.sqrt((head[0]-phoneX)**2 + (head[1]-phoneY)**2)
    if dist <= radius:
        return True
    return False


def bendArmAngle(hand, elbow, shoulder):
    vector_elbow_shoulder = np.array((elbow[0]-shoulder[0], elbow[1]-shoulder[1]))
    vector_hand_elbow = np.array(([hand[0]-elbow[0], hand[1]-elbow[1]]))
    angle = np.arctan2(np.abs(np.cross(vector_elbow_shoulder, vector_hand_elbow)), np.dot(vector_elbow_shoulder, vector_hand_elbow))
    angle = 180 - abs(np.rad2deg(angle))
    return angle

def point_distance_line(point,line_point1,line_point2, radius):
	#计算向量
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
    return distance<=radius


def interSection_PhoneHand(phone, hand, elbow, expand=0.4):
    phone = expand_box(phone, imshape, expand)  # 判断线段与矩形的两条对角线是否相交
    h0, h1, e0, e1 = float(hand[0]), float(hand[1]), float(elbow[0]), float(elbow[1])
    xmin, ymin, xmax, ymax = float(phone[0]), float(phone[1]), float(phone[2]), float(phone[3])
    up_sector = (ymin-e1)*(h0-e0)/(h1-e1) + e0
    if(up_sector>=xmin and up_sector<=xmax):
        return True
    down_sector = (ymax-e1)*(h0-e0)/(h1-e1) + e0

    lengthHandElbow = matp.sqrt((hand[0]-elbow[0])**2 + (hand[1]-elbow[1])**2)
    distPhoneHand = math.sqrt((phone[0]-hand[0])**2 + (phone[1]-hand[1])**2)
    return distPhoneHand <= lengthHandElbow

def expand_box(rect, imshape, expand_ratio=0.15):
    imgh, imgw = imshape # HWC
    xmin, ymin, xmax, ymax = rect
    h_half = (ymax - ymin) * (1 + expand_ratio) / 2.
    w_half = (xmax - xmin) * (1 + expand_ratio) / 2.
    center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
    ymin = max(0, int(center[0] - h_half))  # 向上扩张
    ymax = min(imgh - 1, int(center[0] + h_half)) # 向下扩张
    xmin = max(0, int(center[1] - w_half))  
    xmax = min(imgw - 1, int(center[1] + w_half))
    return [xmin, ymin, xmax, ymax]

def get_target_result(results, target, cellphone_threshold=0.2, person_threshold=0.5):
    """
    Args:
        results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                        matix element:[class, score, x_min, y_min, x_max, y_max]
                        MaskRCNN's results include 'masks': np.ndarray:
                        shape:[N, im_h, im_w]
        target(list)： target labels
        threshold (float): threshold of box
    Returns:
        results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                        matix element:[class, score, x_min, y_min, x_max, y_max]
                        MaskRCNN's results include 'masks': np.ndarray:
                        shape:[N, im_h, im_w]
    """
    np_boxes = results["boxes"]
    expect_boxes = (np_boxes[:, 1] > cellphone_threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]

    target_bbox = []
    for dt in np_boxes:
        clsid, score, bbox = int(dt[0]), dt[1], dt[2:]
        if 0 in target and clsid == 0 and score > person_threshold:  # get person bbox
            target_bbox.append(dt)
        elif clsid in target:  # get cellphone bbox
            target_bbox.append(dt)
    results["boxes"] = np.array(target_bbox)
    return results

def get_cellphone_boxes(results, threshold=0.3, target=67,):
    np_boxes = results["boxes"]
    expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]

    phone_bbox = []
    for dt in np_boxes:
        clsid, score, bbox = int(dt[0]), dt[1], dt[2:]
        if clsid==target:
            phone_bbox.append(bbox)
    return phone_bbox


def judge_playing_phone(detect_results,
              keypoint_results,
              imshape = (640,480),
              phone_threshold=0.3,
              visual_thresh=0.2,  # 可见性
              ):

    # 计算手机方框坐标
    # phone_bbox = get_cellphone_boxes(detect_results, phone_threshold)
    phone_bbox = [det[:4] for det in detect_results] # [[[xyxy conf cls ]]]

    # 获取人体关键点和方框坐标
    skeletons = [kp['keypoints'].tolist() for kp in keypoint_results]
    # print(skeletons)
    scores = [kp['kp_score'].tolist() for kp in keypoint_results]
    # print(scores)
    person_bbox = [kp['bbox'].tolist() for kp in keypoint_results]
    # skeletons, scores = keypoint_results['keypoints'], keypoint_results['kp_score']
    # person_bbox = keypoint_results["bbox"].tolist()
    # print(person_bbox)
    kpt_nums = 13

    # 统计手机数量、人数
    num_phone = len(phone_bbox)
    num_person = len(skeletons)

    isPlayPhone = False
    angleLeftArm = 181
    angleRightArm = 181
    if num_person>0 and num_phone>0:
        for i in range(num_phone):
            phone_center = [(phone_bbox[i][0]+phone_bbox[i][2])//2, (phone_bbox[i][1] + phone_bbox[i][3])//2]
            for j in range(num_person):
                # 手机距离该人较远（0.5倍扩张），或者该人关键点可信度较低
                if not phoneNearPerson(phone_center, person_bbox[j], imshape, 0.5):
                    continue

                # 计算人体方框高、宽
                xmin, ymin, xmax, ymax = person_bbox[j]
                pwidth, pheight = xmax-xmin, ymax-ymin

                # 计算人体头部坐标
                # conf_head_joints_index = np.argmax(skeletons[j][:5][2])
                head_center = skeletons[j][0][:2]  # nose
                # print("conf_head_joints_index:", conf_head_joints_index)
                print("head_center:", head_center)
                
                # 计算右手、左手坐标
                right_hand = skeletons[j][6][:2]  # 10  list 不支持一次性读取一列
                right_hand_vis = scores[j][6][0] > visual_thread
                left_hand = skeletons[j][5][:2]   # 9
                left_hand_vis = scores[j][5][0] > visual_thread

                # 计算右肘、左肘坐标
                right_elbow_vis = scores[j][4][0] > visual_thread # 8
                right_elbow = skeletons[j][4][:2]
                left_elbow_vis = scores[j][3][0] > visual_thread # 7
                left_elbow = skeletons[j][3][:2]
                
                # 计算右肩、左肩坐标
                right_shoulder_vis = scores[j][2][0] > visual_thread # 6
                right_shoulder = skeletons[j][2][:2]
                left_shoulder_vis = scores[j][1][0] > visual_thread # 5
                left_shoulder = skeletons[j][1][:2]

                # 计算左臂弯曲角度:[0, 180]
                if left_hand_vis and left_elbow_vis and left_shoulder_vis:
                    angleLeftArm = bendArmAngle(left_hand, left_elbow, left_shoulder)

                # 计算右臂弯曲角度:[0, 180]
                if right_hand_vis and right_elbow_vis and right_shoulder_vis:
                    angleRightArm = bendArmAngle(right_hand, right_elbow, right_shoulder)
                
                print("angleLeftArm:", angleLeftArm)
                print("angleRightArm:", angleRightArm)

                # 计算左右肩中心
                shoulder_center = [(left_shoulder[0]+right_shoulder[1])//2, (right_shoulder[0]+right_shoulder[1])//2]

                # 以下情况认为在玩手机：
                # 1.手机在手的以人体宽度一半为半径的圆内，且相应的手臂弯曲角度大于120度；
                # 2.手机在头部中心的以人体宽度为半径的圆内，且双手手臂弯曲角度大于120度；
                if angleLeftArm < 120 and phoneNearHand(phone_center, left_hand, pwidth/2) \
                    or angleRightArm < 120 and phoneNearHand(phone_center, right_hand, pwidth/2) \
                    or phoneNearHead(phone_center, head_center, pwidth) and not(angleRightArm>120 and angleLeftArm>120):
                    # or phoneNearPerson(phone_center, person_bbox[j], imshape, 0.3):
                    isPlayPhone = True

    return isPlayPhone, angleRightArm, angleLeftArm

def maybe_playing_phone(
              keypoint_results,
              imshape = (640,480),
              visual_thresh=0.2,  # 可见性
              ):

    # 获取人体关键点和方框坐标
    skeletons = [kp['keypoints'].tolist() for kp in keypoint_results]
    scores = [kp['kp_score'].tolist() for kp in keypoint_results]
    # person_bbox = [kp['bbox'].tolist() for kp in keypoint_results]
    # kpt_nums = 13

    # 统计检测出姿态的人数
    num_person = len(skeletons)
    print("num_person", num_person)

    # MaybePlayPhoneMask = [[False] for _ in range(num_person)]
    MaybePlayPhoneMask = []
    angleLeftArm = 181
    angleRightArm = 181
    if num_person>0 :
        for j in range(num_person):
            # 手机距离该人较远（0.5倍扩张），或者该人关键点可信度较低
            # if not phoneNearPerson(phone_center, person_bbox[j], imshape, 0.5):
            #     continue

            # 计算人体方框高、宽
            # xmin, ymin, xmax, ymax = person_bbox[j]
            # pwidth, pheight = xmax-xmin, ymax-ymin

            # 计算人体头部坐标
            # head_center = skeletons[j][0][:2]  # nose
            # print("conf_head_joints_index:", conf_head_joints_index)
            # print("head_center:", head_center)
            
            # 计算右手、左手坐标
            right_hand = skeletons[j][6][:2]  # 10  list 不支持一次性读取一列
            right_hand_vis = scores[j][6][0] > visual_thread
            left_hand = skeletons[j][5][:2]   # 9
            left_hand_vis = scores[j][5][0] > visual_thread

            # 计算右肘、左肘坐标
            right_elbow_vis = scores[j][4][0] > visual_thread # 8
            right_elbow = skeletons[j][4][:2]
            left_elbow_vis = scores[j][3][0] > visual_thread # 7
            left_elbow = skeletons[j][3][:2]
            
            # 计算右肩、左肩坐标
            right_shoulder_vis = scores[j][2][0] > visual_thread # 6
            right_shoulder = skeletons[j][2][:2]
            left_shoulder_vis = scores[j][1][0] > visual_thread # 5
            left_shoulder = skeletons[j][1][:2]

            # 计算左臂弯曲角度:[0, 180]
            if left_hand_vis and left_elbow_vis and left_shoulder_vis:
                angleLeftArm = bendArmAngle(left_hand, left_elbow, left_shoulder)

            # 计算右臂弯曲角度:[0, 180]
            if right_hand_vis and right_elbow_vis and right_shoulder_vis:
                angleRightArm = bendArmAngle(right_hand, right_elbow, right_shoulder)
            
            print("angleLeftArm:", angleLeftArm)
            print("angleRightArm:", angleRightArm)

            # 计算左右肩中心
            # shoulder_center = [(left_shoulder[0]+right_shoulder[1])//2, (right_shoulder[0]+right_shoulder[1])//2]

            # 以下情况认为可能在玩手机：手臂弯曲角度大于120度；
            if angleLeftArm < 120 or angleRightArm < 120:
                # MaybePlayPhoneMask[j] = [True]
                MaybePlayPhoneMask.append(j)

    return MaybePlayPhoneMask

def phone_smoke_position_valid(
              keypoint_results,
              det = None,  # xyxy conf cls
              imshape = (640,480),
              visual_thresh=0.2,  # 可见性
              ):
    det = det.tolist()[0]
    phone_center = [(det[0]+det[2])//2, (det[1] + det[3])//2]

    # 获取人体关键点和方框坐标
    skeletons = [kp['keypoints'].tolist() for kp in keypoint_results]
    scores = [kp['kp_score'].tolist() for kp in keypoint_results]
    person_bbox = [kp['bbox'].tolist() for kp in keypoint_results]

    # 统计检测出姿态的人数
    num_person = len(skeletons)

    isValidPhone = True
    if num_person>0 :
        for j in range(num_person):
            # 手机距离该人较远（0.5倍扩张），或者该人关键点可信度较低
            if not phoneNearPerson(phone_center, person_bbox[j], imshape, 0.4):
                continue

            # 计算人体方框高、宽
            xmin, ymin, xmax, ymax = person_bbox[j]
            pwidth, pheight = xmax-xmin, ymax-ymin

            # 计算人体头部坐标
            head_center = skeletons[j][0][:2]  # nose
            # print("conf_head_joints_index:", conf_head_joints_index)
            # print("head_center:", head_center)
            
            # 计算右手、左手坐标
            right_hand = skeletons[j][6][:2]  # 10  list 不支持一次性读取一列
            right_hand_vis = scores[j][6][0] > visual_thresh
            left_hand = skeletons[j][5][:2]   # 9
            left_hand_vis = scores[j][5][0] > visual_thresh\
            
            '''
            # 计算右肘、左肘坐标
            right_elbow_vis = scores[j][4][0] > visual_thresh # 8
            right_elbow = skeletons[j][4][:2]
            left_elbow_vis = scores[j][3][0] > visual_thresh # 7
            left_elbow = skeletons[j][3][:2]
            
            # 计算右肩、左肩坐标
            right_shoulder_vis = scores[j][2][0] > visual_thresh # 6
            right_shoulder = skeletons[j][2][:2]
            left_shoulder_vis = scores[j][1][0] > visual_thresh # 5
            left_shoulder = skeletons[j][1][:2]

            # 计算左臂弯曲角度:[0, 180]
            if left_hand_vis and left_elbow_vis and left_shoulder_vis:
                angleLeftArm = bendArmAngle(left_hand, left_elbow, left_shoulder)

            # 计算右臂弯曲角度:[0, 180]
            if right_hand_vis and right_elbow_vis and right_shoulder_vis:
                angleRightArm = bendArmAngle(right_hand, right_elbow, right_shoulder)

            print("angleLeftArm:", angleLeftArm)
            print("angleRightArm:", angleRightArm)
            '''

            # 计算左右肩中心
            # shoulder_center = [(left_shoulder[0]+right_shoulder[1])//2, (right_shoulder[0]+right_shoulder[1])//2]
            
            if not (phoneNearHand(phone_center, left_hand, pwidth) \
                and phoneNearHand(phone_center, right_hand, pwidth) \
                and phoneNearHead(phone_center, head_center, pwidth)):
                isValidPhone = False
    print("isValidPhone",isValidPhone)
    return isValidPhone

def getHeadArea(keypoint_results, imshape=(640,480), padding=0.2, visual_thresh=0.5):
    width, height = imshape[0], imshape[1]
    # 获取人体关键点方框和坐标
    skeletons, scores = keypoint_results['keypoint']
    person_box = keypoint_results['bbox']

    # 计算人数
    person_nums = len(skeletons)
    heads = []
    hands = []
    for i in range(person_nums):

        conf_head_joints_index = np.argmax(skeletons[i][:5][2])
        # 计算右手、左手坐标
        right_hand = skeletons[i][10, :2]
        conf_right_hand = scores[i][10] > visual_thread

        left_hand = skeletons[i][9, :2]
        conf_left_hand = scores[i][9] > visual_thread

        xmin, ymin, xmax, ymax = person_box[i]
        pwidth, pheight = xmax-xmin, ymax-ymin     # xmax-xmin, ymax-ymin 

        if scores[i][0] > visual_thread:
            head_center = skeletons[i][conf_head_joints_index][:2]
            # h = pwidth/4

            neck = (skeletons[i][5][:2] + skeletons[i][6][:2])/2
            orient_vect = head_center - neck
            h = math.sqrt(orient_vect[0]**2 + orient_vect[1]**2)
            h = h*(1+padding)

            leftTop = [int(head_center[0]-h), int(head_center[1]-h)]
            rightDown = [int(head_center[0]+h), int(head_center[1]+h)]
            leftTop = [max(min(leftTop[0], width), 0), max(min(leftTop[1], height), 0)]
            rightDown = [max(min(rightDown[0], width), 0), max(min(rightDown[1], height), 0)]

            heads.append([leftTop, rightDown])

        if conf_right_hand > visual_thread:  # Error @yjy
            # h = pwidth/6
            right_elbow = skeletons[i][8, :2]
            orient_vect = right_hand - right_elbow
            h = math.sqrt(orient_vect[0]**2 + orient_vect[1]**2)
            right_hand = right_hand + orient_vect/2
            h = h/2
            h = h*(1+padding)

            leftTop = [int(right_hand[0]-h), int(right_hand[1]-h)]
            rightDown = [int(right_hand[0]+h), int(right_hand[1]+h)]
            leftTop = [max(min(leftTop[0], width), 0), max(min(leftTop[1], height), 0)]
            rightDown = [max(min(rightDown[0], width), 0), max(min(rightDown[1], height), 0)]

            hands.append([leftTop, rightDown])
        

        if conf_left_hand > visual_thread:
            # h = pwidth/6
            left_elbow = skeletons[i][7, :2]
            orient_vect = left_hand - left_elbow
            h = math.sqrt(orient_vect[0]**2 + orient_vect[1]**2)
            left_hand = left_hand + orient_vect/2
            h = h/2
            h = h*(1+padding)

            leftTop = [int(left_hand[0]-h), int(left_hand[1]-h)]
            rightDown = [int(left_hand[0]+h), int(left_hand[1]+h)]
            leftTop = [max(min(leftTop[0], width), 0), max(min(leftTop[1], height), 0)]
            rightDown = [max(min(rightDown[0], width), 0), max(min(rightDown[1], height), 0)]

            hands.append([leftTop, rightDown])


        # top = [int((xmin+xmax)/2), ymin]
        # down = [int((xmin+xmax)/2), ymax]
        # 计算人体头部坐标
    # print("heads:", heads)
    return heads + hands
        
def getBodyRects(
              keypoint_results,
              imshape = [640,640],
              visual_thresh=0.3,  # 可见性
              ):
    # 获取人体关键点和方框坐标
    skeletons = [kp['keypoints'].tolist() for kp in keypoint_results]
    scores = [kp['kp_score'].tolist() for kp in keypoint_results]
    # person_bbox = [kp['bbox'].tolist() for kp in keypoint_results]

    # 统计检测出姿态的人数
    num_person = len(skeletons)

    bodyRects = []
    if num_person>0 :
        for j in range(num_person):
            # 计算人体方框高、宽
            # xmin, ymin, xmax, ymax = person_bbox[j]
            # pwidth, pheight = xmax-xmin, ymax-ymin
            
            # 计算右肩、左肩坐标
            right_shoulder_vis = scores[j][1][0] > visual_thread # 5 右边
            right_shoulder = skeletons[j][1][:2]
            left_shoulder_vis = scores[j][2][0] > visual_thread # 6 左边
            left_shoulder = skeletons[j][2][:2]
            # print("left:",left_shoulder)
            # print("right:",right_shoulder)

            # 计算右胯、左胯坐标
            right_hip_vis = scores[j][7][0] > visual_thread # 11  右边
            right_hip = skeletons[j][7][:2]
            left_hip_vis = scores[j][8][0] > visual_thread # 12  左边
            left_hip = skeletons[j][8][:2]
            # print("left_hip",left_hip)
            # print("right_hip",right_hip)

            if right_shoulder_vis and left_shoulder_vis and right_hip_vis and left_hip_vis:
                # 计算左右肩膀距离
                width_sholder = math.sqrt((right_shoulder[0]-left_shoulder[0])**2 + (right_shoulder[1]-left_shoulder[1])**2)
                down_y = (right_hip[1]+left_hip[1])//2
                xmin, ymin = int(left_shoulder[0]), int(left_shoulder[1])
                xmax, ymax = int(right_shoulder[0]), int(down_y)
                bodyRects.append(expand_box([xmin,ymin,xmax,ymax], imshape=imshape, expand_ratio=0.2))

    return bodyRects

def getArmRects(
              keypoint_results,
              imshape = [640,640],
              ignore_thresh=0.05,
              visual_thresh=0.3,  # 可见性
              ):
    # 获取人体关键点和方框坐标
    skeletons = [kp['keypoints'].tolist() for kp in keypoint_results]
    scores = [kp['kp_score'].tolist() for kp in keypoint_results]
    person_bbox = [kp['bbox'].tolist() for kp in keypoint_results]

    # 统计检测出姿态的人数
    num_person = len(skeletons)

    bodyRects = []
    if num_person>0 :
        for j in range(num_person):
            # 计算人体方框高、宽
            xmin, ymin, xmax, ymax = person_bbox[j]
            pwidth, pheight = xmax-xmin, ymax-ymin
            if pwidth*pheight < 0.05 * imshape[0]*imshape[1]:
                continue

            # 计算右手、左手坐标
            right_hand = skeletons[j][6][:2]  # 10  list 不支持一次性读取一列
            right_hand_vis = scores[j][6][0] > visual_thresh
            left_hand = skeletons[j][5][:2]   # 9
            left_hand_vis = scores[j][5][0] > visual_thresh

            # 计算右肘、左肘坐标
            right_elbow_vis = scores[j][4][0] > visual_thresh # 8
            right_elbow = skeletons[j][4][:2]
            left_elbow_vis = scores[j][3][0] > visual_thresh # 7
            left_elbow = skeletons[j][3][:2]

            if right_hand_vis and right_elbow_vis or left_hand_vis and left_elbow_vis:
                # 计算手臂法向量
                # alpha = 0.5*np.pi
                if right_hand_vis and right_elbow_vis:
                    if right_hand[1]<right_elbow[1]: # (x,y) hand_y < elbow_y
                        vector_right = np.array([right_hand[0]-right_elbow[0], right_hand[1]-right_elbow[1]])
                    else:
                        vector_right = np.array([right_elbow[0]-right_hand[0], right_elbow[1]-right_hand[1]])
                    lengthArm = np.linalg.norm(vector_right)
                    if lengthArm == 0:
                        continue
                    vector_right /= lengthArm
                    # x = vector_right[1]*np.sin(alpha)+vector_right[0]*np.cos(alpha)
                    # y = vector_right[1]*np.cos(alpha)+vector_right[0]*np.sin(alpha)
                    x = vector_right[1]  # 法向量
                    y = vector_right[0]
                    vector_vertical = np.array([x, y])
                    # print("hand",right_hand)
                    # print("elbow", right_elbow)
                    if right_hand[1]<=right_elbow[1]: # (x,y) hand_y < elbow_y 手在上方
                        if right_hand[0]>=right_elbow[0]: # 图像中手在肘部右上方
                            leftTop = np.array([right_elbow[0], right_hand[1]])
                            rightDown = np.array([right_hand[0], right_elbow[1]])
                        else:
                            leftTop = np.array(right_hand) + vector_vertical*lengthArm/4
                            rightDown = np.array(right_elbow) - vector_vertical*lengthArm/4
                    else:
                        if right_hand[0]>=right_elbow[0]:
                            rightDown = np.array(right_hand) - vector_vertical*lengthArm/5
                            leftTop = np.array(right_elbow) + vector_vertical*lengthArm/5
                        else:  # 图像中肘部在手的右上方
                            leftTop = np.array([right_hand[0], right_elbow[1]])
                            rightDown = np.array([right_elbow[0], right_hand[1]])
                    xyxy = [int(leftTop[0]), int(leftTop[1]), int(rightDown[0]), int(rightDown[1])]
                    bodyRects.append(xyxy)

                if left_hand_vis and left_elbow_vis:
                    if left_hand[1]<left_elbow[1]: # (x,y) hand_y < elbow_y
                        vector_left = np.array([left_hand[0]-left_elbow[0], left_hand[1]-left_elbow[1]])
                    else:
                        vector_left = np.array([left_elbow[0]-left_hand[0], left_elbow[1]-left_hand[1]])
                    lengthArm = np.linalg.norm(vector_left)
                    if lengthArm == 0:
                        continue
                    vector_left /= lengthArm
                    # x = vector_left[1]*np.sin(alpha)+vector_left[0]*np.cos(alpha)
                    # y = vector_left[1]*np.cos(alpha)+vector_left[0]*np.sin(alpha)
                    x = vector_left[1] # 法向量
                    y = vector_left[0]
                    vector_vertical = np.array([x, y])
                    # print("hand",left_hand)
                    # print("elbow", left_elbow)
                    if left_hand[1]<=left_elbow[1]: # (x,y) hand_y < elbow_y 手在上方
                        if left_hand[0]>=left_elbow[0]: # 图像中手在肘部右上方
                            leftTop = np.array([left_elbow[0], left_hand[1]])
                            rightDown = np.array([left_hand[0], left_elbow[1]])
                        else:
                            leftTop = np.array(left_hand) + vector_vertical*lengthArm/4
                            rightDown = np.array(left_elbow) - vector_vertical*lengthArm/4
                    else:
                        if left_hand[0]>=left_elbow[0]:
                            rightDown = np.array(left_hand) - vector_vertical*lengthArm/5
                            leftTop = np.array(left_elbow) + vector_vertical*lengthArm/5
                        else:  # 图像中肘部在手的右上方
                            leftTop = np.array([left_hand[0], left_elbow[1]])
                            rightDown = np.array([left_elbow[0], left_hand[1]])
                    # print(vector_vertical)
                    xyxy = [int(leftTop[0]), int(leftTop[1]), int(rightDown[0]), int(rightDown[1])]
                    bodyRects.append(xyxy)

    return bodyRects

def getArmKneeRects(
              keypoint_results,
              imshape = [640,640],
              ignore_thresh=0.05,
              visual_thresh=0.3,  # 可见性
              ):
    # 获取人体关键点和方框坐标
    skeletons = [kp['keypoints'].tolist() for kp in keypoint_results]
    scores = [kp['kp_score'].tolist() for kp in keypoint_results]
    person_bbox = [kp['bbox'].tolist() for kp in keypoint_results]

    # 统计检测出姿态的人数
    num_person = len(skeletons)

    bodyRects = []
    if num_person>0 :
        for j in range(num_person):
            # 计算人体方框高、宽
            xmin, ymin, xmax, ymax = person_bbox[j]
            pwidth, pheight = xmax-xmin, ymax-ymin
            if pwidth*pheight < 0.05 * imshape[0]*imshape[1]:
                continue

            # 计算右手、左手坐标
            right_hand = skeletons[j][6][:2]  # 10  list 不支持一次性读取一列
            right_hand_vis = scores[j][6][0] > visual_thresh
            left_hand = skeletons[j][5][:2]   # 9
            left_hand_vis = scores[j][5][0] > visual_thresh

            # 计算右肘、左肘坐标
            right_elbow_vis = scores[j][4][0] > visual_thresh # 8
            right_elbow = skeletons[j][4][:2]
            left_elbow_vis = scores[j][3][0] > visual_thresh # 7
            left_elbow = skeletons[j][3][:2]

            # 计算左脚、右脚坐标
            right_ankle = skeletons[j][12][:2]  # 10  list 不支持一次性读取一列
            right_ankle_vis = scores[j][12][0] > visual_thresh
            left_ankle = skeletons[j][11][:2]   # 9
            left_ankle_vis = scores[j][11][0] > visual_thresh

            # 计算左膝、右膝坐标
            right_knee_vis = scores[j][10][0] > visual_thresh # 8
            right_knee = skeletons[j][10][:2]
            left_knee_vis = scores[j][9][0] > visual_thresh # 7
            left_knee = skeletons[j][9][:2]

            # 计算左右手臂
            if right_hand_vis and right_elbow_vis or left_hand_vis and left_elbow_vis:
                # 计算手臂法向量
                # alpha = 0.5*np.pi
                if right_hand_vis and right_elbow_vis:
                    if right_hand[1]<right_elbow[1]: # (x,y) hand_y < elbow_y
                        vector_right = np.array([right_hand[0]-right_elbow[0], right_hand[1]-right_elbow[1]])
                    else:
                        vector_right = np.array([right_elbow[0]-right_hand[0], right_elbow[1]-right_hand[1]])
                    lengthArm = np.linalg.norm(vector_right)
                    if lengthArm == 0:
                        continue
                    vector_right /= lengthArm
                    # x = vector_right[1]*np.sin(alpha)+vector_right[0]*np.cos(alpha)
                    # y = vector_right[1]*np.cos(alpha)+vector_right[0]*np.sin(alpha)
                    x = vector_right[1]  # 法向量
                    y = vector_right[0]
                    vector_vertical = np.array([x, y])
                    # print("hand",right_hand)
                    # print("elbow", right_elbow)
                    if right_hand[1]<=right_elbow[1]: # (x,y) hand_y < elbow_y 手在上方
                        if right_hand[0]>=right_elbow[0]: # 图像中手在肘部右上方
                            leftTop = np.array([right_elbow[0], right_hand[1]])
                            rightDown = np.array([right_hand[0], right_elbow[1]])
                        else:
                            leftTop = np.array(right_hand) + vector_vertical*lengthArm/4
                            rightDown = np.array(right_elbow) - vector_vertical*lengthArm/4
                    else:
                        if right_hand[0]>=right_elbow[0]:
                            rightDown = np.array(right_hand) - vector_vertical*lengthArm/4
                            leftTop = np.array(right_elbow) + vector_vertical*lengthArm/4
                        else:  # 图像中肘部在手的右上方
                            leftTop = np.array([right_hand[0], right_elbow[1]])
                            rightDown = np.array([right_elbow[0], right_hand[1]])
                    xyxy = [int(leftTop[0]), int(leftTop[1]), int(rightDown[0]), int(rightDown[1])]
                    bodyRects.append(xyxy)

                if left_hand_vis and left_elbow_vis:
                    if left_hand[1]<left_elbow[1]: # (x,y) hand_y < elbow_y
                        vector_left = np.array([left_hand[0]-left_elbow[0], left_hand[1]-left_elbow[1]])
                    else:
                        vector_left = np.array([left_elbow[0]-left_hand[0], left_elbow[1]-left_hand[1]])
                    lengthArm = np.linalg.norm(vector_left)
                    if lengthArm == 0:
                        continue
                    vector_left /= lengthArm
                    # x = vector_left[1]*np.sin(alpha)+vector_left[0]*np.cos(alpha)
                    # y = vector_left[1]*np.cos(alpha)+vector_left[0]*np.sin(alpha)
                    x = vector_left[1] # 法向量
                    y = vector_left[0]
                    vector_vertical = np.array([x, y])
                    # print("hand",left_hand)
                    # print("elbow", left_elbow)
                    if left_hand[1]<=left_elbow[1]: # (x,y) hand_y < elbow_y 手在上方
                        if left_hand[0]>=left_elbow[0]: # 图像中手在肘部右上方
                            leftTop = np.array([left_elbow[0], left_hand[1]])
                            rightDown = np.array([left_hand[0], left_elbow[1]])
                        else:
                            leftTop = np.array(left_hand) + vector_vertical*lengthArm/4
                            rightDown = np.array(left_elbow) - vector_vertical*lengthArm/4
                    else:
                        if left_hand[0]>=left_elbow[0]:
                            rightDown = np.array(left_hand) - vector_vertical*lengthArm/4
                            leftTop = np.array(left_elbow) + vector_vertical*lengthArm/4
                        else:  # 图像中肘部在手的右上方
                            leftTop = np.array([left_hand[0], left_elbow[1]])
                            rightDown = np.array([left_elbow[0], left_hand[1]])
                    # print(vector_vertical)
                    xyxy = [int(leftTop[0]), int(leftTop[1]), int(rightDown[0]), int(rightDown[1])]
                    bodyRects.append(xyxy)
                
            # 计算左右膝盖
            if right_ankle_vis and right_knee_vis or left_ankle_vis and left_knee_vis:
                # 计算腿部法向量
                # alpha = 0.5*np.pi
                if right_ankle_vis and right_knee_vis:
                    if right_ankle[1]<right_knee[1]: # (x,y) hand_y < elbow_y
                        vector_right = np.array([right_ankle[0]-right_knee[0], right_ankle[1]-right_knee[1]])
                    else:
                        vector_right = np.array([right_knee[0]-right_ankle[0], right_knee[1]-right_ankle[1]])
                    lengthLeg = np.linalg.norm(vector_right)
                    if lengthLeg == 0:
                        continue
                    vector_right /= lengthLeg
                    x = vector_right[1]  # 法向量
                    y = vector_right[0]
                    vector_vertical = np.array([x, y])
                    # print("hand",right_hand)
                    # print("elbow", right_elbow)
                    if right_ankle[1]<=right_knee[1]: # (x,y) hand_y < elbow_y 手在上方
                        if right_ankle[0]>=right_knee[0]: # 图像中手在肘部右上方
                            leftTop = np.array([right_knee[0], right_ankle[1]])
                            rightDown = np.array([right_ankle[0], right_knee[1]])
                        else:
                            leftTop = np.array(right_ankle) + vector_vertical*lengthLeg/4
                            rightDown = np.array(right_knee) - vector_vertical*lengthLeg/4
                    else:
                        if right_ankle[0]>=right_knee[0]:
                            rightDown = np.array(right_ankle) - vector_vertical*lengthLeg/4
                            leftTop = np.array(right_knee) + vector_vertical*lengthLeg/4
                        else:  # 图像中肘部在手的右上方
                            leftTop = np.array([right_ankle[0], right_knee[1]])
                            rightDown = np.array([right_knee[0], right_ankle[1]])
                    xyxy = [int(leftTop[0]), int(leftTop[1]), int(rightDown[0]), int(rightDown[1])]
                    bodyRects.append(xyxy)

                if left_ankle_vis and left_knee_vis:
                    if left_ankle[1]<left_knee[1]: # (x,y) hand_y < elbow_y
                        vector_left = np.array([left_ankle[0]-left_knee[0], left_ankle[1]-left_knee[1]])
                    else:
                        vector_left = np.array([left_knee[0]-left_ankle[0], left_knee[1]-left_ankle[1]])
                    lengthLeg = np.linalg.norm(vector_left)
                    if lengthLeg == 0:
                        continue
                    vector_left /= lengthLeg
                    x = vector_left[1]  # 法向量
                    y = vector_left[0]
                    vector_vertical = np.array([x, y])
                    # print("hand",right_hand)
                    # print("elbow", right_elbow)
                    if left_ankle[1]<=left_knee[1]: # (x,y) hand_y < elbow_y 手在上方
                        if left_ankle[0]>=left_knee[0]: # 图像中手在肘部右上方
                            leftTop = np.array([left_knee[0], left_ankle[1]])
                            rightDown = np.array([left_ankle[0], left_knee[1]])
                        else:
                            leftTop = np.array(left_ankle) + vector_vertical*lengthLeg/4
                            rightDown = np.array(left_knee) - vector_vertical*lengthLeg/4
                    else:
                        if left_ankle[0]>=left_knee[0]:
                            rightDown = np.array(left_ankle) - vector_vertical*lengthLeg/4
                            leftTop = np.array(left_knee) + vector_vertical*lengthLeg/4
                        else:  # 图像中肘部在手的右上方
                            leftTop = np.array([left_ankle[0], left_knee[1]])
                            rightDown = np.array([left_knee[0], left_ankle[1]])
                    xyxy = [int(leftTop[0]), int(leftTop[1]), int(rightDown[0]), int(rightDown[1])]
                    bodyRects.append(xyxy)

    return bodyRects

def getArmKneeRectsOnePerson(
              keypoint_results,
              bbox,
              imshape = [640,640],
              ignore_thresh=0.05,
              visual_thresh=0.3,  # 可见性
              ):
    # 获取人体关键点和方框坐标
    skeletons = keypoint_results  # [x, y, score]
    person_bbox = bbox

    # 统计检测出姿态的人数
    num_person = len(skeletons)

    bodyRects = []
    if num_person>0 :
        # 计算人体方框高、宽
        xmin, ymin, xmax, ymax = person_bbox
        pwidth, pheight = xmax-xmin, ymax-ymin
        if pwidth*pheight < 0.05 * imshape[0]*imshape[1]:
            return bodyRects

        # 计算右手、左手坐标
        right_hand = skeletons[6][:2]  # 10  list 不支持一次性读取一列
        right_hand_vis = skeletons[6][2] > visual_thresh
        left_hand = skeletons[5][:2]   # 9
        left_hand_vis = skeletons[5][2] > visual_thresh

        # 计算右肘、左肘坐标
        right_elbow_vis = skeletons[4][2] > visual_thresh # 8
        right_elbow = skeletons[4][:2]
        left_elbow_vis = skeletons[3][2] > visual_thresh # 7
        left_elbow = skeletons[3][:2]

        # 计算左脚、右脚坐标
        right_ankle = skeletons[12][:2]  # 10  list 不支持一次性读取一列
        right_ankle_vis = skeletons[12][2] > visual_thresh
        left_ankle = skeletons[11][:2]   # 9
        left_ankle_vis = skeletons[11][2] > visual_thresh

        # 计算左膝、右膝坐标
        right_knee_vis = skeletons[10][2] > visual_thresh # 8
        right_knee = skeletons[10][:2]
        left_knee_vis = skeletons[9][2] > visual_thresh # 7
        left_knee = skeletons[9][:2]

        # 计算左右手臂
        if right_hand_vis and right_elbow_vis or left_hand_vis and left_elbow_vis:
            # 计算手臂法向量
            if right_hand_vis and right_elbow_vis:
                if right_hand[1]<right_elbow[1]: # (x,y) hand_y < elbow_y
                    vector_right = np.array([right_hand[0]-right_elbow[0], right_hand[1]-right_elbow[1]])
                else:
                    vector_right = np.array([right_elbow[0]-right_hand[0], right_elbow[1]-right_hand[1]])
                lengthArm = np.linalg.norm(vector_right)
                if lengthArm != 0:
                    vector_right /= lengthArm
                    # x = vector_right[1]*np.sin(alpha)+vector_right[0]*np.cos(alpha)
                    # y = vector_right[1]*np.cos(alpha)+vector_right[0]*np.sin(alpha)
                    x = vector_right[1]  # 法向量
                    y = vector_right[0]
                    vector_vertical = np.array([x, y])
                    if right_hand[1]<=right_elbow[1]: # (x,y) hand_y < elbow_y 手在上方
                        if right_hand[0]>=right_elbow[0]: # 图像中手在肘部右上方
                            leftTop = np.array([right_elbow[0], right_hand[1]])
                            rightDown = np.array([right_hand[0], right_elbow[1]])
                        else:
                            leftTop = np.array(right_hand) + vector_vertical*lengthArm/4
                            rightDown = np.array(right_elbow) - vector_vertical*lengthArm/4
                    else:
                        if right_hand[0]>=right_elbow[0]:
                            rightDown = np.array(right_hand) - vector_vertical*lengthArm/4
                            leftTop = np.array(right_elbow) + vector_vertical*lengthArm/4
                        else:  # 图像中肘部在手的右上方
                            leftTop = np.array([right_hand[0], right_elbow[1]])
                            rightDown = np.array([right_elbow[0], right_hand[1]])
                    xyxy = [int(leftTop[0]), int(leftTop[1]), int(rightDown[0]), int(rightDown[1])]
                    bodyRects.append(xyxy)

            if left_hand_vis and left_elbow_vis:
                if left_hand[1]<left_elbow[1]: # (x,y) hand_y < elbow_y
                    vector_left = np.array([left_hand[0]-left_elbow[0], left_hand[1]-left_elbow[1]])
                else:
                    vector_left = np.array([left_elbow[0]-left_hand[0], left_elbow[1]-left_hand[1]])
                lengthArm = np.linalg.norm(vector_left)
                if lengthArm != 0:
                    vector_left /= lengthArm
                    # x = vector_left[1]*np.sin(alpha)+vector_left[0]*np.cos(alpha)
                    # y = vector_left[1]*np.cos(alpha)+vector_left[0]*np.sin(alpha)
                    x = vector_left[1] # 法向量
                    y = vector_left[0]
                    vector_vertical = np.array([x, y])
                    if left_hand[1]<=left_elbow[1]: # (x,y) hand_y < elbow_y 手在上方
                        if left_hand[0]>=left_elbow[0]: # 图像中手在肘部右上方
                            leftTop = np.array([left_elbow[0], left_hand[1]])
                            rightDown = np.array([left_hand[0], left_elbow[1]])
                        else:
                            leftTop = np.array(left_hand) + vector_vertical*lengthArm/4
                            rightDown = np.array(left_elbow) - vector_vertical*lengthArm/4
                    else:
                        if left_hand[0]>=left_elbow[0]:
                            rightDown = np.array(left_hand) - vector_vertical*lengthArm/4
                            leftTop = np.array(left_elbow) + vector_vertical*lengthArm/4
                        else:  # 图像中肘部在手的右上方
                            leftTop = np.array([left_hand[0], left_elbow[1]])
                            rightDown = np.array([left_elbow[0], left_hand[1]])
                    # print(vector_vertical)
                    xyxy = [int(leftTop[0]), int(leftTop[1]), int(rightDown[0]), int(rightDown[1])]
                    bodyRects.append(xyxy)
            
        # 计算左右膝盖
        if right_ankle_vis and right_knee_vis or left_ankle_vis and left_knee_vis:
            # 计算腿部法向量
            # alpha = 0.5*np.pi
            if right_ankle_vis and right_knee_vis:
                if right_ankle[1]<right_knee[1]: # (x,y) hand_y < elbow_y
                    vector_right = np.array([right_ankle[0]-right_knee[0], right_ankle[1]-right_knee[1]])
                else:
                    vector_right = np.array([right_knee[0]-right_ankle[0], right_knee[1]-right_ankle[1]])
                lengthLeg = np.linalg.norm(vector_right)
                if lengthLeg != 0:
                    vector_right /= lengthLeg
                    x = vector_right[1]  # 法向量
                    y = vector_right[0]
                    vector_vertical = np.array([x, y])
                    # print("hand",right_hand)
                    # print("elbow", right_elbow)
                    if right_ankle[1]<=right_knee[1]: # (x,y) hand_y < elbow_y 手在上方
                        if right_ankle[0]>=right_knee[0]: # 图像中手在肘部右上方
                            leftTop = np.array([right_knee[0], right_ankle[1]])
                            rightDown = np.array([right_ankle[0], right_knee[1]])
                        else:
                            leftTop = np.array(right_ankle) + vector_vertical*lengthLeg/4
                            rightDown = np.array(right_knee) - vector_vertical*lengthLeg/4
                    else:
                        if right_ankle[0]>=right_knee[0]:
                            rightDown = np.array(right_ankle) - vector_vertical*lengthLeg/4
                            leftTop = np.array(right_knee) + vector_vertical*lengthLeg/4
                        else:  # 图像中肘部在手的右上方
                            leftTop = np.array([right_ankle[0], right_knee[1]])
                            rightDown = np.array([right_knee[0], right_ankle[1]])
                    xyxy = [int(leftTop[0]), int(leftTop[1]), int(rightDown[0]), int(rightDown[1])]
                    bodyRects.append(xyxy)

            if left_ankle_vis and left_knee_vis:
                if left_ankle[1]<left_knee[1]: # (x,y) hand_y < elbow_y
                    vector_left = np.array([left_ankle[0]-left_knee[0], left_ankle[1]-left_knee[1]])
                else:
                    vector_left = np.array([left_knee[0]-left_ankle[0], left_knee[1]-left_ankle[1]])
                lengthLeg = np.linalg.norm(vector_left)
                if lengthLeg != 0:
                    vector_left /= lengthLeg
                    x = vector_left[1]  # 法向量
                    y = vector_left[0]
                    vector_vertical = np.array([x, y])
                    # print("hand",right_hand)
                    # print("elbow", right_elbow)
                    if left_ankle[1]<=left_knee[1]: # (x,y) hand_y < elbow_y 手在上方
                        if left_ankle[0]>=left_knee[0]: # 图像中手在肘部右上方
                            leftTop = np.array([left_knee[0], left_ankle[1]])
                            rightDown = np.array([left_ankle[0], left_knee[1]])
                        else:
                            leftTop = np.array(left_ankle) + vector_vertical*lengthLeg/4
                            rightDown = np.array(left_knee) - vector_vertical*lengthLeg/4
                    else:
                        if left_ankle[0]>=left_knee[0]:
                            rightDown = np.array(left_ankle) - vector_vertical*lengthLeg/4
                            leftTop = np.array(left_knee) + vector_vertical*lengthLeg/4
                        else:  # 图像中肘部在手的右上方
                            leftTop = np.array([left_ankle[0], left_knee[1]])
                            rightDown = np.array([left_knee[0], left_ankle[1]])
                    xyxy = [int(leftTop[0]), int(leftTop[1]), int(rightDown[0]), int(rightDown[1])]
                    bodyRects.append(xyxy)

    return bodyRects
