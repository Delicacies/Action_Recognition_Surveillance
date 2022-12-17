import json
from random import choice

import cv2
import time
import multiprocessing as mp

import numpy as np
import asyncio
import websockets
# import pymongo

from camTracker_python_native.PTZControl import PTZControl

frame = None
mouse_events = ["UP_LEFTPressed", "UP_LEFTReleased", "UP_RIGHTPressed", "UP_RIGHTReleased", "DOWN_LEFTPressed",
                "DOWN_LEFTReleased", "DOWN_RIGHTPressed", "DOWN_RIGHTReleased", "PAN_LEFTPressed", "PAN_LEFTReleased",
                "PAN_RIGHTPressed", "PAN_RIGHTReleased", "TILT_UPPressed", "TILT_UPReleased", "TILT_DOWNPressed",
                "TILT_DOWNReleased", "ZOOM_INPressed", "ZOOM_INReleased", "ZOOM_OUTPressed", "ZOOM_OUTReleased"]
# camera_infos = [{"sDVRIP": "192.168.1.64", "wDVRPort": 8000, "sUserName": "admin", "sPassword": "hk888888"},
#                 {"sDVRIP": "192.168.1.64", "wDVRPort": 8000, "sUserName": "admin", "sPassword": "hk888888"},
#                 {"sDVRIP": "192.168.1.64", "wDVRPort": 8000, "sUserName": "admin", "sPassword": "hk888888"},
#                 {"sDVRIP": "192.168.1.64", "wDVRPort": 8000, "sUserName": "admin", "sPassword": "hk888888"}]


def websocket_process(img_dict, show_id, view_all):
    # 服务器端主逻辑
    async def main_logic(websocket, path):
        await send_img(websocket, img_dict, show_id, view_all)
        # await recv_msg(websocket, img_dict)
        # asyncio.get_event_loop().create_task(send_img(websocket, img_dict))

    start_server = websockets.serve(main_logic, '0.0.0.0', 2000)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


def websocket_process2(img_dict, q, show_id, camera_cfgs, view_all):
    # 服务器端主逻辑
    async def main_logic(websocket, path):
        await recv_msg(websocket, img_dict, q, show_id, camera_cfgs, view_all)
        # await recv_msg(websocket, img_dict)
        # asyncio.get_event_loop().create_task(send_img(websocket, img_dict))

    start_server = websockets.serve(main_logic, '0.0.0.0', 2001)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()


async def recv_msg(websocket, img_dict, q_list, show_id, camera_cfgs, view_all):
    while True:
        recv_text = await websocket.recv()
        print(recv_text)
        if recv_text == 'begin':
            print()
        elif recv_text == 'screenshot':
            cv2.imwrite(f"CameraData/camera{show_id.value}/screenshot.jpg", img_dict[show_id.value]['img'])
            # myclient = pymongo.MongoClient("mongodb://localhost:27017/")
            # mydb = myclient["action_recognition"]
            # mycol = mydb["screenshot"]
            # mydict = {"frame": str(img_dict[show_id.value]['img']), "result": img_dict[show_id.value]['result'], "time": time.time()}
            # mycol.insert_one(mydict)
        elif recv_text == '0':
            show_id.value = 0
        elif recv_text == '1':
            show_id.value = 1
        elif recv_text == '2':
            show_id.value = 2
        elif recv_text == '3':
            show_id.value = 3
        elif recv_text in mouse_events:
            # ip, port, user, password
            PTZControl(recv_text, camera_cfgs[show_id.value][0], 8000,
                       camera_cfgs[show_id.value][1], camera_cfgs[show_id.value][2])
        elif 'x' in recv_text:
            q_list[show_id.value][1].put(json.loads(recv_text))
            # choose_points = json.loads(recv_text)
            # list_area[show_id.value] = [[choose_points[i]['x'], choose_points[i]['y']] for i in range(len(choose_points))]
            # print(choose_points)
        elif recv_text == "clear_region":
            q_list[show_id.value][1].put(recv_text)
        elif recv_text == "view_all":
            view_all.value = 1
        elif recv_text == "view_one":
            view_all.value = 0
        elif 'behaviors_chose' in recv_text:
            q_list[show_id.value][2].put(json.loads(recv_text)['behaviors_chose'])
            # print(behaviors_chose)


async def send_img(websocket, img_dict, show_id, view_all):
    last = [{"No_helmet": -100, "Invalid_clothes": -100, "Playing_phone": -100,
            "Smoking": -100, "Fall_down": -100, "Dangerous_zoom": -100, "Cross_border":-100} for _ in range(4)]
    frame_num = 20
    while True:
        changed = False
        behavior = img_dict[show_id.value]['behavior']
        number = img_dict[show_id.value]['num_person']
        if view_all.value != 1:
            frame = img_dict[show_id.value]['img']
            enconde_data = cv2.imencode('.jpg', frame)[1].tostring()
        else:
            for i in range(4):
                if img_dict[i]:
                    frame = img_dict[i]['img']
                    h, w = frame.shape[:2] # 720 1280 3
                    new_h, new_w = int(h/2), int(w/2)
                    frame_stitch = np.zeros_like(frame, dtype=np.uint8)
                    break
            for i in range(4):
                if img_dict[i]:
                    frame = img_dict[i]['img']
                    frame = cv2.resize(frame, (new_w, new_h))
                    if i == 0:
                        frame_stitch[0:new_h, 0:new_w, :] = frame
                    elif i == 1:
                        frame_stitch[0:new_h, new_w:2*new_w, :] = frame
                    elif i == 2:
                        frame_stitch[new_h:2*new_h, 0:new_w, :] = frame
                    elif i == 3:
                        frame_stitch[new_h:2*new_h, new_w:2*new_w, :] = frame
                else:
                    frame = np.zeros((new_h,new_w,3), dtype=np.uint8)
                    frame = cv2.putText(frame, f"{i+1}", (new_w//2,new_h//2), cv2.FONT_HERSHEY_COMPLEX,
                        2, (0,0,255), thickness=3)
                    if i == 0:
                        frame_stitch[0:new_h, 0:new_w, :] = frame
                    elif i == 1:
                        frame_stitch[0:new_h, new_w:2*new_w, :] = frame
                    elif i == 2:
                        frame_stitch[new_h:2*new_h, 0:new_w, :] = frame
                    elif i == 3:
                        frame_stitch[new_h:2*new_h, new_w:2*new_w, :] = frame
            frame_stitch = cv2.line(frame_stitch, (0,new_h), (w,new_h), (255,255,255))
            frame_stitch = cv2.line(frame_stitch, (new_w,0), (new_w,h), (255,255,255))
            enconde_data = cv2.imencode('.jpg', frame_stitch)[1].tostring()
        try:
            await websocket.send(enconde_data)
            time.sleep(0.03)  # 设置时延
        except Exception as e:
            print(e)
            return True
        if behavior['No_helmet'] and (behavior['Frame'] - last[show_id.value]['No_helmet']) >= frame_num:
            last[show_id.value]['No_helmet'] = behavior['Frame']
            changed = True
        if behavior['Invalid_clothes'] and (behavior['Frame'] - last[show_id.value]['Invalid_clothes']) >= frame_num:
            last[show_id.value]['Invalid_clothes'] = behavior['Frame']
            changed = True
        if behavior['Playing_phone'] and (behavior['Frame'] - last[show_id.value]['Playing_phone']) >= frame_num:
            last[show_id.value]['Playing_phone'] = behavior['Frame']
            changed = True
        if behavior['Smoking'] and (behavior['Frame'] - last[show_id.value]['Smoking']) >= frame_num:
            last[show_id.value]['Smoking'] = behavior['Frame']
            changed = True
        if behavior['Fall_down'] and (behavior['Frame'] - last[show_id.value]['Fall_down']) >= frame_num:
            last[show_id.value]['Fall_down'] = behavior['Frame']
            changed = True
        if behavior['Dangerous_zoom'] and (behavior['Frame'] - last[show_id.value]['Dangerous_zoom']) >= frame_num:
            last[show_id.value]['Dangerous_zoom'] = behavior['Frame']
            changed = True
        if behavior['Cross_border'] and (behavior['Frame'] - last[show_id.value]['Cross_border']) >= frame_num:
            last[show_id.value]['Cross_border'] = behavior['Frame']
            changed = True
        
        if not (behavior['No_helmet'] or behavior['Invalid_clothes'] or behavior['Playing_phone'] or behavior[
            'Smoking'] or behavior['Fall_down'] or behavior['Dangerous_zoom'] or behavior['Cross_border']):
            changed = True

        if changed:
            await websocket.send(json.dumps(behavior))
                # myclient = pymongo.MongoClient("mongodb://localhost:27017/")
                # mydb = myclient["action_recognition"]
                # mycol = mydb["log"]
                # mydict = {"frame": str(frame), "behavior": behavior, "time": time.time()}
                # mycol.insert_one(mydict)
        await websocket.send(json.dumps({'number': number}))