# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams, LoadCrops  # added @yjy
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, is_ascii, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

from playingCellPhoneUtils import phone_smoke_position_valid  # @yjy

class Yolov5Detector(object):
    """Load trained YOLOv5 detection model."""
    def __init__(self,
            weights= 'output/exp23/weights/best.pt',  # model.pt path(s)
            source='0',  # file/dir/URL/glob, 0 for webcam
            imgsz=[480],  # inference size (pixels)
            conf_thres=0.7,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            phone_smoke_filter=False, # use poses to filt phones or smoke by position
            helmet_filter=False, # use higher threshold for helmet
            ):
        imgsz *= 2 if len(imgsz) == 1 else 1 
        self.source = source
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.view_img = view_img
        self.save_txt = save_txt
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.save_crop = save_crop
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.half = half
        self.phone_smoke_filter = phone_smoke_filter
        self.helmet_filter = helmet_filter
        self.save_img = not nosave and not source.endswith('.txt')  # save inference images
        self.webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        self.save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        # self.device = select_device(device)
        self.device = device
        # self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        # w = weights[0] if isinstance(weights, list) else weights
        # classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        # check_suffix(w, suffixes)  # check weights have acceptable suffix
        # pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
        # stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        weights = str(ROOT) + '/' + weights
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        
        # self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        # ascii_names = {'helmet':"安全帽", 'no_helmet':"未戴安全帽", 'cell phone':"手机", 'smoke':"香烟"}
        # self.names = [ascii_names[x] for x in self.names]
        self.names = ["安全帽", "未戴安全帽", "手机", "香烟"]

        if self.half:
            self.model.half()  # to FP16
        
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.ascii = is_ascii(self.names)  # names are not ascii (use PIL for UTF-8)
        self.font = "Arial.ttf" if self.ascii else "simheittf.ttf"

    def transform_box(self, boxes, ul):
        num_box = boxes.shape[0]
        ul = torch.Tensor(ul).to(self.device)
        new_boxes = torch.zeros(boxes.size())
        new_boxes[:, 0] = boxes[:, 0] + ul[0].repeat(1, num_box)
        new_boxes[:, 1] = boxes[:, 1] + ul[1].repeat(1, num_box)
        new_boxes[:, 2] = boxes[:, 2] + ul[0].repeat(1, num_box)
        new_boxes[:, 3] = boxes[:, 3] + ul[1].repeat(1, num_box)
        return new_boxes

    def dataOnPerson(self, imgs=None, ori_frame=None, new_rects=None):
        # Dataloader
        # self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
        # new_rect_ims, orig_rect_ims = LoadCrops(imgs, img_size=self.imgsz, stride=self.stride)
        # self.frame = ori_frame.copy()[:, :, ::-1] # RGB to BGR HWC 480-640
        # self.frame = np.ascontiguousarray(self.frame)
        dataset = LoadCrops(imgs, new_rects=new_rects, img_size=self.imgsz, stride=self.stride, auto=False)  # auto=False for rectangle
        # bs = 1  # batch_size
        # self.vid_path, self.vid_writer = [None] * bs, [None] * bs
        return dataset

    @torch.no_grad()
    def detectOnPerson(self, dataset=None, frame=None, poses=None, show_detect_classes=[1,2,3]):
        # Run inference
        # if self.device.type != 'cpu':
        #     self.model(torch.zeros(1, 3, *self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        dt, seen = [0.0, 0.0, 0.0], 0
        result = []  # bbexes, conf, class
        # for [img, im0s, rect] in dataset:  # per img: BGR
        if len(dataset):
            img, im0s, rect = dataset  # all imgs: RGB
            t1 = time_sync()
            img = torch.from_numpy(img).to(self.device) # BCHW RGB
            # img = img.half() if self.half else img.float()  # uint8 to fp16/32

            img = img / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim
            t2 = time_sync()
            # dt[0] += t2 - t1

            # Inference
            pred = self.model(img, augment=self.augment, visualize=False)[0]
            t3 = time_sync()
            # dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            # dt[2] += time_sync() - t3

            # Process predictions
            annotator = Annotator(frame, line_width=self.line_thickness, font=self.font, pil=not self.ascii)
            # s = '%gx%g ' % img.shape[2:]  # print string
            for i, det in enumerate(pred):  # per image
                seen += 1
                im0 = im0s[i]  # im0 not used
                '''
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                '''
                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # imc = im0.copy() if self.save_crop else im0  # for save_crop
                # annotator = Annotator(im0, line_width=self.line_thickness, pil=not self.ascii)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    det[:, :4] = self.transform_box(det[:, :4], rect[i][:2])  # transform crop coords to frame coords
                    # Print results
                    # for c in det[:, -1].unique():
                    #     n = (det[:, -1] == c).sum()  # detections per class
                        # s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    result.append(det.tolist())  # n个检测框
                    # Write results
                    for *xyxy, conf, cls in reversed(det): # 遍历所有目标
                        if self.phone_smoke_filter:
                            if not phone_smoke_position_valid(poses, det, frame.shape[:2]):
                                print("Invalid phone!!!")
                                continue
                        if self.helmet_filter: # 检测安全帽使用更高阈值 0:helmet, 1:no_helmet,手机使用低一些阈值
                            if cls in [0,1] and conf<0.6:  
                                continue
                        if int(cls) not in show_detect_classes:
                            continue
                        '''
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        '''
                        if self.save_img or self.save_crop or self.view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if self.save_crop:
                                save_one_box(xyxy, imc, file=self.save_dir / 'crops' / self.names[c] / f'0.jpg', BGR=True)

            # Print time (inference-only)
            # print(f'{s}Done. ({t3 - t2:.3f}s)')
            # Stream results
            if self.view_img:
                im0 = annotator.result()  # RGB
                im0 = im0[:, :, ::-1] # BGR
                cv2.imshow("Person", im0)
                cv2.waitKey(1)  # 1 millisecond

            if not self.ascii:
                frame = annotator.result()
                
            # Save results (image with detections)
            '''
            if self.save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
            '''

        # Print results
        # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.imgsz)}' % t)
        return frame, result  # RGB, [[[xyxy,conf,cls]]]
        '''
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
        '''

    '''
    def dataLoader(self, imgs=None):
        # Dataloader
        if self.webcam:
            # view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(self.source, img_size=self.imgsz, stride=self.stride)
            bs = len(self.dataset)  # batch_size
        else:
            self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
            bs = 1  # batch_size
        # self.vid_path, self.vid_writer = [None] * bs, [None] * bs
    '''


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True, action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    # run(**vars(opt))
    model = Yolov5Detector()
    model.dataLoader()
    model.inference()


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)








