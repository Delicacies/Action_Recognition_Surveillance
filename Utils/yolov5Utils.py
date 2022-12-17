import cv2
import threading

class MyThread(threading.Thread):  
    def __init__(self, func, args=()):  
        super(MyThread, self).__init__()  
        self.func = func  
        self.args = args  
  
    def run(self):  
        self.result = self.func(*self.args)
  
    def get_result(self):  
        try:  
            return self.result  
        except Exception as e:  
            return None  


def get_person_crops(img, boxes, ignore_threshold=0.01, expand_ratio=0.15, cut_legs=False):
    crops, new_rects = get_person_from_rect(img, boxes, ignore_threshold=ignore_threshold, expand_ratio=expand_ratio, cut_legs=cut_legs)
    return crops, new_rects  #pt1, pt2

def get_person_from_rect(image, results, ignore_threshold=0.05, expand_ratio=0.15, cut_legs=False):
    # crop the person result from image
    # det_results = results
    # mask = det_results[:, 1] > det_threshold  # 黑屏时没有物体，下标1可能报错 @yjy
    # valid_rects = det_results[mask]

    imgh, imgw = image.shape[:2]  # 480*640*3 @yjy
    rect_images = []
    new_rects = []
    # org_rects = []
    for rect in results:
        rect_image, new_rect = expand_crop(image, rect, expand_ratio=expand_ratio, cut_legs=cut_legs)  # crop label=person @yjy
        if rect_image is None or rect_image.size == 0:  
            continue
        # cv2.imshow("Crops", rect_image[:, :, ::-1]) # RGB HWC
        # if cv2.waitKey(1) & 0xFF==ord('q'):
        #     break
        pheight, pwidth = new_rect[3]-new_rect[1], new_rect[2]-new_rect[0]   # xmin,ymin,xmax,ymax, pheight=ymax-ymin @yjy
        if pwidth*pheight < ignore_threshold*imgh*imgw:  # ignore small person @yjy
            continue
        rect_images.append(rect_image)
        new_rects.append(new_rect)
        # org_rects.append(org_rect)
    return rect_images, new_rects #, org_rects

def expand_crop(images, rect, expand_ratio=0.15, cut_legs=False):
    imgh, imgw = images.shape[:2] # HWC
    xmin, ymin, xmax, ymax = [int(x) for x in rect.tolist()]
    # if label != 0:   # crop person only @yjy
    #     return None, None
    org_rect = [xmin, ymin, xmax, ymax]
    h_half = (ymax - ymin) * (1 + expand_ratio) / 2.
    w_half = (xmax - xmin) * (1 + expand_ratio) / 2.
    # if h_half > w_half * 4 / 3:
    #     w_half = h_half * 0.75
    center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
    ymin = max(0, int(center[0] - h_half * 1.1))  # 增加向上扩张
    # ymax = min(imgh - 1, int(center[0] + h_half))
    if cut_legs:
        ymax = min(imgh - 1, int(center[0])) # 减小向下扩张 1/8
    else:
        ymax = min(imgh - 1, int(center[0] + h_half*2/3)) # 减小向下扩张 1/2
    xmin = max(0, int(center[1] - w_half))  
    xmax = min(imgw - 1, int(center[1] + w_half))
    return images[ymin:ymax, xmin:xmax, :], [xmin, ymin, xmax, ymax] #, org_rect