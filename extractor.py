import sys
import os

currentDir = os.path.dirname(os.path.abspath(__file__))

import argparse
import time, datetime
from pathlib import Path
import string
import cv2
import torch
import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel

####################################################################################################

def GenerateRandomString():
    return ''.join(random.choices(string.ascii_lowercase + "_" + string.ascii_uppercase +  string.digits, k=5))

####################################################################################################

def printt(msg):
    print(">>>>" + str(msg))

####################################################################################################

class Extractor():
    def __init__(self, args ):
        torch.no_grad() #what is this?

        # Initialize
        # set_logging()
       
        # Load model     
        weights=""   
        if(args.weights == ''):
            currentDir = os.path.dirname(os.path.abspath(__file__))
            weights = os.path.join(currentDir, "best.pt")
        else:
            weights = args.weights

        self.device = select_device(args.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        if self.half:
            self.model.half()  # to FP16

        trace = False
        self.imgsz=args.img_size
        if trace:            
            self.model = TracedModel(self.model, self.device, self.imgsz)
        self.conf_thres = args.conf_thres
        self.skip_frame = args.skip_frame
        self.ignore_no_object = args.ignore_no_object
        self.draw_img = args.draw_img
        self.save_txt = args.save_txt
        if(len(args.resize) == 2):
            self.resize_width = args.resize[0]
            self.resize_height = args.resize[1]
        else:
            self.resize_width = 0
            self.resize_height = 0

        self.strTime = ""
        self.fileIndex = 0

    ####################################################################################################

    def GenerateRandFileName(self, ext=".jpg"):    
        dateVN = datetime.datetime.utcnow() + datetime.timedelta(hours=7)
        strTime = dateVN.strftime("%Y-%m-%d_%H-%M-%S")
        if(strTime != self.strTime):
            self.strTime = strTime
            self.fileIndex = 0
        self.fileIndex += 1
        return self.strTime + "_" + "{:02d}".format(self.fileIndex) + GenerateRandomString() + ext

    ####################################################################################################

    def Detect(self, source, outputPath=""):
        

        line_thickness=1        
        
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check img_size


        # Set Dataloader
        dataset = LoadImages(source, img_size=self.imgsz, stride=stride)

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


        classes = []

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        t0 = time.time()

        #main loop
        countFrame = 0
        for path, img, im0s, vid_cap in dataset:
            if(self.skip_frame != 0):
                if(countFrame < self.skip_frame):
                    countFrame += 1
                    continue
                else:
                    countFrame = 0

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            #t1 = time_synchronized()
            pred = self.model(img, augment=False)[0]

            # Apply NMS
            iou_thres=0.45
            pred = non_max_suppression(pred, self.conf_thres, iou_thres, classes=None, agnostic=False)
            #t2 = time_synchronized()

            rows = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image          
                p, s, mat, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                #gn = torch.tensor(mat.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                [im_height, im_width] = mat.shape[:2]
                if len(det):
                    # Rescale boxes from img_size to mat size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], mat.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):  
                        cls = int(cls)
                        left = int(xyxy[0])
                        top = int(xyxy[1])
                        right = int(xyxy[2])
                        bot = int(xyxy[3])
                        xCenter = ((left + right) / 2) / im_width
                        yCenter = ((top + bot) / 2) / im_height
                        width = (right - left) / im_width
                        height = (bot - top) / im_height
                        rows.append('{0} {1} {2} {3} {4}'.format(cls, xCenter, yCenter, width, height) )


                        if self.draw_img:  # Add bbox to image
                            classname = names[cls]
                            label = f'{classname} {conf:.2f}'
                            plot_one_box(xyxy, mat, label=label, color=colors[int(cls)], line_thickness=line_thickness)

                                    

                # Save results (image with detections)
                if self.save_txt:
                    if(self.ignore_no_object and len(rows) == 0):
                        continue

                    _randFilename = self.GenerateRandFileName(".jpg")
                    outputPath = os.path.join("extracted", _randFilename)
                    if(self.resize_width > 0 and self.resize_height > 0):
                        mat = cv2.resize(mat, (self.resize_width, self.resize_height))
                    cv2.imwrite(outputPath, mat)

                    textFile = os.path.join("extracted", _randFilename.replace(".jpg", ".txt"))
                    with open(textFile, 'w') as f:
                        for line in rows:
                            f.write(f"{line}\n")

                    print(f" : {outputPath}\n")
        return mat, classes
    
####################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--draw-img', action='store_true', help='draw rectangle and class name')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')    
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--skip-frame', type=int, default=0, help='skip num frame')
    parser.add_argument('--ignore-no-object', action='store_true', help='no save if no object detected')
    parser.add_argument('--resize', nargs='+', type=int, help='resize output image: --resize 1280 720')
    args = parser.parse_args()
    printt(args)
    #check_requirements(exclude=('pycocotools', 'thop'))
    extractor = Extractor(args)
    mat, classes = extractor.Detect(args.source)
