import os
import cv2
import torch
import numpy as np

from numpy import random
from models.experimental import attempt_load
from utils.dataloaders import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_boxes
from utils.plots import plot_one_box
from utils.torch_utils import select_device


SOURCE = os.getcwd() + "/result/" + "result/normalized_1_object_3_pillar_1.png"
WEIGHTS = os.getcwd() + "/models/" + 'best1.pt'
IMG_SIZE = 64
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.6
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False

# stave_list가 들어올 경우 result_list에서

def detectBeat(image):
    source, weights, imgsz = SOURCE, WEIGHTS, IMG_SIZE
    source=image
    result_list=[] # 인식 결과를 담은 list 생성
    # Initialize
    device = select_device('cpu') # 일단 CPU defalut로
    # device = select_device(DEVICE)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # Load image
    img0 = image  # BGR
    assert img0 is not None, 'Image Not Found ' + source

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=AUGMENT)[0]

    # Apply NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

    # Process detections
    det = pred[0]

    s = ''
    s += '%gx%g ' % img.shape[2:]  # print string

    if len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        sorted_det = sorted(det, key=lambda x: (x[0] + x[2]) / 2)  # x 중점을 기준으로 정렬

        for *xyxy, conf, cls in sorted_det:
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

            # 추가: bounding box의 중점 좌표와 확률(label 이름 포함) 출력
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            x_center, y_center = round(float(x_center), 2), round(float(y_center), 2)  # 텐서를 숫자로 변환 및 라운딩
            # print(f'Box Center: ({x_center}, {y_center}), Confidence: {conf:.2f}, Class: {names[int(cls)]}') # 인식 결과를 x 좌표 순서대로 정렬
            result_list.append(names[int(cls)])

    # print(result_list)
    # # Stream results
    # print(s)
    return result_list