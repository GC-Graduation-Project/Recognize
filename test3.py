#기울어진 오선에 대한 보정 camera.py
import os
import numpy as np
import cv2
import functions as fs
import modules

# 이미지를 읽어옵니다.
resource_path = os.getcwd() + "/resources/"
src = cv2.imread(resource_path+"music5.jpg")

image = modules.deskew(src)
image_0, subimages = modules.camera_remove_noise(image)
image_1= fs.camera_threshold(image)

# 글자나 다른 노이즈를 제거하기위해서 썼는데 잘 안되네

# min_area_threshold = image_1.shape[1]*0.5
# mask = np.zeros(image_1.shape, np.uint8)  # 보표 영역만 추출하기 위해 마스크 생성
# cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(image_1)  # 레이블링
# for i in range(1, cnt):
#     x, y, w, h, area = stats[i]
#     if w > image_1.shape[1] * 0.5 and area > min_area_threshold:  # 보표 영역에만
#         cv2.rectangle(mask, (x, y, w, h), (255, 0, 0), -1)  # 사각형 그리기
#
# masked_image = cv2.bitwise_and(image_1, mask)  # 보표 영역 추출
#
# # 작은 객체를 제거하기 위한 마스크 생성
# small_objects_mask = np.zeros(image_1.shape, np.uint8)
#
# cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(masked_image)  # 레이블링
# for i in range(1, cnt):
#     x, y, w, h, area = stats[i]
#     if h < 12:
#         cv2.rectangle(small_objects_mask, (x, y, w, h), (255, 0, 0), -1)  # 사각형 그리기
#
# masked_image = cv2.bitwise_and(masked_image, small_objects_mask)  # 보표 영역 추출


# 결과를 출력합니다.
cv2.imshow('result', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()