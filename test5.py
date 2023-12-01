#기울어진 오선에 대한 보정 camera.py
import os
import numpy as np
import cv2
import functions as fs
import modules

# 이미지를 읽어옵니다.
resource_path = os.getcwd() + "/resources/"
src = cv2.imread(resource_path+"music6.jpg")

stave_list=[] # 해당 악보의 모든 오선 정보를 담고 있는 리스트

image = modules.deskew(src)
image_0, subimages = modules.camera_remove_noise(image)

# 오선 제거된 분할 이미지와 오선 정보에 대해 정규화 수행
normalized_images = []
print(subimages)
for subimage_coords in subimages:
    x, y, w, h = subimage_coords
    subimage = image_0[y:y+h, x:x+w+10] #분할 좌표를 찾아 이미지화 margin을 10px 줬음 안그러면 템플릿 매칭때 오류 발생.
    normalized_image, stave_info = modules.remove_staves(subimage) #오선 제거
    # 이미지 띄우기
    cv2.imshow('subimage', normalized_image)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    normalized_image, stave_info = modules.normalization(normalized_image, stave_info, 10) # 정규화
    normalized_images.append((normalized_image))

    # 마지막 인덱스에 10을 더한 값을 추가
    stave_info.append(stave_info[-1] + 10)

    # 원래 리스트에 중간 값을 추가한 리스트 생성
    new_stave_info = [stave_info[0]]

    for i in range(len(stave_info) - 1):
        mid_value = (stave_info[i] + stave_info[i + 1]) / 2
        new_stave_info.extend([mid_value, stave_info[i + 1]])

    stave_list.append(new_stave_info) # 도 레 미 파 솔 라 시 도

print(stave_list)

# 이미지 띄우기
cv2.imshow('image', image_0)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()