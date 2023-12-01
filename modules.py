# modules.py
import cv2
import numpy as np
import functions as fs


def deskew(image): # 이미지 보정 함수 작성 완료.
    src=image.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    length = src.shape[1]
    canny = cv2.Canny(gray, 5000, 1500, apertureSize=5, L2gradient=True)
    lines = cv2.HoughLinesP(canny, 0.9, np.pi / 180, 90, minLineLength=length * 0.70,
                            maxLineGap=100)  # 우리가 탐색할 선은 오선이므로 이미지의 70%이상인 선을 가지고 있으면 오선으로 간주

    # 모든 선의 기울기를 계산하고 평균을 구합니다.
    angles = []
    for line in lines:
        line = line[0]
        angle = np.arctan2(line[3] - line[1], line[2] - line[0]) * 180. / np.pi
        if (angle >= 0):  # 직선 무시 일반적인 악보는 기울기가 0.0이기때문
            angles.append(angle)
    avg_angle = np.mean(angles)

    # 이미지의 중심을 기준으로 회전 변환 행렬을 계산합니다.
    h, w = src.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, avg_angle, 1)
    rotated = cv2.warpAffine(src, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    output = cv2.fastNlMeansDenoising(rotated, None, 10, 7, 21)
    return output

def remove_noise(image):
    image = fs.threshold(image)  # 이미지 이진화
    mask = np.zeros(image.shape, np.uint8)  # 보표 영역만 추출하기 위해 마스크 생성
    subimages = []  # subimage들을 저장할 리스트
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(image)  # 레이블링
    for i in range(1, cnt):
        x, y, w, h, area = stats[i]
        if w > image.shape[1] * 0.5:  # 보표 영역에만
            cv2.rectangle(mask, (x, y, w, h), (255, 0, 0), -1)  # 사각형 그리기

            # 원본 이미지에서 사각형 영역 추출하여 subimages 리스트에 추가
            subimage =[x, y, w, h]
            subimages.append(subimage)

    masked_image = cv2.bitwise_and(image, mask)  # 보표 영역 추출

    return masked_image, subimages

def camera_remove_noise(image):
    image = fs.camera_threshold(image) # 이미지 이진화
    mask = np.zeros(image.shape, np.uint8)  # 보표 영역만 추출하기 위해 마스크 생성
    subimages = []  # subimage들을 저장할 리스트
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(image)  # 레이블링
    for i in range(1, cnt):
        x, y, w, h, area = stats[i]
        if w > image.shape[1] * 0.7:  # 보표 영역에만
            cv2.rectangle(mask, (x, y, w, h), (255, 0, 0), -1)  # 사각형 그리기

            # 원본 이미지에서 사각형 영역 추출하여 subimages 리스트에 추가
            subimage =[x, y, w, h]
            subimages.append(subimage)

    masked_image = cv2.bitwise_and(image, mask)  # 보표 영역 추출

    # 보표 추출한 이미지에서 오선 탐색후 이미지 나눔

    subimage_array = []

    for subimage_coords in subimages:
        x, y, w, h = subimage_coords
        subimage_1 = masked_image[y:y + h-2, x:x + w + 10]  # 분할 좌표를 찾아 이미지화 margin을 10px 줬음 안그러면 템플릿 매칭때 오류 발생.
        # 이미지 띄우기
        cv2.imshow('result_subimage', subimage_1)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
        height, width = subimage_1.shape
        staves = []  # 오선의 좌표들이 저장될 리스트

        for row in range(height):
            pixels = 0
            for col in range(width):
                pixels += (subimage_1[row][col] == 255)  # 한 행에 존재하는 흰색 픽셀의 개수를 셈
            if pixels >= width * 0.4:  # 이미지 넓이의 50% 이상이라면
                if len(staves) == 0 or abs(staves[-1][0] + staves[-1][1] - row) > 1:  # 첫 오선이거나 이전에 검출된 오선과 다른 오선
                    staves.append([row, 0])  # 오선 추가 [오선의 y 좌표][오선 높이]
                else:  # 이전에 검출된 오선과 같은 오선
                    staves[-1][1] += 1  # 높이 업데이트

        print(len(staves))

        if(len(staves)>5) :
            # 이미지를 수평으로 반으로 나누기
            half_height = height // 2
            subimage_array.append([x, y, w, half_height])  # 위쪽 절반의 좌표
            subimage_array.append([x, y + half_height, w, half_height])  # 아래쪽 절반의 좌표

        else :
            subimage_array.append([x,y,w,h])

    return masked_image, subimage_array


def remove_staves(image):
    height, width = image.shape
    staves = []  # 오선의 좌표들이 저장될 리스트

    for row in range(height):
        pixels = 0
        for col in range(width):
            pixels += (image[row][col] == 255)  # 한 행에 존재하는 흰색 픽셀의 개수를 셈
        if pixels >= width * 0.4:  # 이미지 넓이의 50% 이상이라면
            if len(staves) == 0 or abs(staves[-1][0] + staves[-1][1] - row) > 1:  # 첫 오선이거나 이전에 검출된 오선과 다른 오선
                staves.append([row, 0])  # 오선 추가 [오선의 y 좌표][오선 높이]
            else:  # 이전에 검출된 오선과 같은 오선
                staves[-1][1] += 1  # 높이 업데이트

    for staff in range(len(staves)):
        top_pixel = staves[staff][0]  # 오선의 최상단 y 좌표
        bot_pixel = staves[staff][0] + staves[staff][1]  # 오선의 최하단 y 좌표 (오선의 최상단 y 좌표 + 오선 높이)
        for col in range(width):
            if image[top_pixel - 1][col] == 0 and image[bot_pixel + 1][col] == 0:  # 오선 위, 아래로 픽셀이 있는지 탐색
                for row in range(top_pixel, bot_pixel + 1):
                    image[row][col] = 0  # 오선을 지움

    return image, [x[0] for x in staves]

def normalization(image, staves, standard):
    avg_distance = 0
    lines = int(len(staves) / 5)  # 보표의 개수
    for line in range(lines):
        for staff in range(4):
            staff_above = staves[line * 5 + staff]
            staff_below = staves[line * 5 + staff + 1]
            avg_distance += abs(staff_above - staff_below)  # 오선의 간격을 누적해서 더해줌
    avg_distance /= len(staves) - lines  # 오선 간의 평균 간격

    height, width = image.shape  # 이미지의 높이와 넓이
    weight = standard / avg_distance  # 기준으로 정한 오선 간격을 이용해 가중치를 구함
    new_width = int(width * weight)  # 이미지의 넓이에 가중치를 곱해줌
    new_height = int(height * weight)  # 이미지의 높이에 가중치를 곱해줌

    image = cv2.resize(image, (new_width, new_height))  # 이미지 리사이징
    # ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 이미지 이진화 !! 이진화 작업시 음표 손실됨
    staves = [x * weight for x in staves]  # 오선 좌표에도 가중치를 곱해줌

    return image, staves