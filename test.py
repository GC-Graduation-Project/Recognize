import cv2
import os
import numpy as np
import functions as fs
import modules
from beatDetection import detectBeat

resource_path = os.getcwd() + "/resources/"
src = cv2.imread(resource_path + "music.jpg")

image = modules.deskew(src)
image_0, subimages = modules.remove_noise(image)

normalized_images, stave_list = modules.digital_preprocessing(image_0, subimages)

split_list = []


# normalized_images 배열의 각 이미지에 대해 처리를 반복
for idx, normalized_image in enumerate(normalized_images):
    # 레이블링을 사용한 검출
    closing_image = fs.closing(normalized_image)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(closing_image)  # 모든 객체 검출하기
    temp_list = []

    # stats 배열을 x 좌표를 기준으로 정렬
    sorted_stats = sorted(stats[1:], key=lambda x: x[0])

    # 모든 객체를 반복하며 (배경 제외)
    for i in range(1, cnt):
        (x, y, w, h, area) = sorted_stats[i - 1]

        # 작은 객체는 무시합니다 (필요에 따라 최소 크기 임계값을 조정하세요)
        if w < fs.weighted(5) or h < fs.weighted(5):
            continue

        # 객체의 ROI를 추출합니다
        object_roi = normalized_image[y - 2:y + h + 4, x - 2: x + w + 4]
        # 객체의 높이와 너비를 계산합니다
        height, width = object_roi.shape

        # 수직 히스토그램을 저장할 배열을 생성합니다
        histogram = np.zeros((height, width), np.uint8)

        # 각 열에 대한 수직 히스토그램을 계산합니다
        for col in range(width):
            pixels = np.count_nonzero(object_roi[:, col])
            histogram[height - pixels:, col] = 255  # 히스토그램을 아래부터 채웁니다

        # 수직 히스토그램을 기반으로 객체 내의 음표 기둥 개수를 계산합니다
        note_pillar_count = 0
        threshold = 30  # 열을 음표 기둥으로 간주할 임계값
        max_duplicate_distance = 3  # 중복 기둥으로 간주할 최대 거리

        previous_pillar_position = None

        for col in range(width):
            if np.count_nonzero(histogram[:, col]) >= threshold:
                if previous_pillar_position is None or abs(col - previous_pillar_position) >= max_duplicate_distance:
                    note_pillar_count += 1
                previous_pillar_position = col
        if(note_pillar_count==0):
            temp_list.append([object_roi, x, (y+h)/2])

        # 객체를 개별 파일로 저장합니다 (기둥 개수에 따라 분리)
        for j in range(note_pillar_count):
            x1 = x + j * (w // note_pillar_count)  # 분리된 객체의 왼쪽 x 좌표
            x2 = x1 + (w // note_pillar_count)  # 분리된 객체의 오른쪽 x 좌표
            object_pillar = object_roi[:, x1 - (x - 2): x2 - (x - 2)]  # 기둥에 해당하는 부분 추출
            temp_list.append([object_pillar, x, (y+h)/2])
    split_list.append(temp_list)

note_list = []
rest_list = []
recognition_list = []

# recognition_list에 있는 이미지를 참조하여 디텍트 결과 반환
for i, temp_list in enumerate(split_list):
    temp=[]
    temp_note = []
    temp_rest = []
    for j, (object_pillar, x,center_y) in enumerate(temp_list):
        object_pillar= cv2.bitwise_not(object_pillar)
        result = detectBeat(cv2.cvtColor(object_pillar, cv2.COLOR_GRAY2BGR))
        # 결과에 대한 처리 수행 (예: 리스트에 추가)
        if(result==[]):
            continue
        temp.append([result, x, center_y])

    recognition_list.append(temp)

# recognition_list에는 지금 담겨져있는 총 인식 결과를 note따로 rest따로 분리
for temp_list in recognition_list:
    temp_note = []
    temp_rest = []
    for item in temp_list:
        if item and item[0][0].endswith('_note'):
            temp_note.append(item)
        elif item and item[0][0].endswith('_rest'):
            temp_rest.append(item)
    note_list.append(temp_note)
    rest_list.append(temp_rest)

print(note_list)
print(rest_list)