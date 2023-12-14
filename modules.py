# modules.py
import cv2
import numpy as np
import functions as fs
from pitchDetection import detect, detect1
from beatDetection import detectBeat

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

    # 보표 추출한 이미지에서 오선 탐색후 이미지 나눔

    subimage_array = []

    for subimage_coords in subimages:
        x, y, w, h = subimage_coords
        subimage_1 = masked_image[y:y + h - 2, x:x + w + 10]  # 분할 좌표를 찾아 이미지화 margin을 10px 줬음 안그러면 템플릿 매칭때 오류 발생.
        height, width = subimage_1.shape
        staves = []  # 오선의 좌표들이 저장될 리스트

        for row in range(height):
            pixels = 0
            for col in range(width):
                pixels += (subimage_1[row][col] == 255)  # 한 행에 존재하는 흰색 픽셀의 개수를 셈
            if pixels >= width * 0.5:  # 이미지 넓이의 50% 이상이라면
                if len(staves) == 0 or abs(staves[-1][0] + staves[-1][1] - row) > 1:  # 첫 오선이거나 이전에 검출된 오선과 다른 오선
                    staves.append([row, 0])  # 오선 추가 [오선의 y 좌표][오선 높이]
                else:  # 이전에 검출된 오선과 같은 오선
                    staves[-1][1] += 1  # 높이 업데이트

        if (len(staves) > 5):  # 악보 오선이 5개 이상 (큰 보표면)
            # 이미지를 수평으로 반으로 나누기
            half_height = int((staves[4][0]+staves[5][0])/2)
            subimage_array.append([x, y, w, half_height])  # 위쪽 절반의 좌표
            subimage_array.append([x, y + half_height, w, half_height])  # 아래쪽 절반의 좌표

        else:
            subimage_array.append([x, y, w, h])

    return masked_image, subimage_array

def camera_remove_noise(image):
    image = fs.camera_threshold(image) # 이미지 이진화
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

    # 보표 추출한 이미지에서 오선 탐색후 이미지 나눔

    subimage_array = []

    for subimage_coords in subimages:
        x, y, w, h = subimage_coords
        subimage_1 = masked_image[y:y + h-2, x:x + w + 10]  # 분할 좌표를 찾아 이미지화 margin을 10px 줬음 안그러면 템플릿 매칭때 오류 발생.
        height, width = subimage_1.shape
        staves = []  # 오선의 좌표들이 저장될 리스트

        for row in range(height):
            pixels = 0
            for col in range(width):
                pixels += (subimage_1[row][col] == 255)  # 한 행에 존재하는 흰색 픽셀의 개수를 셈
            if pixels >= width * 0.5:  # 이미지 넓이의 50% 이상이라면
                if len(staves) == 0 or abs(staves[-1][0] + staves[-1][1] - row) > 1:  # 첫 오선이거나 이전에 검출된 오선과 다른 오선
                    staves.append([row, 0])  # 오선 추가 [오선의 y 좌표][오선 높이]
                else:  # 이전에 검출된 오선과 같은 오선
                    staves[-1][1] += 1  # 높이 업데이트

        print(len(staves))

        if(len(staves)>5) : # 악보 오선이 5개 이상 (큰 보표면)
            # 이미지를 수평으로 반으로 나누기
            half_height = int((staves[4][0]+staves[5][0])/2)
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

def digital_preprocessing(image, subimage_array):
    image_0 = image.copy()
    stave_list = []
    normalized_images = []
    for subimage_coords in subimage_array:
        x, y, w, h = subimage_coords
        subimage = image_0[y:y + h + 5, x:x + w]
        normalized_image, stave_info = remove_staves(subimage)  # 오선 제거
        normalized_image, stave_info = normalization(normalized_image, stave_info, 10)  # 정규화
        normalized_images.append((normalized_image))

        # 마지막 인덱스에 10을 더한 값을 추가
        stave_info.append(stave_info[-1] + 10)

        # 원래 리스트에 중간 값을 추가한 리스트 생성
        new_stave_info = [stave_info[0]]

        for i in range(len(stave_info) - 1):
            mid_value = (stave_info[i] + stave_info[i + 1]) / 2
            new_stave_info.extend([mid_value, stave_info[i + 1]])

        stave_list.append(new_stave_info)  # 도 레 미 파 솔 라 시 도

    return normalized_images, stave_list

def pitch_extraction(stave_list, normalized_images):
    original_list = []
    final_result = []
    clef_list = []
    ind = 0

    for img in normalized_images:
        img = cv2.bitwise_not(img)
        result = detect(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))  # YOLO모델에는 BGR로 들어가야하기때문에 convert해서 넣어줌.
        result2 = detect1(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        clef_list.append(result2)
        original_list.append(result)

    for clef, notes in zip(clef_list, original_list):
        notes.insert(0, clef[0])
        notes = fs.add_dot(notes)
        note_tmp_list = fs.mapping_notes(stave_list[ind], notes)
        final_result.append(note_tmp_list)
        ind += 1

    return original_list, final_result
      
def beat_extraction(normalized_images):
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
                temp_list.append([ (y+h)/2, object_roi, x])

            # 객체를 개별 파일로 저장합니다 (기둥 개수에 따라 분리)
            for j in range(note_pillar_count):
                x1 = x + j * (w // note_pillar_count)  # 분리된 객체의 왼쪽 x 좌표
                x2 = x1 + (w // note_pillar_count)  # 분리된 객체의 오른쪽 x 좌표
                object_pillar = object_roi[:, x1 - (x - 2): x2 - (x - 2)]  # 기둥에 해당하는 부분 추출
                temp_list.append([(y+h)/2, object_pillar, x])
        split_list.append(temp_list)

    note_list = []
    rest_list = []
    recognition_list = []

    # recognition_list에 있는 이미지를 참조하여 디텍트 결과 반환
    for i, temp_list in enumerate(split_list):
        temp=[]
        temp_note = []
        temp_rest = []
        for j, (center_y, object_pillar, x) in enumerate(temp_list):
            object_pillar= cv2.bitwise_not(object_pillar)
            result = detectBeat(cv2.cvtColor(object_pillar, cv2.COLOR_GRAY2BGR))
            # 결과에 대한 처리 수행 (예: 리스트에 추가)
            if(result==[]):
                continue
            temp.append([center_y, result[0], x])

        recognition_list.append(temp)

    # recognition_list에는 지금 담겨져있는 총 인식 결과를 note따로 rest따로 분리
    for temp_list in recognition_list:
        temp_note = []
        temp_rest = []
        for item in temp_list:
            if item and item[1].endswith('_note'):
                temp_note.append(item)
            elif item and item[1].endswith('_rest'):
                temp_rest.append(item)
        note_list.append(temp_note)
        rest_list.append(temp_rest)

    return recognition_list, note_list, rest_list
