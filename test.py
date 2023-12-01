import cv2
import os
import numpy as np
import functions as fs
import modules

resource_path = os.getcwd() + "/resources/"
src = cv2.imread(resource_path+"music6.jpg")

image = modules.deskew(src)
image_0, subimages = modules.remove_noise(image)


normalized_images, stave_list = modules.digital_preprocessing(image_0, subimages)

# 결과를 저장할 디렉토리
result_dir = 'result'
os.makedirs(result_dir, exist_ok=True)

# normalized_images 배열의 각 이미지에 대해 처리를 반복
for idx, normalized_image in enumerate(normalized_images):
    # 레이블링을 사용한 검출
    closing_image = fs.closing(normalized_image)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(closing_image)  # 모든 객체 검출하기

    # stats 배열을 x 좌표를 기준으로 정렬
    sorted_stats = sorted(stats[1:], key=lambda x: x[0])

    # 모든 객체를 반복하며 (배경 제외)
    for i in range(1, cnt):
        (x, y, w, h, area) = sorted_stats[i - 1]

        # 작은 객체는 무시합니다 (필요에 따라 최소 크기 임계값을 조정하세요)
        if w < fs.weighted(5) or h < fs.weighted(5):
            continue

        # 객체의 ROI를 추출합니다
        object_roi = normalized_image[y - 2:y + h + 2, x - 2: x + w + 2]

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

        # 현재 객체의 결과를 출력합니다
        print(f"객체 {i + 1} - 음표 기둥 개수: {note_pillar_count}")

        # 객체를 개별 파일로 저장합니다 (기둥 개수에 따라 분리)
        for j in range(note_pillar_count):
            x1 = x + j * (w // note_pillar_count)  # 분리된 객체의 왼쪽 x 좌표
            x2 = x1 + (w // note_pillar_count)  # 분리된 객체의 오른쪽 x 좌표
            object_pillar = object_roi[:, x1 - (x - 2): x2 - (x - 2)]  # 기둥에 해당하는 부분 추출
            filename = f"{result_dir}/normalized_{idx + 1}_object_{i + 1}_pillar_{j + 1}.png"
            cv2.imwrite(filename, object_pillar)


# 음표 기둥 개수가 표시된 최종 결과 이미지를 표시합니다
cv2.imshow('result_image', normalized_image)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
