import os
import cv2
import modules  # 모듈이 있는 파일의 이름에 따라 수정이 필요할 수 있습니다.

# 현재 디렉토리 설정
resource_path = os.getcwd() + "/train/images/"
output_path = os.getcwd() + "/train/processed_images/"

# 결과 디렉토리가 없으면 생성
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 모든 이미지 파일에 대해 처리
for filename in os.listdir(resource_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 필요에 따라 확장자를 수정하세요.
        # 이미지 불러오기
        image_0 = cv2.imread(os.path.join(resource_path, filename))

        # 1. 보표 영역 추출 및 그 외 노이즈 제거
        image_1, subimage = modules.remove_noise(image_0)

        # 2. 오선 제거
        image_2, staves = modules.remove_staves(image_1)

        image_2 = cv2.bitwise_not(image_2)

        # 결과 이미지 저장
        output_filename = os.path.join(output_path, filename)
        cv2.imwrite(output_filename, image_2)

        print(f"Processed {filename} and saved as {output_filename}")