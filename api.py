from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from modules import deskew, remove_noise, digital_preprocessing
from pitchDetection import detect

app = FastAPI()
@app.get("/")
def HelloWorld():
    return FastAPI

@app.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 이미지 처리
    image = deskew(img)
    image_0, subimages = remove_noise(image)
    normalized_images, stave_list = digital_preprocessing(image_0, subimages)

    # YOLO 모델 적용
    yolo_results = []
    for img in normalized_images:
        img = cv2.bitwise_not(img)
        result = detect(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        yolo_results.append(result)


    # 결과를 JSON 형식으로 반환
    return JSONResponse(content={"yolo_results": yolo_results})
