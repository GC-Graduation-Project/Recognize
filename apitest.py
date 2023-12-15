import requests as r

def send_request(image='resources/music.jpg'):
    files = {'file': ('image_filename.jpg', open(image, 'rb'))}
    res = r.post("http://localhost:8000/uploadfile/", files=files)

    if res.status_code == 200:
        print("Request Successful")
        print(res.text)  # 응답을 일반 텍스트로 출력
    else:
        print(f"Request Failed with status code {res.status_code}")

send_request()
