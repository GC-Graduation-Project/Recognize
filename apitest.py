import requests as r
import json
from pprint import pprint


def send_request(image='resources/music1.jpg'):
    files = {'file': ('image_filename.png', open(image, 'rb'))}
    res = r.post("http://localhost:8000/uploadfile/", files=files)

    if res.status_code == 200:
        print("Request Successful")
        pprint(json.loads(res.text))
    else:
        print(f"Request Failed with status code {res.status_code}")


send_request()
