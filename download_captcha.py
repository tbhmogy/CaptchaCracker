import os
import time
import requests
import CaptchaCracker as cc

from datetime import datetime


def download_captcha_images(url, count=10):
    # 모델 설정
    img_width = 200
    img_height = 50
    max_length = 6
    characters = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    weights_path = "model/weights_v2.h5"
    
    # 모델 초기화
    model = cc.ApplyModel(weights_path, img_width, img_height, max_length, characters)
    
    for i in range(count):
        try:
            # 캡차 이미지 다운로드
            response = requests.get(url)
            if response.status_code == 200:
                # 임시 파일명으로 저장
                temp_filename = f"captcha_{int(datetime.now().timestamp()*1000)}.png"
                
                # 이미지 저장
                with open(temp_filename, 'wb') as f:
                    f.write(response.content)
                
                # 예측 수행
                predicted_text = model.predict(temp_filename)
                
                # 파일 이름 변경
                new_filename = f"{predicted_text}.png"
                
                os.rename(temp_filename, new_filename)
                
                print(f"Downloaded and renamed: {new_filename}")
                
                # 서버 부하를 줄이기 위한 대기
                time.sleep(1)
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    # 검증하는 용도로 사용해도 되고, 캡차 이미지는 관련 라이브러리로 직접 생성해서 테스트하는게 빠름
    captcha_url = "https://www.gov.kr/nlogin/captcha/Login"
    download_captcha_images(captcha_url)
