import glob
import CaptchaCracker as cc

from datetime import datetime

# 학습 이미지 데이터 경로
train_img_path_list = glob.glob("data/train_numbers_only/*.png") + glob.glob("data/train_numbers_only_2/*.png")

# 학습 이미지 데이터 크기
img_width = 200
img_height = 50

# 모델 생성 인스턴스
CM = cc.CreateModel(train_img_path_list, img_width, img_height)

# 모델 학습
model = CM.train_model(epochs=100)

# 모델이 학습한 가중치 파일로 저장
date = datetime.now()
model.save_weights(f"model/weights_v2.h5")