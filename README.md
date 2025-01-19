# ViViT를 활용한 한국어 수어 번역기 

📢 2024년 1학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다  

## 소개

저희 프로젝트는 청각 장애인의 의사소통 접근성을 향상시키는 것을 목표로 ViViT를 사용한 한국 수화 번역에 대한 새로운 접근 방식  
(결과는 좋지 않았지만..)을 제안합니다.   
한국어 수어 번역 비디오 데이터 세트를 활용하여 관절 좌표를 추출하고 ViViT을 실시간(약간?의 딜레이) 추론을 가능하게 합니다.

## 방법론
### 데이터셋
[AI Hub - 수어 영상](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=103) 데이터셋을 활용했습니다.   
(2.63TB의 거대한 용량으로 인해 동사 위주의 10개 단어를 선별하여 모델을 학습시켰습니다..)

### 모델
![Uploading 스크린샷 2025-01-19 20.48.06.png…]()

저희가 제안하는 모델의 작동 과정은 다음과 같습니다.
+ 입력 영상에서 Mediapipe를 통해 추출된 관절 좌표 정보를 활용하여 ViViT를 finetuning 합니다.
+ 학습된 ViViT를 통해 각 단어 영상의 대표 feature값을 추출하여 단어 사전을 만듭니다.  
+ 추론 단계에서 입력 받은 영상과 단어 사전의 feature를 비교하여 가장 유사한 단어를 출력합니다.

## 환경 설정

A100

## 사용 방법

```
python demomediapipe.py
```

## 예시 결과

<img width="517" alt="image" src="https://github.com/kkumtori/readme/assets/112691501/3acc7351-55ec-4498-9be2-3aa14c869b17">


## 팀원

- [성유진]: (데이터 수집 및 전처리, 데모)
- [김은진](https://github.com/eunbob): (데이터 수집 및 전처리, 모델링)
- [노지예](https://github.com/kkumtori): (데이터 수집 및 전처리, 모델링)
- [송현지](https://github.com/kelly062001): (데이터 수집 및 전처리, 데모)
- [이민하](https://github.com/mlnha): (데이터 수집 및 전처리, 모델링)
