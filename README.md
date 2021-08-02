# OCT 이미지를 이용한 황반변성 분류 프로젝트


![jaemin-don-7ZlXsihxD2c-unsplash](https://user-images.githubusercontent.com/80465347/119299360-bb1a1900-bc99-11eb-908b-789de1220f89.jpg)


제주도를 찾는 관광객들이 선택하는 주요 교통수단은 단연 렌트카이지만, (특히 코로나 19의 여파로) 국내 여행지의 대표인 제주도가 점점 더 각광받으면서 대중교통의 이용 또한 증가하고 있습니다. 
이에 대중교통을 이용하는 관광객의 수를 예측하여 관광지 주변 상권이나 Public Mobile 업체들이 활용할 수 있는 데이터를 만들고자 하였습니다. 

## 1. Project Summary 
### 1.1 목적 
황반변성 질환(CNV, DME, Drusen) 분류하기 

### 1.2 결과
(참고. Kaggle에 공개되어있는 모델 중 가장 성능이 좋은 모델이 validation accuracy 0.99를 기록하였습니다.)

### 1.3 데이터

- Retinal OCT Image Dataset / [Kaggle](https://www.kaggle.com/paultimothymooney/kermany2018) - University of California San Diego

### 1.4 진행순서 

- 데이터셋 탐색
- 샘플링
- 이미지 처리 기법 적용  
- CNN 모델 적용  

### 1.5 주요 이미지 처리 기법 및 CNN 모델 

- 이미지 처리 
  - CLAHE / HE
  - Difference of Gaussians
  - HSV mask
  - Image Contour
- CNN 모델
  - LeNet
  - VGG16, VGG19
  - MobileNet
  - ResNet50
  - GoogleNet, InceptionV3
  - InceptionResNetV2 

## 2. File List 

-  JejuRegion.py : 권역 그룹핑을 위한 모듈 
-  jeju_datahub_api : 제주데이터허브 API 이용하기 
-  jeju_visualization_map : pydeck을 이용한 제주도 지역별 버스 이용량 시각화  
-  df_regression.csv : regression 용 데이터프레임 파일
-  jeju_clustering.ipynb : 버스 이용객을 대상으로 한 kmeans clustering
-  jeju_regression_part2.ipynb : 회귀분석 


## 3. Contributors

* [정현](https://github.com/JeonghyunKo) - 데이터 샘플링 / 이미지 처리 
* [동주](https://github.com/lee-edgar) - CNN 모델 적용 

## 4. What We Learned, and more... 

- DASK를 사용하면서 필요한 데이터만 가져와서 처리하거나, 최대한 간결하게 작동할 수 있는 코드를 작성하는 등 작업 효율에 대한 고민을 깊게 할 수 있었습니다. 다만 DASK에 적응하는 시간이 소요되면서 프로젝트 착수가 약간 늦어지고, 그만큼 모델 성능 향상을 위한 개선 작업을 단축시켜야 했던 점은 아쉽습니다. 지속적으로 후속 작업들을 이어갈 예정입니다. 
- 추후 제주 지역에서 일레클, ZET와 같은 공유 모빌리티 / 라스트마일 서비스 업체들의 적절한 정류장 위치 선정을 위한 이용자 수 예측이나, 가동률 예측 등 대중교통 이용 관광객 수를 이용한 심화된 프로젝트를 진행해보고 싶습니다.  

## 5. Acknowledgments 

* [Flycode77](https://github.com/FLY-CODE77)  
* [PinkWink](https://github.com/PinkWink) 

## 프레젠테이션 

* [최종 발표자료](https://drive.google.com/file/d/1yByxhh9NdsVLmhzN3Y-tGkHe22ZX3taA/view?usp=sharing)
* https://unsplash.com/photos/3nQ4pIOW2g4?utm_source=unsplash&utm_medium=referral&utm_content=creditShareLink
