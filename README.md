# Energy Forecasting 모델
## 프로젝트 소개
* 다종(전기,수도,가스,온수,열)의 에너지 사용량을 입력 받아 특정 에너지의 사용량을 예측하는 모델입니다.
* 해당 모델은 multi-modal learning 구조를 가지며 개별 에너지원은 하나의 모달리티를 의미합니다.
* 각각의 에너지 사용량은 feature extractor를 거쳐 예측하고자 하는 에너지와 관련된 feature vector를 추출합니다.
* 추출된 각 에너지의 feature vector는 concatenation 된 뒤 Fully-connected layer를 통과해 원하는 에너지의 사용량을 예측합니다.

## Setup
### Requirements
* python 3.7.12
* tensorflow 2.6.4
### Docker 환경 setup
* 본 프로젝트는 Ubuntu 18.04.6 LTS, docker 환경에서 진행되었습니다.
* 사용한 GPU는 RTX A4000, A5000 이며 아래 docker image를 사용하면 CUDA  및 다른 라이브러리 version 충돌 없이 사용 가능합니다.
* docker image 설치
```
sudo docker pull gcr.io/kaggle-gpu-images/python:v121
```
* baseline_transformer.py를 실행하면 모델을 실행할 수 있습니다.

## Models 
* 현재 feature extractor에 사용된 모델은 transformer 입니다.
* model.py 는 transformer 모델이 들어있는데 코드만 수정하면 다른 모델을 사용할 수 있습니다.
