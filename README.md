# HDAI 2021 주제2: 심전도 데이터셋을 활용한 부정맥 진단 AI 모델

(메모)

- 대회 종료 시 결과 요약지(양식 1 : [팀명]H.D.A.I 2021 결과 요약지.pdf)와 개발된 AI 모델을 제출합니다.
- 결과물 제출: 주제 2. 2021년 12월 2일 (목) ~ 2021년 12월 12일 (일) 23:59
- 제출처 E-mail: hdaidatathon@gmail.com

A. AI 모델 제출 : 학습코드, 모델 Weight, 환경, 모델설명(1GB 미만)<br>
** *성능평가 검증을 위한 실행 가이드를 readme.md에 작성해주시기 바랍니다.*<br>
B. 결과 요약지 : 첨부파일 양식 1. 결과 요약지(H.D.A.I 2021).doc 활용  
** *단, 결과요약지 내에 주제 1은 DSC/JI , 주제 2는 AUC 출력값이 보이도록 스크린샷 첨부* <br>

**[TO-DO] source_arranged는 나중에 지울 것, ~~train.py~~랑 ~~전처리 코드 수정~~, 검증 코드 추가**

# 실행 가이드

## 0. Requirements

모델 weights는 따로 구글 드라이브 링크

<!--## 실행 방법-->

## 1. 데이터 전처리

전처리를 어떤 식으로 했는지 설명을 여기다가도 추가하는 게 좋을까?을
ex. label의 경우 주어진 데이터가 정상, 비정상으로 분류되어 제공되었으므로 각 데이터 개수에 맞춰 정상=0, 비정상=1로 인코딩한 리스트를 만들고 합쳐서 배열 생성했습니다.

```python
from utils.data_preprocess import DataPreprocess

DataPreprocess(path_arr='./data/test/arrhythmia', path_nor='./data/test/normal/',
               data_filename='./test_data', label_filename='./test_label')
```

## 2. 검증 방법

<!--## 간단한 사용 방법-->

<!--### 코드 라이센스-->
