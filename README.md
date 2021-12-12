# HDAI 2021 주제2: 심전도 데이터셋을 활용한 부정맥 진단 AI 모델

# 실행 가이드

## 0. Requirements

Python Version: `3.7.11`<br>
PyTorch Version: `1.9.0+cu102`<br>

Required Libraries:

```
xmltodict
sklearn
matplotlib
pandas
```

## 1. 데이터 전처리

저희는 모든 입력 데이터를 (12, 4096) 형태로 바꾸고 이를 넘파이 배열 파일로 저장하도록 전처리 작업을 수행하였습니다.<br>
레이블의 경우, 주어진 데이터가 정상 또 비정상으로 분류되어 제공되었으므로 각 데이터 개수에 맞춰 정상=0, 비정상=1로 인코딩한 리스트를 만들고 합쳐서 넘파이 배열을 저장하도록 만들었습니다.

### DataPreprocess

> path_arr, path_nor, data_filename='./data', label_filename='./label', data_type='test'

- `path_arr`과 `path_nor`는 각각 비정상 데이터와 정상 데이터가 저장된 경로입니다.
- 전처리된 데이터는 넘파이 배열 파일인 `.npy` 파일로 저장되는데, `data_filename`과 `label_filename`은 저장하고 싶은 데이터 및 레이블의 파일명을 의미합니다.
- `data_type`은 'train', 'valid', 'test'로 세 가지 데이터의 종류에 맞춰 전처리 작업을 수행하도록 합니다.

```python
from utils.data_preprocess import DataPreprocess

DataPreprocess(path_arr='electrocardiogram/data/test/arrhythmia/',
               path_nor='electrocardiogram/data/test/normal/',
               data_filename='./test_data', label_filename='./test_label')
```

## 2. 검증 방법

`test_modules` 모듈 안에 `predict`, `print_scores`, `plot_roc_curve` 메서드를 정의했습니다.

```python
from test_modules import predict, print_scores, plot_roc_curve
```

### predict(model, data_path='./data.npy', label_path='./label.npy') -> y_target, y_predicted

정답과 예측 결과를 함께 반환하는 메서드입니다.<br>
이 메서드를 사용하기 위해서는 전처리 작업을 마친 npy 파일이 있어야 하고, 모델을 먼저 불러와야 합니다.

```python
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 모델 불러오기
from models.resnet import resnet34

model = resnet34()
model.load_state_dict(torch.load('./weights/epoch_100_model_trial_1', map_location=torch.device(device)))

# 전체 테스트 데이터셋에 대한 정답과 예측 결과를 반환합니다.
y_target, y_predicted = predict(model, data_path='./valid_data.npy', label_path='./valid_label.npy')
```

### print_scores(y_target, y_predicted, save_csv=True) -> None

score를 출력하는 메서드입니다. `save_csv=True`로 놓으면 score들이 csv 파일로 저장됩니다.

```python
print_scores(y_target, y_predicted)
```

예상 결과:

```
                              Scores
Area Under the Curve (AUC)  0.910898
Average Precision           0.829775
Accuracy Score              0.904507
Recall Score                0.988336
Precision Score             0.834154
F1 Score                    0.904723
```

### plot_roc_curve(y_target, y_predicted, guideline=False, save_png=True) -> None

ROC 커브를 그려주는 메서드입니다. `save_png=True`로 놓으면 ROC 커브의 이미지가 png 파일로 저장됩니다.

```python
plot_roc_curve(y_target, y_predicted, guideline=False)
```

![wo_guideline](ROC_Curves/roc_curve_test1.png)

```python
plot_roc_curve(y_target, y_predicted, guideline=True)
```

![w_guideline](ROC_Curves/roc_curve_test2.png)


