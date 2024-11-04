import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import F1Score
from torchinfo import summary
from torchmetrics.regression import *
from torchmetrics.classification import *
from torchmetrics.functional.regression import r2_score
from torchmetrics.functional.classification import f1_score

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------
# 클래스 목적 : 학습용 데이터셋 텐서화 및 전처리
# 클래스 이름 : CustomDataSet
# 부모 클래스 : torch.utils.data.Dataset
# 매개   변수 : featureDF, targetDF
# ----------------------------------------------------

class CustomDataset(Dataset):
    # 데이터 로딩 및 전처리 진행과 인스턴스 생성 메서드
    def __init__(self, featureDF, targetDF):
        super().__init__()
        self.featureDF = featureDF
        self.targetDF = targetDF
        self.n_rows = featureDF.shape[0]
        self.n_features = featureDF.shape[1]

    # 데이터의 개수 반환 메서드
    def __len__(self):
        return self.n_rows

    # 특정 index의 데이터와 타겟 반환 메서드 => Tensor 반환!!!
    def __getitem__(self, idx): # 클래스 인스턴스 생성하면 자동으로 호출되는 콜백 메서드
        featureTS = torch.FloatTensor(self.featureDF.iloc[idx].values)
        targetTS = torch.FloatTensor(self.targetDF.iloc[idx].values)
        return featureTS, targetTS

# -----------------------------------------------------------------
# 사용자 정의 모델 클래스
# -----------------------------------------------------------------
# 부모 클래스 : nn.Module
# 필수 오버라이딩 : 
#     => __init__()  : 모델 층 구성 (설계)
#     => forward()   : 순방향 학습 진행 코드 구현
# -----------------------------------------------------------------

# cpu로 할지 gpu로 할지
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 입력 피쳐 수, 은닉층 퍼셉트론 수, 은닉층 개수가 모두 동적인 모델
class DeepModel(nn.Module):
    def __init__(self, input_in, output_out, hidden_list,
                 act_func, model_type):
        super().__init__() # 부모 클래스 생성

        # 입력층
        self.input_layer = nn.Linear(input_in, hidden_list[0])
        # 은닉층
        self.hidden_layer_list = nn.ModuleList()
        for i in range(len(hidden_list)-1):
            self.hidden_layer_list.append(nn.Linear(hidden_list[i], hidden_list[i+1]))
        # 출력층
        self.output_layer = nn.Linear(hidden_list[-1], output_out)

        self.act_func = act_func
        self.model_type = model_type

    # 순방향/전방향 학습 진행 시 자동 호출되는 메서드 (콜백 함수: 시스템에서 호출되는 함수)
    def forward(self, x):
        # 입력층 학습
        x = self.input_layer(x)
        x = self.act_func(x)

        # 은닉층 학습
        for i in self.hidden_layer_list:
            x = i(x)
            x = self.act_func(x)

        if self.model_type == 'regression': # 회귀
            return self.output_layer(x) # 활성화 함수 안거치고 출력
        elif self.model_type == 'binary': # 이진 분류
            return torch.sigmoid(self.output_layer(x)) # 시그모이드
        elif self.model_type == 'multiclass': # 다중 분류
            return self.output_layer(x) # 소프트맥스
            # 다중 분류의 경우 CrossEntropyLoss에서 내부적으로 log-softmax를 처리하기 때문에
            # 모델의 마지막 출력에서 소프트맥스를 적용하지 않고 바로 전달하면 됨

# -----------------------------------------------------------------
## 테스트/검증 함수 
# ==> 가중치, 절편 업데이트 X, 최적화 미진행
# ==> 현재 가중치와 절편값으로 테스트 진행
# -----------------------------------------------------------------
def testing(test_DataLoader, model, model_type, num_classes=None):
    model.eval() # 검증 모드임을 명시적으로 선언 (검증용 통계치 사용 및 드롭아웃 비활성화)
    total_loss_test = 0
    total_score_test = 0

    with torch.no_grad(): # 가중치 업데이트 없이 테스트 진행
        for X_batch, y_batch in test_DataLoader:
            # (1) 순전파 (평가)
            pred_test_y = model(X_batch)
            # (2) 손실 함수 계산

            if model_type == 'regression': # 회귀일 때
                loss_test = F.mse_loss(pred_test_y, y_batch)
                score_test = r2_score(pred_test_y, y_batch)
            elif model_type == 'binary': # 이진 분류일 때
                loss_test = F.binary_cross_entropy(pred_test_y, y_batch)
                score_test = f1_score(pred_test_y, y_batch, task='binary')
            elif model_type == 'multiclass': # 다중 분류일 때
                y_batch1D = y_batch.reshape(-1) # 다중 분류는 y가 반드시 1차원이어야 함.. (너무 불친절)
                loss_test = F.cross_entropy(pred_test_y, y_batch1D.long())
                pred_test_labels = torch.argmax(pred_test_y, dim=1)
                # dim=1을 해야 한 행 내에서 가장 큰 원소의 인덱스를 가져옴
                score_test = f1_score(pred_test_labels, y_batch1D,
                                      task='multiclass', num_classes=num_classes)
                # 다중 분류는 long타입으로 전달해야 하는듯

            total_loss_test += loss_test.item()
            total_score_test += score_test.item()

    loss_test_avg = total_loss_test / len(test_DataLoader)
    score_test_avg = total_score_test / len(test_DataLoader)
    return loss_test_avg, score_test_avg

# -----------------------------------------------------------------
# 모델 학습 함수
# -----------------------------------------------------------------
def training(train_DataLoader, test_DataLoader, model, model_type, optimizer,
             epoch = 1000, endurance_cnt = 10, view_epoch=1, num_classes=None, SAVE_PATH=None):
    model.train() # 학습 모드임을 명시적으로 선언 (학습용 통계치 사용)
    loss_train_history = []
    loss_test_history = []
    score_train_history = []
    score_test_history = []

    EARLY_STOP_LOSS_CNT = 0

    for i in range(1, epoch+1): # 에포크 횟수만큼 반복
        total_loss_train = 0 # 한 에포크당 합산할 손실값 (나중에 평균 계산)
        total_score_train = 0 # 한 에포크당 합산할 손실값 (나중에 평균 계산)

        for X_batch, y_batch in train_DataLoader:
            # (1) 순전파 (학습)
            pred_train_y = model(X_batch)
            # (2) 손실 함수 계산
            if model_type == 'regression': # 회귀일 때
                loss_train = F.mse_loss(pred_train_y, y_batch)
                score_train = r2_score(pred_train_y, y_batch)
            elif model_type == 'binary': # 이진 분류일 때
                loss_train = F.binary_cross_entropy(pred_train_y, y_batch)
                score_train = f1_score(pred_train_y, y_batch, task='binary')
            elif model_type == 'multiclass': # 다중 분류일 때
                y_batch1D = y_batch.reshape(-1) # 다중 분류는 y가 반드시 1차원이어야 함.. (너무 불친절)
                loss_train = F.cross_entropy(pred_train_y, y_batch1D.long())
                pred_train_labels = torch.argmax(pred_train_y, dim=1)
                # dim=1을 해야 한 행 내에서 가장 큰 원소의 인덱스를 가져옴
                score_train = f1_score(pred_train_labels, y_batch1D,
                                       task='multiclass', num_classes=num_classes)
                # 다중 분류는 long타입으로 전달해야 하는듯

            # (3) 최적화
            optimizer.zero_grad() # 그레디언트 초기화
            loss_train.backward() # 역전파 하면서 그레디언트 계산
            optimizer.step() # 가중치, 절편 업데이트
            # (4) 손실값 합산
            total_loss_train += loss_train.item()
            total_score_train += score_train.item()

        loss_train_avg = total_loss_train / len(train_DataLoader)
        score_train_avg = total_score_train / len(train_DataLoader)
        
        # 한 에포크마다 테스트 실행
        if model_type == 'regression':
            loss_test_avg, score_test_avg = testing(test_DataLoader, model, model_type='regression')
        elif model_type == 'binary':
            loss_test_avg, score_test_avg = testing(test_DataLoader, model, model_type='binary')
        elif model_type == 'multiclass':
            loss_test_avg, score_test_avg = testing(test_DataLoader, model,
                                    model_type='multiclass', num_classes=num_classes)

        loss_train_history.append(loss_train_avg)
        loss_test_history.append(loss_test_avg)
        score_train_history.append(score_train_avg)
        score_test_history.append(score_test_avg)

        if len(loss_test_history) == 1: # 첫 에포크일때
            best_loss = loss_test_avg
            torch.save(model.state_dict(), f'{SAVE_PATH}/best_model_epoch_{i}.pth')
            print(f"[EPOCH] : {i}에서 모델 저장 완료.")

        else:
            if best_loss > loss_test_avg: # 손실값이 개선 됐다면
                best_loss = loss_test_avg
                EARLY_STOP_LOSS_CNT = 0 # 개선되면 카운트 초기화
                torch.save(model.state_dict(), f'{SAVE_PATH}/best_model_epoch_{i}.pth')
                print(f"[EPOCH] : {i}에서 모델 저장 완료.")
            else:                         # 손실값이 개선되지 않았다면
                EARLY_STOP_LOSS_CNT += 1

        if EARLY_STOP_LOSS_CNT == endurance_cnt:
            print(f'[Loss]값의 개선이 이루어지지 않아 [{i}] EPOCH에서 학습을 종료합니다.')
            break

        # (4) 학습 결과 출력
        if i % view_epoch == 0:
            print(f"[Loss : {i}/{epoch}] Train : {loss_train_avg:.4f}, Test : {loss_test_avg:.4f}")
            print(f"[Score  : {i}/{epoch}] Train : {score_train_avg:.4f}, Test : {score_test_avg:.4f}")
    
    return loss_train_history, loss_test_history, score_train_history, score_test_history

# -----------------------------------------------------------------
# 손실값과 스코어 그려주는 함수
# -----------------------------------------------------------------

def DrawPlot(result):
    fig, axs = plt.subplots(1, 2, figsize = (14, 5))

    label_list = ['Loss', 'Score']

    LENGTH = len(result[0])

    for i in range(2):
        axs[i].plot(range(1, LENGTH+1), result[2*i], label = f'Train {label_list[i]}')
        axs[i].plot(range(1, LENGTH+1), result[2*i+1], label = f'Valid {label_list[i]}')
        axs[i].set_title(label_list[i])
        axs[i].set_xlabel('EPOCH')
        axs[i].set_ylabel('Loss')
        axs[i].legend()
    plt.show()

# -----------------------------------------------------------------
# 예측값 출력하는 함수
# -----------------------------------------------------------------

def predict_value(test_inputDF, model):
    test_inputTS = torch.FloatTensor(test_inputDF.values)
    return torch.argmax(model(test_inputTS), dim=1)