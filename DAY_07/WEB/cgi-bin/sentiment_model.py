import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')  # 불용어 리스트 다운로드
nltk.download('punkt')      # 토크나이저 데이터 다운로드
nltk.download('wordnet')    # 표제어 추출(lemmatization)을 위한 WordNet 데이터 다운로드

# 모듈 로딩
# 모델관련 모듈
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from torchmetrics.classification import F1Score, BinaryF1Score
from torchmetrics.classification import BinaryConfusionMatrix
from torchinfo import summary

from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split

from flask import Flask, request, jsonify, render_template_string
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

FILE = 'training.1600000.processed.noemoticon.csv'
# ISO-8859-1 또는 latin1 인코딩으로 파일 읽기
df = pd.read_csv(FILE, encoding='ISO-8859-1', header=None)
df.columns = ['sentiment', 'id', 'date', 'flag', 'user', 'text']

df = df[['sentiment', 'text']]

import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# 전처리 함수 정의
def preprocess_text(text):
    # 1. 소문자 변환
    text = text.lower()
    
    # 2. URL 제거
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. @, # 기호 제거
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # 4. 불필요한 특수문자 제거 (단어 외의 모든 것 제거)
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # 알파벳과 공백만 남기기
    
    # 5. 불용어 제거
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # 결과 반환 (전처리된 단어 리스트를 공백으로 연결된 문자열로 반환)
    return ' '.join(words)

# 기존 text 열에 전처리 적용하여 덮어쓰기
df['text'] = df['text'].apply(preprocess_text)

# 'processed_text' 열 제거 (만약 존재한다면)
if 'processed_text' in df.columns:
    df.drop(columns=['processed_text'], inplace=True)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler

# 데이터셋의 50%만 샘플링
df_sampled = df.sample(frac=0.1, random_state=42)

# TF-IDF 벡터화 - 희소 행렬 사용 (max_features=1000으로 줄임)
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = tfidf_vectorizer.fit_transform(df_sampled['text']).toarray()

# 예시 데이터 프레임 (df_sampled를 이전 코드에서 사용한 데이터 프레임으로 대체)
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# 텍스트 데이터를 기반으로 TF-IDF 벡터화기 학습
tfidf_vectorizer.fit(df_sampled['text'])

# 벡터화기 저장
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# 샘플링된 데이터프레임에서 레이블 생성 (0=부정, 4=긍정)
y = df_sampled['sentiment'].apply(lambda x: 1 if x == 4 else 0).values

# 훈련 데이터, 검증 데이터, 테스트 데이터 분할 (80% 학습, 10% 검증, 10% 테스트)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 데이터 크기 확인
print(f'Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}')

# 학습에 필요한 하이퍼파라미터 설정
EPOCHS = 30
BATCH_SIZE = 16
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 텐서로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(DEVICE)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

# 데이터셋 및 데이터 로더
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

import torch
import torch.nn as nn

# Flask 앱 생성
app = Flask(__name__)
class SentimentClassifier(nn.Module):
    def __init__(self):
        super(SentimentClassifier, self).__init__()
        self.fc1 = nn.Linear(1000, 512)   # 은닉층 1
        self.bn1 = nn.BatchNorm1d(512)    # Batch Normalization 1
        self.fc2 = nn.Linear(512, 256)    # 은닉층 2
        self.bn2 = nn.BatchNorm1d(256)    # Batch Normalization 2
        self.fc3 = nn.Linear(256, 128)    # 은닉층 3
        self.bn3 = nn.BatchNorm1d(128)    # Batch Normalization 3
        self.fc4 = nn.Linear(128, 64)     # 은닉층 4
        self.bn4 = nn.BatchNorm1d(64)     # Batch Normalization 4
        self.fc5 = nn.Linear(64, 1)       # 출력층: 1개의 출력 (이진 분류)
        
        self.relu = nn.ReLU()             # 활성화 함수 (ReLU)
        self.sigmoid = nn.Sigmoid()       # 출력층에서 사용하는 Sigmoid (이진 분류)
        self.dropout = nn.Dropout(p=0.3)  # Dropout (30%)

    def forward(self, x):
        # 입력층 -> 은닉층 1
        x = self.relu(self.fc1(x))
        x = self.bn1(x)                   # Batch Normalization 적용
        x = self.dropout(x)               # Dropout 적용
        
        # 은닉층 1 -> 은닉층 2
        x = self.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        
        # 은닉층 2 -> 은닉층 3
        x = self.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout(x)
        
        # 은닉층 3 -> 은닉층 4
        x = self.relu(self.fc4(x))
        x = self.bn4(x)
        x = self.dropout(x)
        
        # 은닉층 4 -> 출력층
        x = self.sigmoid(self.fc5(x))     # Sigmoid 활성화 (이진 분류)
        return x
# 모델, 손실 함수, 옵티마이저 설정
model = SentimentClassifier().to(DEVICE)
criterion = nn.BCELoss()  # 이진 분류용 손실 함수
optimizer = optim.Adam(model.parameters(), lr=LR)

# 학습률 감소 스케줄러
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Early Stopping 클래스 정의
class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 학습 및 검증 과정
early_stopping = EarlyStopping(patience=100)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
        
        # 검증
        val_loss = validate_model(model, val_loader)
        
        # Early Stopping 체크
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        # 학습률 스케줄러 업데이트
        scheduler.step(val_loss)

def validate_model(model, val_loader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    print(f"Validation Loss: {val_loss/len(val_loader)}")
    return val_loss

# 모델 학습
train_model(model, train_loader, val_loader, criterion, optimizer)

# 모델 평가 과정
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()  # 0.5 이상의 값은 긍정으로 분류
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 모델 평가
evaluate_model(model, test_loader)

# 학습된 모델 저장
torch.save(model.state_dict(), 'model_weights.pth')

# 모델 로드
model = SentimentClassifier()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # 평가 모드

import torch
from sklearn.feature_extraction.text import TfidfVectorizer

# 학습된 모델 불러오기
model = SentimentClassifier()  # 동일한 모델 구조
model.load_state_dict(torch.load('model_weights.pth'))  # 저장된 가중치 불러오기
model.eval()  # 평가 모드로 전환

# TF-IDF 벡터화 도구 불러오기
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_vectorizer.fit(df_sampled['text'])  # 기존 데이터로 벡터라이저 학습

def predict_sentiment(model, text, vectorizer, threshold=0.69):
    # 입력된 텍스트를 벡터화 (TF-IDF 또는 다른 벡터화 방식 사용)
    input_vector = vectorizer.transform([text]).toarray()  # 문장을 벡터화
    input_tensor = torch.tensor(input_vector, dtype=torch.float32)  # 텐서로 변환

    # 모델로 예측 수행
    with torch.no_grad():  # 기울기 계산 불필요
        output = model(input_tensor)
        prediction = torch.sigmoid(output)  # 이진 분류이므로 Sigmoid 사용
        predicted_label = (prediction >= threshold).item()  # 0.5 이상이면 긍정

    # 예측 결과 반환 (Positive 또는 Negative)
    return 'Positive' if predicted_label == 1 else 'Negative'



