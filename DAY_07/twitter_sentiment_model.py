#!/usr/bin/env python3

import cgi
import cgitb
import pandas as pd
import torch
import torch.nn.functional as F
from nltk.corpus import stopwords
import joblib
import sys
import os
MODULE_PATH = '/Users/anhyojun/VSCode/K-Digital Training/MyModule'
sys.path.append(MODULE_PATH)
from KDTModule import *

# 디버깅을 위해 CGI 오류를 보여줌
cgitb.enable()

# Content-Type 헤더 출력 (HTML 형식)
print("Content-Type: text/html; charset=UTF-8\n")
# print(f"<p>현재 작업 디렉토리: {os.getcwd()}</p>")

# CGI를 사용해 폼 데이터를 처리
form = cgi.FieldStorage()

# 입력값 가져오기
sentence = form.getfirst("string", "")

try:
    # LSTM 모델 작동
    sentenceDF = pd.DataFrame([sentence], columns=['clean_text'])
    loaded_vectorizer = joblib.load('/Users/anhyojun/VSCode/project/cgi-bin/tfid_vectorizer.pkl')
    input_vector = loaded_vectorizer.transform(sentenceDF['clean_text']).toarray()
    input_vectorDF = pd.DataFrame(input_vector)
    best_model = LSTMModel(input_size = 8000, output_size = 3, hidden_list = [100, 80, 60, 40, 20],
                    act_func=F.relu, model_type='multiclass', num_layers=1)
    best_model.load_state_dict(torch.load('/Users/anhyojun/VSCode/project/cgi-bin/Best_LSTM_Model.pth', weights_only=True))
    result = predict_value(input_vectorDF, best_model, dim=3)

    if result.item() == 0:
        answer = '이 문장은 중립적인 문장입니다.'
    elif result.item() == 1:
        answer = '이 문장은 긍정적인 문장입니다.'
    elif result.item() == 2:
        answer = '이 문장은 부정적인 문장입니다.'
except:
    answer = '올바른 문장을 입력하세요.'

# 결과 출력
print(f"""
<html>
<head><title>감정 분석 결과</title>
      <style>
        body {{
            text-align: center;
        }}
      </style>
</head>
<body>
    <h1>분석 결과</h1>
    <p>결과: {answer}</p>
    <a href="/">다시 분석하기</a>
</body>
</html>
""")
