##최대 분산투자 포트폴리오(Maximum Diversification Portfolio, MDP)
--volatility of individual assets(weighted)/portfolio volatility(weighted)
--분산이 최대이면 개별 자산의 리스크는 최대가 되지만, 포트폴리오로 묶으면 변동성이 줄어든다-> 골고루 잘 배분됐다!
--개별 자산의 리스크 대비 포트폴리오 전체의 리스크는 최소가 되는 가중치를 찾기-> 최대한 골고루 잘 섞은 포트폴리오를 찾기!
--미래 수익률이 불확실할 때(기대수익률을 추정하기 어려울때) 리스크 중심 접근이 합리적
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize
sns.set()

# ETF 데이터 다운로드

tickers = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
data=yf.download(tickers, start='2010-01-01')['Close']
data=data.resample('ME').last()
print(data.head())

#data = etf.history(start='2010-01-01', actions=False)     #actions=False: 배당, 분할정보 제외
#data.drop(['Open', 'High', 'Low', 'Volume'], inplace=True, axis=1)   #inplace=True: column을 원본 data에서 직접 삭제
#data = data.droplevel(0, axis=1).resample('ME').last()               #axis=0: 행 방향(제거), axis=1: 열 방향(제거)

# 수익률 계산
rets = data.pct_change().dropna()
'''
# 색깔 팔레트
pal = sns.color_palette('Spectral', len(tickers))     #spectral: seaborn에서 제공하는 컬러 팔레트
'''
# 공분산행렬
cov = np.array(rets.cov() * 12)
print(cov)

# 각 자산별 변동성
vol = np.diag(cov)         #각 자산의 분산 구하기
print(vol)

# 초기값 설정
noa = rets.shape[1]
init_guess = np.repeat(1/noa, noa)

# 상하한값
bounds= ((0.0, 1.0), ) * noa

# 제약조건
weights_sum_to_1 = {'type': 'eq',
                    'fun': lambda weights: np.sum(weights) - 1}

# 목적함수 : 마이너스 분산비율
def neg_div_ratio(weights, vol, cov):
    weighted_vol = weights.T @ vol
    port_vol = np.sqrt(weights.T @ cov @ weights)
    return - weighted_vol / port_vol

# 가중치 계산
res = minimize(neg_div_ratio,
               init_guess,
               args=(vol, cov),
               method='SLSQP',
               constraints=(weights_sum_to_1,),
               bounds=bounds)

#가중치 벡터 표현
weights = res.x
print(res.x)

#가중치 표로 정리하기
weight=np.array([1.49619900e-17, 5.10056012e-01, 3.96817995e-17, 5.52943108e-18,
 1.26250775e-01, 1.58293517e-17, 3.20960337e-01, 1.66967135e-17,
 4.27328765e-02])

opt_weights=pd.DataFrame({"Ticker":tickers, 'Weight':weight})
opt_weights["Weight(%)"]=opt_weights['Weight']*100
opt_weights["Weight(%)"]=opt_weights["Weight(%)"].round(2)

print(opt_weights)

'''
RESULTS:
   Ticker   Weight        Weight(%)
0    XLB  1.496199e-17       0.00
1    XLE  5.100560e-01      51.01
2    XLF  3.968180e-17       0.00
3    XLI  5.529431e-18       0.00
4    XLK  1.262508e-01      12.63
5    XLP  1.582935e-17       0.00
6    XLU  3.209603e-01      32.10
7    XLV  1.669671e-17       0.00
8    XLY  4.273288e-02       4.27
'''

