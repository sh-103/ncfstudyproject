import math
import heapq
import numpy as np
import tensorflow as tf

# 전역 변수: 여러 함수에서 공통으로 참조할 모델과 데이터를 담아두는 저장소입니다.
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    전체적인 평가 프로세스를 관리하는 메인 함수입니다.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    
    # 전달받은 인자들을 전역 변수에 할당하여 어디서든 쓸 수 있게 합니다.
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    
    hits, ndcgs = [], [] # 각 사용자별 결과(성공 여부, 순위 품질)를 담을 리스트입니다.
    
    # 테스트 데이터셋의 각 사용자(idx)를 하나씩 순회하며 평가를 진행합니다.
    for idx in range(len(_testRatings)):
        # 사용자 한 명에 대한 Hit Ratio와 NDCG 점수를 계산해옵니다.
        (hr, ndcg) = eval_one_rating(idx)
        hits.append(hr) # 성공 여부 기록
        ndcgs.append(ndcg) # 순위 품질 기록
        
    return (hits, ndcgs) # 모든 사용자의 평가 결과를 반환합니다.

def eval_one_rating(idx):
    """
    특정 사용자 한 명에 대해 정답 아이템이 Top-K 내에 있는지 계산합니다.
    """
    rating = _testRatings[idx]    # 실제 사용자가 소비한 (User, Item) 쌍을 가져옵니다.
    items = _testNegatives[idx]   # 해당 사용자가 소비하지 않은 무작위 아이템(오답) 리스트입니다.
    u = rating[0]                 # 현재 평가 대상인 사용자 ID입니다.
    gtItem = rating[1]            # 우리가 맞춰야 할 정답 아이템(Ground Truth) ID입니다.
    items.append(gtItem)          # 오답 리스트 끝에 정답을 슬쩍 끼워 넣습니다. (총 100개 등)
    
    map_item_score = {}           # 아이템 ID별 모델의 예측 점수를 저장할 딕셔너리입니다.
    # 모델 입력용: 아이템 개수만큼 사용자 ID를 복사한 배열을 만듭니다. (예: [u, u, u, ...])
    users = np.full(len(items), u, dtype='int32')
    
    # 모델에게 "이 사용자가 이 아이템들을 좋아할 확률이 뭐야?"라고 한꺼번에 물어봅니다.
    predictions = _model.predict([users, np.array(items)], batch_size=100, verbose=0)
    
    # 예측된 점수들을 아이템 ID와 매칭하여 정리합니다.
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i][0] # {아이템ID: 점수} 형태로 저장
    
    items.pop() # 다음 테스트를 위해 아까 넣었던 정답 아이템을 리스트에서 다시 뺍니다.
    
    # heapq를 사용해 점수가 가장 높은 상위 K개의 아이템 ID만 뽑아냅니다. (이게 추천 목록입니다.)
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    
    # 추천된 목록 안에 실제 정답(gtItem)이 있는지 확인하여 지표를 계산합니다.
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    """
    Hit Ratio: 추천 리스트(K개) 안에 정답이 있으면 1, 없으면 0을 반환합니다.
    """
    for item in ranklist:
        if item == gtItem:
            return 1 # 하나라도 걸리면 성공!
    return 0

def getNDCG(ranklist, gtItem):
    """
    NDCG: 정답이 리스트 상단에 있을수록 더 높은 가중치를 줍니다.
    """
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            # i=0(1등)이면 log(2)/log(2) = 1점, 순위가 낮아질수록 점수는 작아집니다.
            return math.log(2) / math.log(i+2)
    return 0