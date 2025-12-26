import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import math
import heapq
from time import time
import os
from tqdm import tqdm

# ==========================================
# 1. 모델 정의 (GMF - Generalized Matrix Factorization)
# ==========================================
def get_model(num_users, num_items, latent_dim, regs=[0,0]):
    """
    사용자와 아이템의 관계를 선형적으로 결합한 GMF 모델을 생성합니다.
    """
    # 입력 레이어: 사용자 ID와 아이템 ID를 1개씩 받음
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # 임베딩 레이어: ID를 밀집 벡터(latent_dim 크기)로 변환
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
                                  embeddings_initializer='random_normal', embeddings_regularizer=l2(regs[0]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
                                  embeddings_initializer='random_normal', embeddings_regularizer=l2(regs[1]), input_length=1)
    
    # 2D 출력을 1D 벡터로 변환
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))
    
    # [GMF 핵심] 두 벡터를 원소별로 곱함 (Element-wise Product)
    predict_vector = Multiply()([user_latent, item_latent])
    
    # 출력층: 최종 선호도를 0~1 사이 값으로 예측
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(predict_vector)
    
    model = Model(inputs=[user_input, item_input], outputs=prediction)
    return model

# ==========================================
# 2. 학습 데이터 생성 (Negative Sampling)
# ==========================================
def get_train_instances(train, num_negatives, num_items):
    """
    정답 데이터(1)와 학습용 오답 데이터(0)를 섞어서 반환합니다.
    """
    user_input, item_input, labels = [], [], []
    for (u, i) in train.keys():
        # Positive Sample (실제 상호작용이 있는 아이템)
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        
        # Negative Samples (사용자가 소비하지 않은 아이템을 무작위 추출)
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train: # 진짜 정답이면 다시 뽑음
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return np.array(user_input), np.array(item_input), np.array(labels)

# ==========================================
# 3. 모델 평가 (Hit Ratio & NDCG)
# ==========================================
import numpy as np
import heapq
import math

def evaluate_model(model, testRatings, testNegatives, K):
    """
    벡터화 연산을 적용하여 속도를 최적화한 평가 함수
    """
    hits, ndcgs = [], []
    
    # 1. 모든 사용자와 아이템 데이터를 하나의 리스트로 통합
    all_users = []
    all_items = []
    
    # 각 사용자별로 정답 아이템 위치를 추적하기 위한 정보
    items_per_user = len(testNegatives[0]) + 1 
    
    for idx in range(len(testRatings)):
        u = testRatings[idx][0]
        gtItem = testRatings[idx][1]
        items = testNegatives[idx] + [gtItem]
        
        all_users.extend([u] * items_per_user)
        all_items.extend(items)
    
    # 2. 모델 예측을 단 한 번(또는 큰 배치로) 수행
    # 루프 밖에서 호출하므로 커널 런칭 오버헤드가 극적으로 줄어듭니다.
    predictions = model.predict([np.array(all_users), np.array(all_items)], 
                                batch_size=2048, verbose=1)
    
    # 3. 결과 해석 (이 부분은 CPU 연산)
    for idx in range(len(testRatings)):
        # 해당 사용자의 예측값 슬라이싱
        start = idx * items_per_user
        end = start + items_per_user
        user_predictions = predictions[start:end].flatten()
        
        # 아이템 리스트 재구성 (마지막이 정답 아이템)
        gtItem = testRatings[idx][1]
        items = testNegatives[idx] + [gtItem]
        
        # 점수 매핑 및 순위 계산
        map_item_score = {items[i]: user_predictions[i] for i in range(items_per_user)}
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        
        # 지표 계산
        if gtItem in ranklist:
            hits.append(1)
            rank = ranklist.index(gtItem)
            ndcgs.append(math.log(2) / math.log(rank + 2))
        else:
            hits.append(0)
            ndcgs.append(0)
            
    return np.mean(hits), np.mean(ndcgs)

# ==========================================
# 4. 메인 실행 프로세스
# ==========================================
if __name__ == '__main__':
    # 설정값 (Hyperparameters)
    dataset_name = 'study'
    epochs = 20
    batch_size = 256
    num_factors = 8
    num_negatives = 4
    learning_rate = 0.001
    topK = 10

    # 주의: Dataset 클래스는 별도의 Dataset.py 파일에 정의되어 있어야 합니다.
    from Dataset import Dataset 
    dataset = Dataset('Data/' + dataset_name)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    
    # 모델 생성 및 컴파일
    model = get_model(num_users, num_items, num_factors)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')

    best_hr, best_ndcg = 0, 0
    
    for epoch in range(epochs):
        t1 = time()
        # 학습 데이터 준비
        u_in, i_in, lbl = get_train_instances(train, num_negatives, num_items)
        
        # 모델 학습
        hist = model.fit([u_in, i_in], lbl, batch_size=batch_size, epochs=1, verbose=1, shuffle=True)
        t2 = time()
        # 성능 평가 (매 에포크 또는 주기적으로)
        if epoch % 1 == 0:
            hr, ndcg = evaluate_model(model, testRatings, testNegatives, topK)
            print(f'Iteration {epoch}: Loss = {hist.history["loss"][0]:.4f}, HR = {hr:.4f}, NDCG = {ndcg:.4f} [{t2-t1:.1f} s]')
            
            # 베스트 모델 저장 로직
            if hr > best_hr:
                best_hr = hr
                if not os.path.exists('Pretrain'): os.makedirs('Pretrain')
                model.save_weights(f'Pretrain/{dataset_name}_GMF.weights.h5', overwrite=True)

    print(f"End. Best HR = {best_hr:.4f}")