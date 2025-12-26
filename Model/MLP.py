import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from Dataset import Dataset
from time import time
import math
import heapq
import os
from tqdm import tqdm


def get_model(num_users, num_items, layers=[20,10], reg_layers=[0,0]):
    num_layer = len(layers)
    
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # MLP 임베딩 (첫 번째 레이어 사이즈의 절반씩 할당)
    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0]/2), name='user_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0]/2), name='item_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]), input_length=1)
    
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    # 변경점: Concatenation (Merge mode='concat' -> Concatenate)
    vector = Concatenate()([user_latent, item_latent])
    
    # MLP 레이어 쌓기
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' %idx)
        vector = layer(vector)
        
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(vector)
    
    model = Model(inputs=[user_input, item_input], outputs=prediction)
    return model


def get_train_instances(train, num_negatives, num_items):
    # GMF와 동일한 로직이므로 생략 가능하나 독립 실행을 위해 포함
    user_input, item_input, labels = [],[],[]
    for (u, i) in train.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels




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





if __name__ == '__main__':
    # 설정값
    path = 'Data/'
    dataset_name = 'study'
    epochs = 20
    batch_size = 256
    num_factors = 8
    layers = [64,32,16,8]
    reg_layers = [0,0,0,0]
    num_negatives = 4
    learning_rate = 0.001
    topK = 10
    
    # 데이터 로드
    t1 = time()
    dataset = Dataset(path + dataset_name)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]." %(time()-t1))
    
    # 모델 생성
    model = get_model(num_users, num_items, layers, reg_layers)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')

    best_hr, best_ndcg = 0, 0
    
    # 학습
    for epoch in range(epochs):
        t1 = time()
        user_input, item_input, labels = get_train_instances(train, num_negatives, num_items)
        
        hist = model.fit([np.array(user_input), np.array(item_input)], np.array(labels), 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        
        if epoch % 1 == 0:
            hr, ndcg = evaluate_model(model, testRatings, testNegatives, topK)
            print(f'Iteration {epoch}: Loss = {hist.history["loss"][0]:.4f}, HR = {hr:.4f}, NDCG = {ndcg:.4f} [{t2-t1:.1f} s]')
            
            # 베스트 모델 저장 로직
            if hr > best_hr:
                best_hr = hr
                if not os.path.exists('Pretrain'): os.makedirs('Pretrain')
                model.save_weights(f'Pretrain/{dataset_name}_MLP.weights.h5', overwrite=True)

    print(f"End. Best HR = {best_hr:.4f}")
