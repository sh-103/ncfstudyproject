'''
NeuMF_TF2.py
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Concatenate, Multiply
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from evaluate import evaluate_model  # <-- 삭제하지 않음(요구대로 유지). 아래에서 같은 이름 함수로 덮어씀.
from Dataset import Dataset
from time import time
import os
import argparse

# ✅ 추가(MLP 벡터화 evaluate에서 필요)
import math
import heapq

# ✅ 추가: H5 가중치 로딩 호환용(NeuMF.py만 수정, GMF/MLP 수정 0)
import h5py

# 중요: 같은 폴더에 있는 GMF.py와 MLP.py를 불러옵니다.
import GMF
import MLP


def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    num_layer = len(layers)

    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # MF Part
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                  embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                  embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_mf), input_length=1)

    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = Multiply()([mf_user_latent, mf_item_latent])

    # MLP Part
    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0]/2), name="mlp_embedding_user",
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0]/2), name='mlp_embedding_item',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]), input_length=1)

    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])

    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name="layer%d" % idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    predict_vector = Concatenate()([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

    model = Model(inputs=[user_input, item_input], outputs=prediction)
    return model


def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' % i).get_weights()
        model.get_layer('layer%d' % i).set_weights(mlp_layer_weights)

    # Prediction weights (Concatenate & Average)
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5 * new_weights, 0.5 * new_b])
    print("Pre-trained weights loaded successfully!")
    return model


def get_train_instances(train, num_negatives, num_items):
    user_input, item_input, labels = [], [], []
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


# =========================================================
# ✅ 추가: "안전 로딩" (NeuMF.py만 수정해서 호환성 해결)
# - 1) 일반 load_weights() 시도
# - 2) 실패하면 legacy H5 로더로 직접 로드
# =========================================================
def safe_load_h5_weights(model, weight_path, desc=""):
    # 1) 일반 로드 시도
    try:
        model.load_weights(weight_path)
        print(f"[Pretrain] {desc} loaded by plain load_weights() ✅")
        return True
    except Exception as e1:
        print(f"[Pretrain] {desc} plain load_weights failed: {e1}")

    # 2) legacy H5 로더로 재시도
    try:
        from tensorflow.keras.saving.legacy import hdf5_format
        with h5py.File(weight_path, "r") as f:
            # 대다수는 model_weights 아래에 저장됨
            if "model_weights" in f:
                hdf5_format.load_weights_from_hdf5_group(f["model_weights"], model)
            else:
                # 루트가 바로 weights인 경우 fallback
                hdf5_format.load_weights_from_hdf5_group(f, model)
        print(f"[Pretrain] {desc} loaded by legacy H5 loader ✅")
        return True
    except Exception as e2:
        print(f"[Pretrain] {desc} legacy H5 load failed: {e2}")
        return False


# =========================================================
# ✅ 추가: MLP/GMF 스타일 "벡터화 evaluate_model"
# - 기존 호출 (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, 10, 1)
#   을 그대로 유지하기 위해 반환을 리스트 2개로 맞춤
# - import 되어있는 evaluate_model은 삭제하지 않고, 여기서 같은 이름으로 덮어씀
# =========================================================
def evaluate_model(model, testRatings, testNegatives, K, num_thread=1):
    """
    벡터화 연산을 적용한 평가 함수(속도 최적화)
    - 모든 (user, item) 쌍을 한 번에 predict
    - HR@K / NDCG@K를 계산하여 hits, ndcgs 리스트로 반환
    """
    hits, ndcgs = [], []

    items_per_user = len(testNegatives[0]) + 1  # neg + gt

    all_users = []
    all_items = []

    for idx in range(len(testRatings)):
        u = testRatings[idx][0]
        gtItem = testRatings[idx][1]
        items = testNegatives[idx] + [gtItem]

        all_users.extend([u] * items_per_user)
        all_items.extend(items)

    predictions = model.predict(
        [np.array(all_users), np.array(all_items)],
        batch_size=2048,
        verbose=0
    ).reshape(-1)

    for idx in range(len(testRatings)):
        start = idx * items_per_user
        end = start + items_per_user
        user_pred = predictions[start:end]

        gtItem = testRatings[idx][1]
        items = testNegatives[idx] + [gtItem]

        map_item_score = {items[i]: float(user_pred[i]) for i in range(items_per_user)}
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)

        if gtItem in ranklist:
            hits.append(1)
            rank = ranklist.index(gtItem)
            ndcgs.append(math.log(2) / math.log(rank + 2))
        else:
            hits.append(0)
            ndcgs.append(0)

    return hits, ndcgs


if __name__ == '__main__':
    path = 'Data/'
    dataset_name = 'study'
    epochs = 20
    batch_size = 256
    mf_dim = 8
    layers = [64, 32, 16, 8]
    reg_mf = 0
    reg_layers = [0, 0, 0, 0]
    num_negatives = 4
    learning_rate = 0.001

    # ✅ 너가 말한대로: 사전학습 가중치는 Pretrain 폴더에 둠
    mf_pretrain = 'Pretrain/study_GMF.weights.h5'
    mlp_pretrain = 'Pretrain/study_MLP.weights.h5'

    t1 = time()
    dataset = Dataset(path + dataset_name)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)

    # 사전 학습 모델 로드 여부 체크
    if os.path.exists(mf_pretrain) and os.path.exists(mlp_pretrain):
        print(f"Loading pre-trained models from {mf_pretrain} and {mlp_pretrain}...")

        gmf_model = GMF.get_model(num_users, num_items, mf_dim)
        ok_gmf = safe_load_h5_weights(gmf_model, mf_pretrain, desc="GMF")

        mlp_model = MLP.get_model(num_users, num_items, layers, reg_layers)
        ok_mlp = safe_load_h5_weights(mlp_model, mlp_pretrain, desc="MLP")

        if ok_gmf and ok_mlp:
            model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
            optimizer = SGD(learning_rate=learning_rate)  # Fine-tuning은 SGD 권장
        else:
            print("[Pretrain] 일부 로딩 실패 -> Training from scratch.")
            optimizer = Adam(learning_rate=learning_rate)
    else:
        print("No pre-trained weights found. Training from scratch.")
        optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='binary_crossentropy')

    best_hr, best_ndcg, best_iter = 0, 0, -1

    for epoch in range(epochs):
        t1 = time()
        user_input, item_input, labels = get_train_instances(train, num_negatives, num_items)

        hist = model.fit([np.array(user_input), np.array(item_input)],
                         np.array(labels),
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        if epoch % 1 == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, 10, 1)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
