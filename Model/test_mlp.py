# from tqdm import tqdm
# import time

# pbar = tqdm(range(100))
# for i in pbar:
#     time.sleep(0.1)
    
#     # 앞쪽 설명 문구를 동적으로 변경
#     pbar.set_description(f"Batch {i}")
    
#     # 뒤쪽 정보 칸에 변수 추가
#     pbar.set_postfix(loss=1.5/(i+1), status="Processing")
import numpy as np
import heapq

def get_top(userid = 1, K = 3):
    import MLP

    dataset_name = "study"
    # num_factors = 8
    layers = [64, 32, 16, 8]
    # 주의: Dataset 클래스는 별도의 Dataset.py 파일에 정의되어 있어야 합니다.
    from Dataset import Dataset 
    dataset = Dataset('Data/' + dataset_name)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape

    model = MLP.get_model(num_users, num_items, layers=layers, reg_layers=[0,0,0,0])

    # 2. 저장된 가중치 불러오기
    model.load_weights(f'Pretrain/{dataset_name}_MLP.weights.h5')
    users = np.array([userid] * (num_items-1)) # [1, 1, 1, ..., 1] (100개)
    items = np.array(range(1, num_items)) # (100개)
    pred = model.predict([users, items])

    map_item_score = {items[i]: pred[i] for i in range(num_items-1)}
    
    ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)

    user_meta_dict = dataset.usermeta
    userid = 0
    print(user_meta_dict[userid])

    
    print("해당 학생의 실력 : ", user_meta_dict[userid]["position"])
    # 과목명 앞에 'LP'를 붙여서 쉼표로 연결
    print(f"해당 학생이 들은 과목 : {', '.join(['LP' + str(data) for data in user_meta_dict[userid]['items']])}")

    print(f"추천 학습 경로: ")
    for itemid in ranklist:
        print(f"LP{itemid}", end=' ')


get_top()