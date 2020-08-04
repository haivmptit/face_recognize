import numpy as np
from numpy import load
from numpy.linalg import norm
def similarity_cos(emb1,emb2): # tính độ tương tự giữa 2 vecto
    sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return sim

data = load('../face_recognize/faces-embeddings.npz') #load emberdings
dataX, data_y, testX, testy = list(data['arr_0']),list(data['arr_1']), list(data['arr_2']), list(data['arr_3'])
# testX la tap emberding can test
print(np.shape(dataX))
name = [] # khoi tao list ten cac khuon mat
for i in range(len(testX)):
    sim = similarity_cos(testX[i],dataX[0] ) # khoi tao sim
    index = 0 # khoi tao index  khuon mat du doan
    for j in range(len(dataX)): # duyet cac khuon mat trong trong tap val
        if similarity_cos(testX[i],dataX[j]) > sim: # so sanh khuon mat trong data co sim lon nhat
            sim = similarity_cos(testX[i],dataX[j]) # cap nhat lai sim
            index =j # cap nhat lai index khuon mat trong tap data
    if sim >= 0.6 : # thiet lap nguong so sanh
        name.append(data_y[index]) # them du doan ten khuon mat
        print("Predict", sim, data_y[index])
    else:
        name.append('Unknown')
        print('Predict: Unknown')
    print("Test: ", testy[i])
    print("------------------------------")
same = 0 # khoi tao dem
for i in range(len(name)):
    if(name[i] == testy[i]):
        same += 1
print("Xác xuất dự đoán đúng là :" , round(float(same/len(name)) * 100, 2) , "%")
