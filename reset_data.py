from face_recognize import get_emberding_faces
from face_recognize import load_face
import numpy as np
from numpy import savez_compressed

# load data dataset
print("Loading dataset......................")
dataX, data_y = load_face.load_dataset('../face_recognize/datasets/data/')
print(dataX.shape, data_y.shape)
# load test dataset
testX, testy = load_face.load_dataset('../face_recognize/datasets/val/')
#Lưu các ảnh đã phát hiện và xử lí khuôn mặt
savez_compressed('faces-dataset.npz', dataX, data_y, testX, testy)
# load the face dataset
data = np.load('C:\AI\Code\Classify-age\ClassifyAge\Process_data\\faces-dataset.npz')
dataX, data_y, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Loaded: ', dataX.shape, data_y.shape, testX.shape, testy.shape)

# Lưu tập vecto emberding
get_emberding_faces.save_emberding()
print("saved emberdings")

