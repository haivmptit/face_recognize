# calculate a face embedding for each face in the dataset using facenet
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model

# get the face embedding for one face
def get_embedding(model, face_pixels):
    # Chuẩn hóa lại các pixel trong ảnh:
    # Việc chuẩn hóa sẽ đưa giá trị các pixel từ dải [0-255] về quanh vị trí trung tâm 0.
    # Chuẩn hóa sẽ giúp việc tính toán nhanh hơn, tránh hiệu ứng che mặt ( giá trị các pixel lớn lấn át hết giá trị
    # các pixcel nhỏ) mà vẫn giữ được  mối quan hệ giữa các pixcel.
    # mean: giá trị trung bình các pixel trong ảnh
    # std : độ lệch chuẩn của các pixel trong ảnh
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]


# load the face dataset
def save_emberding():
    #Load faces
    data = load('../face_recognize/faces-dataset.npz')
    dataX, data_y, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

    print('Loaded: ', dataX.shape, data_y.shape, testX.shape, testy.shape)
    # load the facenet model
    model = load_model('C:\\AI\Code\Classify-age\ClassifyAge\Process_data\\facenet_keras.h5')
    print('Loaded Model')
    new_dataX = list() # danh sách các vecto emberding khuôn mặt trong tập data

    for face_pixels in dataX:
        embedding = get_embedding(model, face_pixels) # get emberding
        new_dataX.append(embedding)
    new_dataX = asarray(new_dataX)
    print(new_dataX.shape)
    newTestX = list() # danh sách các vecto emberding khuôn mặt trong tập test
    for face_pixels in testX:
        embedding = get_embedding(model, face_pixels) # get emberding
        newTestX.append(embedding)
    newTestX = asarray(newTestX)
    print(newTestX.shape)
    savez_compressed('faces-embeddings.npz', new_dataX, data_y, newTestX, testy)

