import cv2
import numpy as np
from numpy.linalg import norm
from face_recognize import get_emberding_faces
from PIL import Image
from numpy import asarray, load
from mtcnn.mtcnn import MTCNN
from keras.models import load_model


def similarity_cos(emb1,emb2): # tính độ tương tự giữa 2 vecto emberding khuôn mặt.
    sim = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
    return sim


def predict_names(faces):
    names = []
    model1 = load_model('C:\\AI\Code\Classify-age\ClassifyAge\Process_data\\facenet_keras.h5')
    data = load('../face_recognize//faces-embeddings.npz')  # load emberdings
    dataX, data_y, testX, testy = list(data['arr_0']), list(data['arr_1']), list(data['arr_2']), list(data['arr_3'])  # testX la tap emberding can test
    for i in range(len(faces)):
        print("Get emberding face : ", i + 1)
        face_test = faces[i]
        print("face-test", np.shape(face_test))
        emberding_test = get_emberding_faces.get_embedding(model1, face_test)  # tạo emberding
        sim = similarity_cos(emberding_test, dataX[0])  # khoi tao sim
        index = 0  # khoi tao index  khuon mat du doan

        for j in range(len(dataX)):
            tmp = similarity_cos(emberding_test, dataX[j])
            if tmp > sim:  # so sanh khuon mat trong data co sim lon nhat
                sim = tmp  # cap nhat lai sim
                index = j  # cap nhat lai index khuon mat trong tap data
        if sim > 0.65:  # thiet lap nguong so sanh
            names.append(data_y[index])  # them du doan ten khuon mat
            print("Predicted", sim, data_y[index])
        else:
            names.append('Unknown')
            print("Predicted Unknown", sim, data_y[index] )
    return names


filename = "../face_recognize/Test_images/khacviet_tuanhung.jpg"  # link ảnh test
# filename = "../face_recognize/Test_images/hoquanghieu.jpg"  # link ảnh test
#đọc file ảnh
image = Image.open(filename)
image = image.convert('RGB') # chuyển sảnh sang RGB nếu cần.
pixels = asarray(image)
detector = MTCNN()
# phát hiện khuôn mặt
results = detector.detect_faces(pixels)
face_array = [] # khởi tạo danh sách khuôn mặt
locates = [] # khởi tạo danh sách vị trí các khuôn mặt trong bức ảnh
print(len(results))

for i in range(len(results)):
    # Lấy tọa độ góc trái trên (x1,y1) và chiều dài, rộng của đường bao khuôn mặt.
    x1, y1, width, height = results[i]['box']
    x2, y2 = x1 + width, y1 + height
    locate ={ # từ điển lưu tọa độ góc trái trên (x1, y1) và góc phải dưới (x2,y2) của khuôn mặt
        'x1': x1,
        'x2': x2,
        'y1': y1,
        'y2': y2,
    }
    locates.append(locate)
    # cắt ảnh khuôn mặt.
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    imagee = Image.fromarray(face)
    imagee = imagee.resize((160, 160)) # resize lại kích thước ảnh.
    face_array.append(asarray(imagee))

names = predict_names(face_array) # đoán tên các khuôn mặt trong ảnh
img = cv2.imread(filename)
if len(names) > 0:
    for i in range(len(names)):
        #Vẽ đường bao khuôn mặt và gán tên người
        dic = locates[i]
        print(dic)
        img = cv2.rectangle(img, (dic['x1'], dic['y1']), (dic['x2'], dic['y2']), (0, 255, 0), 2)
        img = cv2.putText(img, names[i], (dic['x1'] - 10, dic['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
cv2.imshow("Result", img)
cv2.waitKey(0)
