from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import os
from os import listdir

#Tách tên người từ tên file
def get_name(name_file):
    s = name_file.split('.')
    return s[0]

# trích rút khuôn mặt người từ file ảnh
def extract_face(filename, required_size=(160, 160)): # filename đường dẫn ảnh
    # filename đường dẫn ảnh.
    # required_size=(160, 160): kích thước ảnh (dài x rộng) trong Input-layer của model face-net
    # load ảnh và đưa ảnh về dạng mảng ma trận 3 chiều (kích thước mảng:  160x160x 3 ).
    image = Image.open(filename)
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # khởi tạo lớp mtcnn để phát hiện khuôn mặt
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    if(len(results) > 0 ): # kiểm tra ảnh có chưa khuôn mặt hay không
        # Lấy tọa độ góc thuộc đường bao khuôn mặt.
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        # cắt khuôn mặt
        face = pixels[y1:y2, x1:x2]
        # Thay đổi kích thước đường bao khuôn mặt (dài x rộng => 160x160).
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
    else:
        os.remove(filename);
        return None

#Load ảnh trong thư mục data/val
def load_dataset(directory):
    X, y = list(), list() # X : danh sách chua khuon mat, y: nhan cua moi anh
	#Duyệt các ảnh trong mỗi thư mục
    for subdir in listdir(directory):
        file = directory + subdir
        face = extract_face(file)
        print('<3 Loaded the image for datasets: %s' % (subdir))
        if face is not None:
            X.append(face)
            y.append(get_name(subdir))
    return asarray(X), asarray(y)
