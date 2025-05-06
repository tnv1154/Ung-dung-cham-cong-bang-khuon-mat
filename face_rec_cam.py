from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter

import tensorflow as tf
from imutils.video import VideoStream
import imutils
import os
import sys
import pickle
import numpy as np
import cv2
import collections
import time

# ============ ĐƯỜNG DẪN TUYỆT ĐỐI - CÓ THỂ THAY ĐỔI ============
# Thay đổi đường dẫn này khi cần di chuyển thư mục dự án
BASE_DIR = "E:\\PythonProject\\AI"
# ==========================================================

print(f"Thư mục gốc: {BASE_DIR}")
sys.path.insert(0, BASE_DIR)
from AI.src.align import detect_face
import AI.src.facenet as facenet

# Singleton class để đảm bảo model chỉ được tải một lần
class FaceNetModelSingleton:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        if FaceNetModelSingleton._instance is not None:
            raise Exception("Singleton class - sử dụng get_instance() để lấy instance")
        
        self.MINSIZE = 20
        self.THRESHOLD = [0.6, 0.7, 0.7]
        self.FACTOR = 0.709
        self.IMAGE_SIZE = 182
        self.INPUT_IMAGE_SIZE = 160
        self.CLASSIFIER_PATH = os.path.join(BASE_DIR, 'Models', 'facemodel.pkl')
        self.FACENET_MODEL_PATH = os.path.join(BASE_DIR, 'Models', '20180402-114759.pb')
        
        self.model = None
        self.class_names = None
        self.graph = None
        self.sess = None
        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder = None
        self.pnet = None
        self.rnet = None
        self.onet = None
        
        self._is_initialized = False
    
    def initialize(self):
        """Tải model và khởi tạo session - chỉ gọi khi cần thiết"""
        if self._is_initialized:
            return
            
        print("Đang khởi tạo FaceNet model...")
        start_time = time.time()
        
        # Load The Custom Classifier
        with open(self.CLASSIFIER_PATH, 'rb') as file:
            self.model, self.class_names = pickle.load(file)
        print("Custom Classifier, Successfully loaded")
        
        # Khởi tạo tensorflow
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            self.gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False))
            
            with self.sess.as_default():
                # Load the model
                print('Loading feature extraction model')
                facenet.load_model(self.FACENET_MODEL_PATH)
                
                # Get input and output tensors
                self.images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]
                
                align_dir = os.path.join(BASE_DIR, "src", "align")
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, align_dir)
        
        self._is_initialized = True
        end_time = time.time()
        print(f"Khởi tạo model hoàn tất trong {end_time - start_time:.2f} giây")

class FaceRegcognitionCam:
    def __init__(self):
        self.model_singleton = FaceNetModelSingleton.get_instance()
        # Không khởi tạo model ở constructor để tránh tải model khi chưa cần
        
        self.id_arr = []  # Mảng lưu id
        self.person_detected = collections.Counter()
        self.people_detected = set()
        
    def ensure_model_loaded(self):
        """Đảm bảo model đã được tải trước khi sử dụng"""
        if not self.model_singleton._is_initialized:
            self.model_singleton.initialize()
        
    def process_frame(self, frame):
        """Xử lý một frame và trả về frame đã được vẽ và ID nếu nhận diện được"""
        # Đảm bảo model đã được tải
        self.ensure_model_loaded()
        
        detected_id = None
        
        if frame is None:
            return frame, detected_id
            
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, 1)
        
        with self.model_singleton.graph.as_default():
            with self.model_singleton.sess.as_default():
                bounding_boxes, _ = detect_face.detect_face(frame, self.model_singleton.MINSIZE, 
                                                           self.model_singleton.pnet, 
                                                           self.model_singleton.rnet, 
                                                           self.model_singleton.onet, 
                                                           self.model_singleton.THRESHOLD, 
                                                           self.model_singleton.FACTOR)
                
                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 1:
                        cv2.putText(frame, "Only one face", (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (255, 255, 255), thickness=1, lineType=2)
                    elif faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            
                            if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25:
                                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (self.model_singleton.INPUT_IMAGE_SIZE, self.model_singleton.INPUT_IMAGE_SIZE),
                                                interpolation=cv2.INTER_CUBIC)
                                scaled = facenet.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1, self.model_singleton.INPUT_IMAGE_SIZE, self.model_singleton.INPUT_IMAGE_SIZE, 3)
                                feed_dict = {self.model_singleton.images_placeholder: scaled_reshape, 
                                           self.model_singleton.phase_train_placeholder: False}
                                emb_array = self.model_singleton.sess.run(self.model_singleton.embeddings, feed_dict=feed_dict)

                                predictions = self.model_singleton.model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                best_name = self.model_singleton.class_names[best_class_indices[0]]
                                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                                if best_class_probabilities > 0.8:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20

                                    name = self.model_singleton.class_names[best_class_indices[0]]
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (255, 255, 255), thickness=1, lineType=2)
                                    self.person_detected[best_name] += 1
                                    self.id_arr.append(int(name))
                                    detected_id = int(name)
                                else:
                                    name = "Unknown"
                except:
                    pass
                    
        return frame, detected_id
        
    def get_most_common_id(self):
        """Trả về ID xuất hiện nhiều nhất trong các frame đã xử lý"""
        if not self.id_arr:
            return None
            
        counter = Counter(self.id_arr)
        # Tìm phần tử xuất hiện nhiều nhất
        id_employee = counter.most_common(1)[0][0]
        return id_employee
        
    def reset(self):
        """Reset lại các biến theo dõi"""
        self.id_arr = []
        self.person_detected = collections.Counter()
        
    def face_recognition_cam(self):
        """Hàm cũ để tương thích ngược - chỉ trả về kết quả từ main()"""
        # Sử dụng instance đã tạo
        self.ensure_model_loaded()
        return main_optimized()

# Phiên bản tối ưu của hàm main, sử dụng singleton model
def main_optimized():
    face_rec = FaceRegcognitionCam()
    face_rec.ensure_model_loaded()
    model_singleton = FaceNetModelSingleton.get_instance()
    
    id_arr = []  # Mảng lưu id
    person_detected = collections.Counter()
    people_detected = set()

    # Thay đổi src=1 thành src=0 (sử dụng camera mặc định)
    cap = VideoStream(src=0).start()
    print("Đang khởi động camera...")
    time.sleep(2.0)  # Cho camera thời gian khởi động

    while (True):
        frame = cap.read()
        if frame is None:
            print("Không thể đọc frame từ camera. Đang thử lại...")
            time.sleep(0.1)
            continue
        
        processed_frame, detected_id = face_rec.process_frame(frame)
        if detected_id is not None:
            id_arr.append(detected_id)

        cv2.imshow('Face Recognition', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Sửa release() thành stop() cho đối tượng VideoStream
    cap.stop()
    cv2.destroyAllWindows()
    
    if not id_arr:
        return None
        
    counter = Counter(id_arr)
    # Tìm phần tử xuất hiện nhiều nhất
    id_employee = counter.most_common(1)[0][0]
    return id_employee

# Giữ lại hàm main() cho khả năng tương thích ngược - nhưng gọi phiên bản tối ưu
def main():
    return main_optimized()

if __name__ == "__main__":
    main()
