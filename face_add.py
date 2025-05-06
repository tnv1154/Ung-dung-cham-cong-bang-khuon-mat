from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from imutils.video import VideoStream
import imutils
import os
import sys
import cv2
import time
import subprocess

# ============ ĐƯỜNG DẪN TUYỆT ĐỐI - CÓ THỂ THAY ĐỔI ============
# Thay đổi đường dẫn này khi cần di chuyển thư mục dự án
BASE_DIR = "E:\\PythonProject\\AI"
# ==========================================================

print(f"Thư mục gốc: {BASE_DIR}")
sys.path.insert(0, BASE_DIR)
from src.align import detect_face

class Face_Add:
    def __init__(self,id_employee):
        self.face_add= main(id_employee)

def main(id_employee):
    
    # Thông số camera và ảnh
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    NUM_IMAGES = 50
    CAPTURE_INTERVAL = 0.25 #khoảng cách chụp giữa mỗi ảnh
    CAMERA_INDEX = 0
    
    # Thông số MTCNN để phát hiện khuôn mặt
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    
    # Đường dẫn thư mục
    RAW_FOLDER = os.path.join(BASE_DIR, "DataSet", "FaceData", "raw")
    PROCESSED_FOLDER = os.path.join(BASE_DIR, "DataSet", "FaceData", "processed")
    MODEL_PATH = os.path.join(BASE_DIR, "Models", "20180402-114759.pb")
    OUTPUT_CLASSIFIER = os.path.join(BASE_DIR, "Models", "facemodel.pkl")
    
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(RAW_FOLDER):
        os.makedirs(RAW_FOLDER)
    
    # Nhập tên người cần thêm
    person_id = str(id_employee)  # Convert id_employee from int to str
    person_folder = os.path.join(RAW_FOLDER, person_id)
    
    # Tạo thư mục cho người này nếu chưa tồn tại
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    else:
        print(f"Thư mục cho {person_id} đã tồn tại!")
        overwrite = input("Bạn có muốn tiếp tục và ghi đè dữ liệu cũ? (y/n): ")
        if overwrite.lower() != 'y':
            print("Hủy thao tác.")
            return
    
    print(f"Chuẩn bị chụp {NUM_IMAGES} ảnh cho {person_id}...")
    print("Đảm bảo khuôn mặt nằm trong khung hình và di chuyển đầu nhẹ nhàng để có các góc nhìn khác nhau.")
    print("Nhấn 'q' để dừng chương trình bất kỳ lúc nào.")
    time.sleep(2)  # Chờ 2s để người dùng chuẩn bị
    
    # Khởi tạo TensorFlow và MTCNN
    with tf.compat.v1.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.25)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        
        with sess.as_default():
            # Tạo mạng MTCNN
            align_dir = os.path.join(BASE_DIR, "src", "align")
            pnet, rnet, onet = detect_face.create_mtcnn(sess, align_dir)
            
            # Khởi tạo camera
            cap = VideoStream(src=CAMERA_INDEX).start()
            time.sleep(1.0)  # Cho phép camera khởi động
            
            count = 0
            last_capture_time = time.time()
            
            while count < NUM_IMAGES:
                frame = cap.read()
                if frame is None:
                    print("Không thể đọc frame từ camera. Đang thử lại...")
                    time.sleep(0.1)
                    continue
                    
                frame = imutils.resize(frame, width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
                frame = cv2.flip(frame, 1)  # Lật ngang để giống gương
                
                # Phát hiện khuôn mặt
                bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
                
                # Vẽ hình chữ nhật xung quanh khuôn mặt
                faces_found = bounding_boxes.shape[0]
                current_time = time.time()
                
                if faces_found > 0:
                    for i in range(min(faces_found, 1)):  # Chỉ xử lý khuôn mặt đầu tiên
                        det = bounding_boxes[i, 0:4]
                        bb = [int(det[0]), int(det[1]), int(det[2]), int(det[3])]
                        cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
                        
                        # Chụp ảnh sau mỗi khoảng thời gian CAPTURE_INTERVAL
                        if current_time - last_capture_time >= CAPTURE_INTERVAL and count < NUM_IMAGES:
                            # Lưu ảnh
                            image_path = os.path.join(person_folder, f"{person_id}_{count+1:03d}.jpg")
                            cv2.imwrite(image_path, frame)
                            count += 1
                            last_capture_time = current_time
                            print(f"Đã chụp ảnh {count}/{NUM_IMAGES}")
                else:
                    cv2.putText(frame, "Không phát hiện khuôn mặt", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Hiển thị thông tin
                cv2.putText(frame, f"Ảnh đã chụp: {count}/{NUM_IMAGES}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Hiển thị frame
                cv2.imshow("Thêm khuôn mặt mới", frame)
                
                # Nhấn 'q' để thoát
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Giải phóng tài nguyên
            cap.stop()
            cv2.destroyAllWindows()
    
    print(f"Đã hoàn thành việc chụp {count} ảnh cho {person_id}.")
    
    # Thực hiện Bước 2: Cắt khuôn mặt bằng MTCNN
    print("\nĐang thực hiện Bước 2: Cắt khuôn mặt bằng MTCNN...")
    align_script = os.path.join(BASE_DIR, "src", "align_dataset_mtcnn.py")
    align_command = [
        sys.executable, align_script,
        RAW_FOLDER, PROCESSED_FOLDER,
        "--image_size", "160",
        "--margin", "32",
        "--random_order",
        "--gpu_memory_fraction", "0.25"
    ]
    
    try:
        subprocess.run(align_command, check=True)
        print("Bước 2 đã hoàn thành thành công!")
        
        # Thực hiện Bước 3: Huấn luyện mô hình
        print("\nĐang thực hiện Bước 3: Huấn luyện mô hình...")
        classifier_script = os.path.join(BASE_DIR, "src", "classifier.py")
        train_command = [
            sys.executable, classifier_script,
            "TRAIN", PROCESSED_FOLDER,
            MODEL_PATH, OUTPUT_CLASSIFIER,
            "--batch_size", "1000"
        ]
        
        subprocess.run(train_command, check=True)
        print("Bước 3 đã hoàn thành thành công!")
        print("\nQuá trình thêm người mới đã hoàn tất. Bạn có thể chạy chương trình nhận diện khuôn mặt ngay bây giờ.")
    
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi thực hiện các bước xử lý: {e}")

if __name__ == "__main__":
    Face_Add(1)