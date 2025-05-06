"""Một ví dụ về cách sử dụng tập dữ liệu riêng để huấn luyện bộ phân loại nhận diện người.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import các thư viện cần thiết
import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import math
import pickle
from sklearn.svm import SVC

# ============ ĐƯỜNG DẪN TUYỆT ĐỐI - CÓ THỂ THAY ĐỔI ============
# Thay đổi đường dẫn này khi cần di chuyển thư mục dự án
BASE_DIR = "E:\\PythonProject\\AI"
# ==========================================================

print(f"Thư mục gốc: {BASE_DIR}")
sys.path.insert(0, BASE_DIR)
import AI.src.facenet as facenet

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.compat.v1.Session() as sess:
            
            # Đặt seed cho tính ngẫu nhiên để kết quả có thể tái tạo
            np.random.seed(seed=args.seed)
            
            # Xử lý tập dữ liệu: chia hoặc sử dụng toàn bộ
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if (args.mode=='TRAIN'):
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # Kiểm tra rằng mỗi lớp có ít nhất một ảnh huấn luyện
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'Phải có ít nhất một ảnh cho mỗi lớp trong tập dữ liệu')

            # Lấy đường dẫn ảnh và nhãn từ dataset     
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('Số lượng lớp: %d' % len(dataset))
            print('Số lượng ảnh: %d' % len(paths))
            
            # Tải mô hình
            print('Đang tải mô hình trích xuất đặc trưng')
            facenet.load_model(args.model)
            
            # Lấy các tensor đầu vào và đầu ra
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Chạy forward pass để tính toán các embedding
            print('Đang tính toán đặc trưng cho các ảnh')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode=='TRAIN'):
                # Huấn luyện bộ phân loại
                print('Đang huấn luyện bộ phân loại')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)
            
                # Tạo danh sách tên các lớp
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Lưu mô hình bộ phân loại
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Đã lưu mô hình bộ phân loại vào file "%s"' % classifier_filename_exp)
                
            elif (args.mode=='CLASSIFY'):
                # Phân loại ảnh
                print('Đang kiểm tra bộ phân loại')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Đã tải mô hình bộ phân loại từ file "%s"' % classifier_filename_exp)

                # Dự đoán xác suất các lớp
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                # In kết quả phân loại
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                # Tính độ chính xác
                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Độ chính xác: %.3f' % accuracy)
                
            
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    """
    Chia tập dữ liệu thành tập huấn luyện và kiểm tra
    """
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Loại bỏ các lớp có ít hơn min_nrof_images_per_class ảnh
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

            
def parse_arguments(argv):
    """
    Xử lý các tham số dòng lệnh
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Chỉ định nếu nên huấn luyện bộ phân loại mới hoặc sử dụng mô hình phân loại có sẵn', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Đường dẫn đến thư mục dữ liệu chứa ảnh khuôn mặt đã căn chỉnh.')
    parser.add_argument('model', type=str, 
        help='Có thể là thư mục chứa meta_file và ckpt_file hoặc file mô hình protobuf (.pb)')
    parser.add_argument('classifier_filename', 
        help='Tên file mô hình phân loại dạng pickle (.pkl). ' + 
        'Đối với huấn luyện, đây là đầu ra và đối với phân loại, đây là đầu vào.')
    parser.add_argument('--use_split_dataset', 
        help='Chỉ định rằng tập dữ liệu trong data_dir nên được chia thành tập huấn luyện và kiểm tra. ' +  
        'Nếu không, một tập kiểm tra riêng có thể được chỉ định bằng tùy chọn test_data_dir.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Đường dẫn đến thư mục dữ liệu kiểm tra chứa ảnh đã căn chỉnh dùng cho kiểm tra.')
    parser.add_argument('--batch_size', type=int,
        help='Số lượng ảnh xử lý trong một batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Kích thước ảnh (chiều cao, chiều rộng) theo pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Seed ngẫu nhiên.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Chỉ bao gồm các lớp có ít nhất số lượng ảnh này trong tập dữ liệu', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Sử dụng số lượng ảnh này từ mỗi lớp cho huấn luyện và phần còn lại cho kiểm tra', default=10)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
