import os
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# set seeds
np.random.seed(42)
tf.random.set_seed(42)

# file paths
BASE_DIR = "Data_S25"
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
TEST_DIR = os.path.join(BASE_DIR, "Test")

RESIZED_50_DIR = "processed_50x50"
RESIZED_200_DIR = "processed_200x200"

os.makedirs(RESIZED_50_DIR, exist_ok=True)
os.makedirs(RESIZED_200_DIR, exist_ok=True)

CATEGORIES = ['bedroom', 'desert', 'landscape', 'rainforest']


def adjust_brightness(img):
    brightness = np.mean(img) / 255
    if brightness < 0.4:
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
    elif brightness > 0.6:
        img = cv2.convertScaleAbs(img, alpha=0.7, beta=-30)
    return img


def preprocess_and_save():
    print("Preprocessing images...")
    for folder in ['Train', 'Test']:
        for category in CATEGORIES:
            path = os.path.join(BASE_DIR, folder, category)
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                adjusted = adjust_brightness(gray)
                resized_50 = cv2.resize(adjusted, (50, 50))
                resized_200 = cv2.resize(adjusted, (200, 200))

                save_folder_50 = os.path.join(RESIZED_50_DIR, folder, category)
                save_folder_200 = os.path.join(RESIZED_200_DIR, folder, category)
                os.makedirs(save_folder_50, exist_ok=True)
                os.makedirs(save_folder_200, exist_ok=True)

                cv2.imwrite(os.path.join(save_folder_50, img_name), resized_50)
                cv2.imwrite(os.path.join(save_folder_200, img_name), resized_200)


def extract_features():
    sift = cv2.SIFT_create()
    X_sift, y_sift = [], []
    X_hist, y_hist = [], []
    print("Extracting SIFT and Histogram features...")

    for category in CATEGORIES:
        path = os.path.join(RESIZED_50_DIR, 'Train', category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # sift
            keypoints, descriptors = sift.detectAndCompute(img, None)
            if descriptors is not None:
                sift_desc = np.mean(descriptors, axis=0)
            else:
                sift_desc = np.zeros(128)
            X_sift.append(sift_desc)
            y_sift.append(CATEGORIES.index(category))

            # histogram
            hist = cv2.calcHist([img], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            X_hist.append(hist)
            y_hist.append(CATEGORIES.index(category))

    with open("features.pkl", "wb") as f:
        pickle.dump((X_sift, y_sift, X_hist, y_hist), f)


def load_data_50(folder):
    X, y = [], []
    for category in CATEGORIES:
        path = os.path.join(RESIZED_50_DIR, folder, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_flat = img.flatten()
            X.append(img_flat)
            y.append(CATEGORIES.index(category))
    return np.array(X), np.array(y)


def load_data_200(folder):
    X, y = [], []
    for category in CATEGORIES:
        path = os.path.join(RESIZED_200_DIR, folder, category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img.reshape(200, 200, 1)
            X.append(img)
            y.append(CATEGORIES.index(category))
    return np.array(X), np.array(y)


def evaluate_model(y_true, y_pred, title=""):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    FP = np.sum(cm, axis=0) - np.diag(cm)
    FN = np.sum(cm, axis=1) - np.diag(cm)
    total = np.sum(cm)
    fp_rate = np.sum(FP) / total
    fn_rate = np.sum(FN) / total
    print(f"\n=== {title} ===")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"False Positive Rate: {fp_rate * 100:.2f}%")
    print(f"False Negative Rate: {fn_rate * 100:.2f}%")
    return acc, fp_rate, fn_rate


def run_classifiers():
    preprocess_and_save()
    extract_features()

    # pixel neural network
    X_train_pix, y_train_pix = load_data_50('Train')
    X_test_pix, y_test_pix = load_data_50('Test')

    knn_pix = KNeighborsClassifier(n_neighbors=1)
    knn_pix.fit(X_train_pix, y_train_pix)
    y_pred_pix = knn_pix.predict(X_test_pix)
    evaluate_model(y_test_pix, y_pred_pix, title="Pixel NN")

    # sift neural network
    with open("features.pkl", "rb") as f:
        X_sift, y_sift, X_hist, y_hist = pickle.load(f)

    knn_sift = KNeighborsClassifier(n_neighbors=1)
    knn_sift.fit(X_sift, y_sift)

    X_test_sift, y_test_sift = [], []
    sift = cv2.SIFT_create()
    for category in CATEGORIES:
        path = os.path.join(RESIZED_50_DIR, 'Test', category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            keypoints, descriptors = sift.detectAndCompute(img, None)
            if descriptors is not None:
                sift_desc = np.mean(descriptors, axis=0)
            else:
                sift_desc = np.zeros(128)
            X_test_sift.append(sift_desc)
            y_test_sift.append(CATEGORIES.index(category))

    y_pred_sift = knn_sift.predict(X_test_sift)
    evaluate_model(y_test_sift, y_pred_sift, title="SIFT NN")

    # histogram neural network
    knn_hist = KNeighborsClassifier(n_neighbors=1)
    knn_hist.fit(X_hist, y_hist)

    X_test_hist, y_test_hist = [], []
    for category in CATEGORIES:
        path = os.path.join(RESIZED_50_DIR, 'Test', category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            hist = cv2.calcHist([img], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            X_test_hist.append(hist)
            y_test_hist.append(CATEGORIES.index(category))

    y_pred_hist = knn_hist.predict(X_test_hist)
    evaluate_model(y_test_hist, y_pred_hist, title="Histogram NN")


def run_transfer_learning_cnn():
    print("\n=== Starting Transfer Learning CNN ===")
    X_train_cnn, y_train_cnn = load_data_200('Train')
    X_test_cnn, y_test_cnn = load_data_200('Test')

    # 1 channel to 3 channel grayscale
    X_train_cnn_rgb = np.repeat(X_train_cnn, 3, axis=-1)
    X_test_cnn_rgb = np.repeat(X_test_cnn, 3, axis=-1)

    y_train_cat = to_categorical(y_train_cnn, num_classes=4)
    y_test_cat = to_categorical(y_test_cnn, num_classes=4)

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(4, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train_cnn_rgb, y_train_cat, epochs=20, batch_size=32, validation_data=(X_test_cnn_rgb, y_test_cat))

    y_pred_cnn = model.predict(X_test_cnn_rgb)
    y_pred_cnn_labels = np.argmax(y_pred_cnn, axis=1)
    evaluate_model(y_test_cnn, y_pred_cnn_labels, title="Transfer Learning CNN (MobileNetV2)")


if __name__ == "__main__":
    run_classifiers()
    run_transfer_learning_cnn()
