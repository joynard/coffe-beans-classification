# ==============================================================================
# Versi Final: Pipeline Multi-Model & Multi-Dataset
# Melatih semua 6 model (2 jenis model pada 3 dataset)
# ==============================================================================
import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Fungsi preprocessing & ekstraksi fitur (TIDAK BERUBAH)
def normalize_image(image, target_size=(512, 512)):
    h, w = image.shape[:2]; scale = min(target_size[0]/h, target_size[1]/w); new_h, new_w = int(h * scale), int(w * scale); resized = cv2.resize(image, (new_w, new_h)); delta_w = target_size[1] - new_w; delta_h = target_size[0] - new_h; top, bottom = delta_h//2, delta_h-(delta_h//2); left, right = delta_w//2, delta_w-(delta_w//2); padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255]); return padded
def preprocess_image(image_path):
    img = cv2.imread(image_path);
    if img is None: return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); normalized_img = normalize_image(img_rgb); hsv = cv2.cvtColor(normalized_img, cv2.COLOR_RGB2HSV); s_channel = hsv[:,:,1]; saturation_mask = cv2.threshold(s_channel, 40, 255, cv2.THRESH_BINARY)[1]; v_channel = hsv[:,:,2]; value_mask = cv2.threshold(v_channel, 180, 255, cv2.THRESH_BINARY_INV)[1]; combined_mask = cv2.bitwise_and(saturation_mask, value_mask); kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)); cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3); contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
    if not contours: return None, None
    largest_contour = max(contours, key=cv2.contourArea); final_mask = np.zeros_like(cleaned_mask); cv2.drawContours(final_mask, [largest_contour], -1, 255, -1); segmented = cv2.bitwise_and(normalized_img, normalized_img, mask=final_mask); return segmented, final_mask
def extract_features_v1(image, mask):
    features = {}; props_list = regionprops(label(mask));
    if not props_list: return None
    props = props_list[0]; features['area'] = props.area; features['perimeter'] = props.perimeter; features['circularity'] = (4 * np.pi * props.area) / (props.perimeter ** 2 + 1e-6); features['aspect_ratio'] = props.major_axis_length / (props.minor_axis_length + 1e-6); hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV);
    for i, channel in enumerate(['h_mean', 's_mean', 'v_mean']): features[channel] = np.mean(hsv[:, :, i][mask > 0])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY); angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]; glcm = graycomatrix(gray, distances=[1], angles=angles, levels=256, symmetric=True, normed=True); features['contrast'] = graycoprops(glcm, 'contrast').mean(); features['energy'] = graycoprops(glcm, 'energy').mean(); features['homogeneity'] = graycoprops(glcm, 'homogeneity').mean(); features['correlation'] = graycoprops(glcm, 'correlation').mean(); return features

# Fungsi utama untuk menjalankan seluruh proses
def main():
    # === DISESUAIKAN: Aktifkan SEMUA model yang ingin Anda latih ===
    models_to_train = {
        'ANN': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }

    # === DISESUAIKAN: Masukkan SEMUA nama folder dataset Anda ===
    datasets_to_process = [
        '1_clean_studio', 
        '2_augmented_lighting', 
        '3_highly_varied'
    ]
    classes = ["liberica", "arabica", "robusta"]

    if not os.path.exists('model'):
        os.makedirs('model')

    # Looping untuk setiap dataset
    for dataset_folder in datasets_to_process:
        input_path = os.path.join('datasets', dataset_folder)
        if not os.path.exists(input_path):
            print(f"Peringatan: Dataset '{input_path}' tidak ditemukan, melewati...")
            continue
        
        print(f"\n=================================================")
        print(f"MEMPROSES DATASET: {dataset_folder}")
        print(f"=================================================")

        # Ekstraksi Fitur dari dataset saat ini
        data = []
        for class_name in classes:
            class_dir = os.path.join(input_path, class_name)
            if not os.path.exists(class_dir): continue
            for img_file in sorted(os.listdir(class_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    segmented, mask = preprocess_image(img_path)
                    if segmented is None: continue
                    features = extract_features_v1(segmented, mask)
                    if features is None: continue
                    features['class'] = class_name
                    data.append(features)
        
        if not data:
            print(f"Tidak ada data valid di {dataset_folder}, melewati...")
            continue

        # Persiapan Data untuk dataset saat ini
        df = pd.DataFrame(data)
        X = df.drop(columns=['class'])
        y = df['class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Looping untuk melatih setiap model pada dataset saat ini
        for model_name, model_obj in models_to_train.items():
            save_name = f"{model_name}_{dataset_folder}"
            print(f"\n--- Melatih {save_name} ---")
            model_obj.fit(X_train_scaled, y_train)
            
            y_pred = model_obj.predict(X_test_scaled)
            report = classification_report(y_test, y_pred, output_dict=True)
            accuracy = accuracy_score(y_test, y_pred)
            report['overall_accuracy'] = accuracy
            print(f"Akurasi untuk {save_name}: {accuracy * 100:.2f}%")
            
            joblib.dump(model_obj, f'model/{save_name}.pkl')
            joblib.dump(scaler, f'model/{save_name}_scaler.pkl')
            with open(f'model/{save_name}_metrics.json', 'w') as f:
                json.dump(report, f, indent=4)
            print(f"File untuk {save_name} telah disimpan.")

if __name__ == '__main__':
    main()