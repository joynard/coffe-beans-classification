import os
import json
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import joblib
import pandas as pd
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops

# Mendapatkan direktori tempat skrip app.py ini berada
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi path absolut
app.config['UPLOAD_FOLDER'] = os.path.join(SCRIPT_DIR, 'uploads')
app.config['MODEL_FOLDER'] = os.path.join(SCRIPT_DIR, 'model')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Memuat data informasi kopi
JSON_PATH = os.path.join(SCRIPT_DIR, 'coffee_info.json')
try:
    with open(JSON_PATH, 'r', encoding='utf-8') as f: coffee_info_data = json.load(f)
    print(">>> Informasi kopi berhasil dimuat.")
except FileNotFoundError:
    coffee_info_data = {}; print(f">>> PERINGATAN: coffee_info.json tidak ditemukan di path: {JSON_PATH}")

# Fungsi memuat semua model
def load_all_models():
    models = {}; path = app.config['MODEL_FOLDER']
    if not os.path.exists(path): return {}
    for file in os.listdir(path):
        if file.endswith('.pkl') and '_scaler' not in file:
            model_name = file.replace('.pkl', '')
            try:
                model_path = os.path.join(path, file)
                scaler_path = os.path.join(path, f"{model_name}_scaler.pkl")
                metrics_path = os.path.join(path, f"{model_name}_metrics.json")
                model_data = {'model': joblib.load(model_path), 'scaler': joblib.load(scaler_path)}
                with open(metrics_path, 'r') as f: model_data['metrics'] = json.load(f)
                models[model_name] = model_data; print(f"Loaded model: {model_name}")
            except FileNotFoundError: print(f"Warning: Could not load all files for model '{model_name}'. Skipping.")
    return models
models_data = load_all_models()

# Fungsi memeriksa ekstensi file
def allowed_file(filename): return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi Klasifikasi dengan tambahan Logika Threshold
def classify_coffee_bean(file_path, model_name):
    if not models_data or model_name not in models_data: return "Error: Model tidak tersedia.", 0.0, None
    model_info = models_data[model_name]; model = model_info['model']; scaler = model_info['scaler']; metrics = model_info['metrics']
    
    segmented, mask = preprocess_image(file_path)
    if segmented is None: return "Tidak dapat memproses gambar.", 0.0, metrics
    
    features = extract_features_v1(segmented, mask)
    if features is None: return "Tidak dapat mengekstrak fitur.", 0.0, metrics
    
    df_features = pd.DataFrame([features]); scaled_features = scaler.transform(df_features)
    probabilities = model.predict_proba(scaled_features)[0]
    confidence = np.max(probabilities)
    
    # ==============================================================================
    # BAGIAN BARU: Threshold Keyakinan 75%
    # Jika keyakinan tertinggi kurang dari 75%, anggap "Tidak Dikenali"
    # ==============================================================================
    CONFIDENCE_THRESHOLD = 0.75 
    if confidence < CONFIDENCE_THRESHOLD:
        return "Tidak Dikenali / Objek Asing", confidence, metrics
    # ==============================================================================

    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = model.classes_[predicted_class_index]
    
    return predicted_class_name.capitalize(), confidence, metrics

# Rute utama (tidak berubah)
@app.route('/', methods=['GET', 'POST'])
def upload_and_classify():
    model_names = list(models_data.keys())
    if request.method == 'POST':
        if 'file' not in request.files: return render_template('index.html', model_names=model_names, error='Tidak ada file yang dipilih')
        file = request.files['file']; chosen_model = None
        if len(model_names) > 1: chosen_model = request.form.get('model_choice')
        elif len(model_names) == 1: chosen_model = model_names[0]
        if file.filename == '': return render_template('index.html', model_names=model_names, error='Tidak ada file yang dipilih')
        if file and allowed_file(file.filename) and chosen_model:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction_result, confidence_score, metrics_result = classify_coffee_bean(file_path, chosen_model)
            predicted_key = prediction_result.lower()
            coffee_info = coffee_info_data.get(predicted_key, None)
            return render_template('result.html', prediction=prediction_result, confidence=confidence_score, metrics=metrics_result, chosen_model=chosen_model, image_filename=filename, coffee_info=coffee_info)
    return render_template('index.html', model_names=model_names)

# Rute untuk menyajikan gambar (tidak berubah)
@app.route('/uploads/<filename>')
def uploaded_file(filename): return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Menjalankan server (tidak berubah)
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])