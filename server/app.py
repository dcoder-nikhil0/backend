from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import os
import joblib
import torch
import torchvision.models as models_torch
from torchvision import transforms

from keras.models import load_model
import numpy as np
import cv2
import librosa
import traceback

# ========== Setup ==========
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.abspath('uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

NUM_CLASSES = 2
CLASS_MAP = {0: "Fake", 1: "Real"}

allowed_extensions = {
    'image': {'jpg', 'jpeg', 'png'},
    'audio': {'wav'},
    'video': {'mp4', 'avi', 'mov'}
}

def allowed_file(filename, media_type):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions[media_type]

# Detect device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Model Loading ==========
def load_resnet_model(path):
    try:
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        import torch.nn as nn
        base_model = models_torch.resnet18(pretrained=False)

        base_model.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1)
        )

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k.replace("model.", "")] = v
            else:
                new_state_dict[k] = v

        base_model.load_state_dict(new_state_dict)
        base_model.to(device)
        base_model.eval()
        print("✅ ResNet-18 model loaded successfully.")
        return base_model

    except Exception as e:
        print(f"❌ Failed to load ResNet-18 model: {e}")
        traceback.print_exc()
        return None

image_model = load_resnet_model("model/model-resnet18.pth")

try:
    audio_model = joblib.load("model/knn_model.pkl")
    print("✅ Audio model loaded.")
except Exception as e:
    print(f"❌ Error loading audio model: {e}")
    traceback.print_exc()
    audio_model = None

try:
    video_model = load_model("model/efficientnetb7_deepfake_finetuned.keras")
    print("✅ Video model loaded.")
except Exception as e:
    print(f"❌ Error loading video model: {e}")
    traceback.print_exc()
    video_model = None

models = {'image': image_model, 'audio': audio_model, 'video': video_model}

# ========== API Routes ==========
@app.route('/upload/<media_type>', methods=['POST'])
def upload_file(media_type):
    if media_type not in models:
        return jsonify({'error': 'Unsupported media type'}), 400

    model = models[media_type]
    if model is None:
        return jsonify({'error': f'Model for {media_type} not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if file and allowed_file(file.filename, media_type):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = run_detection(media_type, filepath)
        return jsonify({'message': f'{media_type.capitalize()} processed successfully.', 'result': result}), 200

    return jsonify({'error': 'Invalid file type'}), 400

# ========== Core Detection ==========
def run_detection(media_type, filepath):
    model = models[media_type]

    if media_type == 'image':
        try:
            image = cv2.imread(filepath)
            if image is None:
                return {"error": "Could not read image. Invalid file."}

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.uint8)

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output.squeeze()).item()

                # Invert prediction logic here
                pred = 0 if prob >= 0.5 else 1
                confidence = round(prob * 100 if pred == 0 else (1 - prob) * 100, 2)

            return {"label": CLASS_MAP.get(pred, str(pred)), "confidence": f"{confidence}%"}

        except Exception as e:
            app.logger.error(traceback.format_exc())
            return {"error": f"Image processing error: {str(e)}"}



    elif media_type == 'audio':
        try:
            if not filepath.lower().endswith('.wav'):
                return {"error": "Only .wav files are supported for audio."}

            y, sr = librosa.load(filepath, sr=None, mono=True)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs.T, axis=0).reshape(1, -1)

            pred = model.predict(mfccs_mean)[0]
            probs = model.predict_proba(mfccs_mean)[0]
            confidence = round(np.max(probs) * 100, 2)
            return {"label": CLASS_MAP.get(pred, str(pred)), "confidence": f"{confidence}%"}

        except Exception as e:
            app.logger.error(traceback.format_exc())
            return {"error": f"Audio processing error: {str(e)}"}

    elif media_type == 'video':
        try:
            cap = cv2.VideoCapture(filepath)
            frames = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= 10:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame.astype('float32') / 255.0
                frames.append(frame)
                frame_count += 1
            cap.release()

            if not frames:
                return {"error": "No valid frames found in video."}

            frames = np.stack(frames)
            predictions = model.predict(frames)
            avg_prediction = np.mean(predictions, axis=0)
            pred = int(np.argmax(avg_prediction))
            confidence = round(float(np.max(avg_prediction)) * 100, 2)

            return {"label": CLASS_MAP.get(pred, str(pred)), "confidence": f"{confidence}%"}

        except Exception as e:
            app.logger.error(traceback.format_exc())
            return {"error": f"Video processing error: {str(e)}"}

    return {"error": "Unsupported media type"}

# ========== Run App ==========
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

