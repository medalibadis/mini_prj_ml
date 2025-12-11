from flask import Flask, request, jsonify
import pickle
import cv2
import numpy as np
import os
# -------- Load trained data --------
with open("plant_disease_data.pkl", "rb") as f:
    data = pickle.load(f)

rules = data['rules']
all_tokens = data['all_tokens']

# -------- Token mapping for user-friendly output --------
token_mapping = {
    'R0': "Red intensity low",
    'R1': "Red intensity high",
    'G0': "Green intensity low",
    'G1': "Green intensity high",
    'B0': "Blue intensity low",
    'B1': "Blue intensity high",
    'Contrast0': "Low texture/contrast",
    'Contrast1': "High texture/contrast"
}

# -------- Feature extraction for new images --------
def extract_tokens(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_color = cv2.mean(img)[:3]  # BGR
    tokens = []

    tokens.append('R0' if mean_color[2] < 128 else 'R1')
    tokens.append('G0' if mean_color[1] < 128 else 'G1')
    tokens.append('B0' if mean_color[0] < 128 else 'B1')

    # Texture token using simple contrast
    glcm = cv2.createCLAHE().apply(img_gray)
    contrast = glcm.var()
    tokens.append('Contrast0' if contrast < 500 else 'Contrast1')

    return set(tokens)

# -------- Prediction using rules --------
def predict(tokens):
    matched = rules[rules['antecedents'].apply(lambda x: x.issubset(tokens))]
    if matched.empty:
        return {'predicted_tokens': [], 'confidence': 0, 'input_tokens': list(tokens)}
    top_rule = matched.sort_values('confidence', ascending=False).iloc[0]
    return {
        'predicted_tokens': list(top_rule['consequents']),
        'confidence': float(top_rule['confidence']),
        'input_tokens': list(tokens)
    }

def friendly_predict(img):
    tokens = extract_tokens(img)
    result = predict(tokens)
    result['input_tokens'] = [token_mapping[t] for t in result['input_tokens']]
    result['predicted_tokens'] = [token_mapping[t] for t in result['predicted_tokens']]
    result['confidence'] = f"{result['confidence']:.2f}"
    return result

# -------- Flask app --------
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    result = friendly_predict(img)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
