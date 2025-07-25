import sys
import os
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import joblib
import numpy as np
from model import SalaryModel

app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR

print(f"\n Working Directory: {os.getcwd()}")
print(f" Directory Contents: {os.listdir(BASE_DIR)}")

def load_artifacts():
    global model, scaler_x, scaler_y
    try:
        # File paths
        model_path = ARTIFACTS_DIR / "final_salary_model.pth"
        scaler_x_path = ARTIFACTS_DIR / "final_scaler_x.pkl"
        scaler_y_path = ARTIFACTS_DIR / "final_scaler_y.pkl"

        print(f"\n Verifying artifacts:")
        print(f"- Model exists: {os.path.exists(model_path)}")
        print(f"- Scaler X exists: {os.path.exists(scaler_x_path)}")
        print(f"- Scaler Y exists: {os.path.exists(scaler_y_path)}")

        # Initialize model
        model = SalaryModel(input_features=3)
        
        # Debug model architecture
        print("\n Model Architecture Before Loading:")
        print(model)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # Verify loaded weights
        print("\n Model Weights Loaded Successfully")
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}")

        # Load scalers
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        
        print("\n Scaler Statistics:")
        print(f"Scaler X mean: {scaler_x.mean_}")
        print(f"Scaler Y scale: {scaler_y.scale_}")

        # Test prediction
        test_level = 5
        test_input = np.array([[test_level, test_level**2, test_level**3]], dtype=np.float32)
        test_scaled = scaler_x.transform(test_input)
        with torch.no_grad():
            pred = model(torch.tensor(test_scaled))
            salary = scaler_y.inverse_transform(pred.numpy())[0][0]
        print(f"\n Test Prediction (Level 5): ${salary:,.2f}")

    except Exception as e:
        print(f"\n LOAD ERROR: {str(e)}")
        if 'state_dict' in str(e):
            print("\n🔧 Possible Solutions:")
            print("1. Verify model.py matches the training architecture")
            print("2. Check if you modified the model after training")
            print("3. Try loading with strict=False (already implemented)")
            print("4. Compare layer names in saved vs current model:")
            if 'state_dict' in locals():
                print("\nSaved model keys:")
                print(state_dict.keys())
                print("\nCurrent model keys:")
                print(model.state_dict().keys())
        raise SystemExit(1)

load_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debug raw request data
        print("\n Received request data:", request.get_data())
        
        data = request.get_json()
        if data is None:
            return jsonify({
                'error': 'Invalid JSON data',
                'status': 'error'
            }), 400
            
        if 'level' not in data:
            return jsonify({
                'error': 'Missing "level" parameter',
                'status': 'error'
            }), 400
            
        try:
            level = float(data['level'])
        except (ValueError, TypeError):
            return jsonify({
                'error': 'Level must be a number',
                'status': 'error'
            }), 400
            
        if not 1 <= level <= 10:
            return jsonify({
                'error': 'Level must be between 1-10',
                'status': 'error'
            }), 400
            
        # Prepare input
        level_poly = np.array([[level, level**2, level**3]], dtype=np.float32)
        level_scaled = scaler_x.transform(level_poly)
        
        # Predict
        with torch.no_grad():
            pred = model(torch.tensor(level_scaled))
            salary = scaler_y.inverse_transform(pred.numpy())[0][0]
        
        return jsonify({
            'predicted_salary': round(float(salary), 2),
            'status': 'success'
        })
        
    except Exception as e:
        print(f"\n Prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)