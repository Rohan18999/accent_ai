"""Flask application for accent-based cuisine recommendation using HuBERT"""

from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename

from model_utils import load_hubert_predictor
from cuisine_data import get_cuisine_recommendations

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'webm', 'm4a'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load HuBERT predictor once at startup
MODEL_PATH = 'models/hubert_layer_analysis.pth'
predictor = load_hubert_predictor(MODEL_PATH)
print("HuBERT predictor ready!")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle audio upload and prediction"""
    try:
        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict accent using HuBERT
            predicted_accent, confidence, all_probs = predictor.predict(filepath)
            
            # Get cuisine recommendations
            recommendations = get_cuisine_recommendations(predicted_accent)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            # Prepare response
            response = {
                'success': True,
                'predicted_accent': predicted_accent,
                'confidence': round(confidence * 100, 2),
                'region': recommendations['region'],
                'native_language': recommendations['native_language'],
                'cuisines': recommendations['cuisines'],
                'model_info': {
                    'model': 'HuBERT-base',
                    'layer': predictor.best_layer
                }
            }
            
            return jsonify(response)
        
        return jsonify({'error': 'Invalid file format'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'HuBERT-base',
        'best_layer': predictor.best_layer
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
