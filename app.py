from flask import Flask, render_template, request, jsonify
from chatbot import PlantDiseaseBot
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Cấu hình upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Khởi tạo chatbot
bot = PlantDiseaseBot()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint để dự đoán bệnh từ hình ảnh"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'Không có file hình ảnh'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Chưa chọn file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Định dạng file không được hỗ trợ'}), 400
        
        # Lưu file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Dự đoán
        result = bot.predict_disease(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Lấy lịch sử chẩn đoán"""
    try:
        history = bot.get_detection_history()
        return jsonify({'success': True, 'data': history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """Lấy danh sách bệnh"""
    try:
        diseases = bot.get_all_diseases()
        return jsonify({'success': True, 'data': diseases})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/disease/<disease_name>', methods=['GET'])
def get_disease_info(disease_name):
    """Lấy thông tin chi tiết về một bệnh"""
    try:
        if not bot.diseases_info:
            return jsonify({'success': False, 'error': 'Chưa load dữ liệu bệnh'}), 500
        
        info = bot.diseases_info.get(disease_name)
        if not info:
            return jsonify({'success': False, 'error': 'Không tìm thấy bệnh'}), 404
        
        return jsonify({'success': True, 'data': info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Kiểm tra trạng thái server"""
    return jsonify({'status': 'ok', 'model_loaded': bot.model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)