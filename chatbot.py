import json
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import google.generativeai as genai
import re
import base64

class PlantDiseaseBot:
    def __init__(self):
        self.model = None
        self.class_indices = None
        self.diseases_info = None
        self.history_file = 'detection_history.json'
        self.confidence_threshold = 0.80  # Nguong 80%
        
        self.gemini_api_key = 'AIzaSyCGiZ1mXraQwSM5dJRr5t2l5Jctt43bAU0'
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            print("Gemini 2.0 Flash API da duoc cau hinh thanh cong")
        except Exception as e:
            self.gemini_model = None
            print(f"Loi cau hinh Gemini: {e}")
        
        self.load_model_and_data()
        self.init_history()
    
    def load_model_and_data(self):
        """Tai mo hinh va du lieu benh"""
        try:
            self.model = load_model('plant_model.h5')
            print("Mo hinh da duoc tai thanh cong")
        except Exception as e:
            print(f"Loi khi tai mo hinh: {e}")
            return False
        
        try:
            with open('class_indices.json', 'r', encoding='utf-8') as f:
                indices = json.load(f)
                self.class_indices = {v: k for k, v in indices.items()}
            print("Class indices da duoc tai")
            print(f"[DEBUG] Class indices: {list(self.class_indices.values())[:5]}...")  # In 5 class đầu
        except Exception as e:
            print(f"Loi khi tai class indices: {e}")
            return False
        
        try:
            with open('diseases_info.json', 'r', encoding='utf-8') as f:
                self.diseases_info = json.load(f)
            print("Thong tin benh da duoc tai")
            print(f"[DEBUG] Diseases info keys: {list(self.diseases_info.keys())[:5]}...")  # In 5 key đầu
            print(f"[DEBUG] Tong so benh: {len(self.diseases_info)}")
        except Exception as e:
            print(f"Loi khi tai diseases info: {e}")
            return False
        
        return True
    
    def init_history(self):
        """Khoi tao file lich su neu chua co"""
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    
    def predict_with_local_model(self, image_path):
        """Du doan benh bang mo hinh CNN cuc bo"""
        try:
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            disease_name = self.class_indices[predicted_class_idx]
            
            return {
                'success': True,
                'disease': disease_name,
                'confidence': confidence,
                'source': 'local_model'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'source': 'local_model'
            }
    
    def clean_gemini_response(self, text):
        """Loai bo dinh dang markdown va icon tu response cua Gemini"""
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)
        text = re.sub(r'[\U0001F300-\U0001F5FF]', '', text)
        text = re.sub(r'[\U0001F680-\U0001F6FF]', '', text)
        text = re.sub(r'[\U0001F1E0-\U0001F1FF]', '', text)
        text = re.sub(r'[\U00002600-\U000027BF]', '', text)
        text = re.sub(r'[\U0001F900-\U0001F9FF]', '', text)
        text = re.sub(r'[•◦▪▫–—]', '', text)
        
        text = re.sub(r'\s+', ' ', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        return text.strip()
    
    def predict_with_gemini(self, image_path):
        """Du doan benh bang Gemini API"""
        if not self.gemini_model:
            return {
                'success': False,
                'error': 'Gemini API chua duoc cau hinh',
                'source': 'gemini'
            }
        
        try:
            print("[Gemini] Dang doc va gui file...")
            
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            image_base64 = base64.standard_b64encode(image_data).decode('utf-8')
            
            if image_path.lower().endswith('.png'):
                mime_type = 'image/png'
            elif image_path.lower().endswith(('.jpg', '.jpeg')):
                mime_type = 'image/jpeg'
            elif image_path.lower().endswith('.gif'):
                mime_type = 'image/gif'
            elif image_path.lower().endswith('.webp'):
                mime_type = 'image/webp'
            else:
                mime_type = 'image/jpeg'
            
            print(f"[Gemini] Gui file: {image_path} (MIME: {mime_type})")
            
            prompt = """Ban la chuyen gia chan doan benh thuc vat. 
Phan tich hinh anh la cay nay va tra loi KHONG CO DINH DANG MARKDOWN, KHONG CO DAU #, KHONG CO DAU **, KHONG CO ICON.

Tra loi theo cu phap sau (van ban thuan tuy):

Ten benh: [Ten benh bang tieng Viet]

Do tin cay: [So phan tram tu 0-100]%

Dau hieu nhan biet:
- [Dau hieu 1]
- [Dau hieu 2]
- [Dau hieu 3]

Cach dieu tri:
- [Cach 1]
- [Cach 2]
- [Cach 3]

Cach phong ngua:
- [Cach 1]
- [Cach 2]
- [Cach 3]

LUU Y: KHONG dung dau #, dau **, dau ***, icon hay bat ky dinh dang markdown nao."""
            
            print("[Gemini] Dang goi API...")
            
            response = self.gemini_model.generate_content(
                [
                    {
                        "mime_type": mime_type,
                        "data": image_base64,
                    },
                    prompt
                ],
                request_options={"timeout": 30}
            )
            
            print("[Gemini] Nhan duoc response")
            
            if not response or not response.text:
                print("[Gemini] Response rong!")
                return {
                    'success': False,
                    'error': 'Gemini tra ve response rong',
                    'source': 'gemini'
                }
            
            response_text = self.clean_gemini_response(response.text)
            print(f"[Gemini] Response text: {response_text[:200]}...")
            
            result = {
                'success': True,
                'disease': 'Khong xac dinh',
                'confidence': 0.5,
                'description': response_text,
                'symptoms': [],
                'treatment': [],
                'prevention': [],
                'source': 'gemini'
            }
            
            disease_match = re.search(r'Ten benh[:\s]+(.+?)(?:\n|Do tin cay)', response_text, re.IGNORECASE)
            if disease_match:
                result['disease'] = disease_match.group(1).strip()
                print(f"[Gemini] Parsed disease: {result['disease']}")
            
            conf_match = re.search(r'Do tin cay[:\s]+(\d+)', response_text, re.IGNORECASE)
            if conf_match:
                result['confidence'] = float(conf_match.group(1)) / 100.0
                print(f"[Gemini] Parsed confidence: {result['confidence']*100}%")
            
            symptoms_match = re.search(r'Dau hieu.*?:(.*?)(?=Cach dieu tri|Cach phong ngua|$)', response_text, re.IGNORECASE | re.DOTALL)
            if symptoms_match:
                symptoms_text = symptoms_match.group(1)
                result['symptoms'] = [s.strip('- ').strip() for s in symptoms_text.split('\n') if s.strip() and s.strip().startswith('-')]
           
            treatment_match = re.search(r'Cach dieu tri.*?:(.*?)(?=Cach phong ngua|$)', response_text, re.IGNORECASE | re.DOTALL)
            if treatment_match:
                treatment_text = treatment_match.group(1)
                result['treatment'] = [t.strip('- ').strip() for t in treatment_text.split('\n') if t.strip() and t.strip().startswith('-')]
            
            prevention_match = re.search(r'Cach phong ngua.*?:(.*?)$', response_text, re.IGNORECASE | re.DOTALL)
            if prevention_match:
                prevention_text = prevention_match.group(1)
                result['prevention'] = [p.strip('- ').strip() for p in prevention_text.split('\n') if p.strip() and p.strip().startswith('-')]
            
            print(f"[Gemini] Parse thanh cong!")
            return result
        
        except Exception as e:
            print(f"[Gemini] Loi: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Loi Gemini API: {str(e)}',
                'source': 'gemini'
            }
    
    def predict_disease(self, image_path):
        """
        Du doan benh - Logic chinh:
        1. Du doan bang local model
        2. Neu confidence >= 80%: Lay thong tin tu JSON
        3. Neu confidence < 80%: Chuyen sang Gemini API
        """
        try:
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': 'Khong tim thay file hinh anh'
                }
            
            print(f"\n[1] Dang phan tich voi mo hinh CNN...")
            local_result = self.predict_with_local_model(image_path)
            
            if not local_result['success']:
                print(f"Loi mo hinh cuc bo: {local_result.get('error')}")
                print("[2] Chuyen sang Gemini API...")
                return self.predict_with_gemini(image_path)
            
            confidence = local_result['confidence']
            disease_name = local_result['disease']
            
            print(f"Ket qua: {disease_name} (Confidence: {confidence*100:.2f}%)")
            
            if confidence >= self.confidence_threshold:
                print(f"[OK] Confidence >= 80% - Lay thong tin tu database cho: {disease_name}")
                
                # DEBUG: Kiểm tra xem disease_name có trong diseases_info không
                print(f"[DEBUG] Dang tim '{disease_name}' trong diseases_info...")
                print(f"[DEBUG] Co trong diseases_info? {disease_name in self.diseases_info}")
                
                disease_info = self.diseases_info.get(disease_name, {})
                
                # DEBUG: In ra thông tin lấy được
                print(f"[DEBUG] disease_info keys: {list(disease_info.keys()) if disease_info else 'EMPTY'}")
                if disease_info:
                    print(f"[DEBUG] description length: {len(disease_info.get('description', ''))}")
                    print(f"[DEBUG] symptoms count: {len(disease_info.get('symptoms', []))}")
                    print(f"[DEBUG] treatment count: {len(disease_info.get('treatment', []))}")
                    print(f"[DEBUG] prevention count: {len(disease_info.get('prevention', []))}")
                
                result = {
                    'success': True,
                    'disease': disease_name,
                    'confidence': confidence,
                    'description': disease_info.get('description', ''),
                    'symptoms': disease_info.get('symptoms', []),
                    'treatment': disease_info.get('treatment', []),
                    'prevention': disease_info.get('prevention', []),
                    'source': 'local_model'
                }
                
                # DEBUG: In ra result trước khi return
                print(f"[DEBUG] Result description length: {len(result['description'])}")
                print(f"[DEBUG] Result symptoms count: {len(result['symptoms'])}")
                
                self.save_detection(disease_name, confidence, image_path, 'local_model')
                
                return result
            
            else:
                print(f"[!] Confidence < 80% ({confidence*100:.2f}%) - Chuyen sang Gemini API...")
                gemini_result = self.predict_with_gemini(image_path)
                
                if gemini_result['success']:
                    self.save_detection(
                        gemini_result['disease'], 
                        gemini_result['confidence'], 
                        image_path, 
                        'gemini'
                    )
                    return gemini_result
                else:
                    print("[!] Gemini that bai - Su dung ket qua local model")
                    disease_info = self.diseases_info.get(disease_name, {})
                    
                    result = {
                        'success': True,
                        'disease': disease_name,
                        'confidence': confidence,
                        'description': disease_info.get('description', ''),
                        'symptoms': disease_info.get('symptoms', []),
                        'treatment': disease_info.get('treatment', []),
                        'prevention': disease_info.get('prevention', []),
                        'source': 'local_model_fallback',
                        'warning': 'Do tin cay thap, ket qua co the khong chinh xac'
                    }
                    
                    self.save_detection(disease_name, confidence, image_path, 'local_model_fallback')
                    return result
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_detection(self, disease, confidence, image_path, source='local_model'):
        """Luu ket qua chan doan vao lich su"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            record = {
                'timestamp': datetime.now().isoformat(),
                'disease': disease,
                'confidence': confidence if isinstance(confidence, float) else confidence,
                'image_path': image_path,
                'source': source
            }
            
            history.append(record)
            
            if len(history) > 100:
                history = history[-100:]
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            print(f"Loi khi luu lich su: {e}")
    
    def get_detection_history(self, limit=20):
        """Lay lich su chan doan"""
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                return history[-limit:]
        except:
            return []
    
    def get_all_diseases(self):
        """Lay danh sach tat ca benh"""
        return list(self.diseases_info.keys())
    
    def get_disease_info(self, disease_name):
        """Lay thong tin chi tiet ve mot benh"""
        return self.diseases_info.get(disease_name, None)
    
    def get_statistics(self):
        """Thong ke lich su chan doan"""
        try:
            history = self.get_detection_history(limit=1000)
            
            if not history:
                return {
                    'total': 0,
                    'by_source': {},
                    'by_disease': {},
                    'avg_confidence': 0
                }
            
            by_source = {}
            by_disease = {}
            total_confidence = 0
            
            for record in history:
                source = record.get('source', 'unknown')
                by_source[source] = by_source.get(source, 0) + 1
                
                disease = record.get('disease', 'unknown')
                by_disease[disease] = by_disease.get(disease, 0) + 1
               
                conf = record.get('confidence', 0)
                if isinstance(conf, float):
                    total_confidence += conf
                else:
                    total_confidence += float(conf)
            
            return {
                'total': len(history),
                'by_source': by_source,
                'by_disease': by_disease,
                'avg_confidence': total_confidence / len(history) if history else 0
            }
        
        except Exception as e:
            print(f"Loi khi lay thong ke: {e}")
            return {
                'total': 0,
                'by_source': {},
                'by_disease': {},
                'avg_confidence': 0
            }