from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import re
import traceback

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model (update path to your model)
MODEL_PATH = 'best.pt'  # Change this to your model path
model = None

# Egyptian currency values mapping
CURRENCY_VALUES = {
    '5 Pounds(front)': 5, '5 Pounds(back)': 5,
    '10 Pounds(front)': 10, '10 Pounds(back)': 10,
    '20 Pounds(front)': 20, '20 Pounds(back)': 20,
    '50 Pounds(front)': 50, '50 Pounds(back)': 50,
    '100 Pounds(front)': 100, '100 Pounds(back)': 100,
    '200 Pounds(front)': 200, '200 Pounds(back)': 200,
}

def load_model():
    """Load the YOLO model"""
    global model
    if model is None:
        try:
            model = YOLO(MODEL_PATH)
            print(f"✅ Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    return model

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_denomination(class_name):
    """Extract denomination value from class name"""
    # Handle different formats: "5 Pounds(front)", "10 Pounds(back)", etc.
    for key, value in CURRENCY_VALUES.items():
        if key.lower() in class_name.lower():
            return value
    
    # Fallback: extract number from string
    numbers = re.findall(r'\d+', class_name)
    if numbers:
        return int(numbers[0])
    return 0

def detect_and_calculate(image_path):
    """
    Run detection on image and calculate total amount
    Returns: (annotated_image_path, detections, total_amount)
    """
    model = load_model()
    
    # Read image for annotation
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
        
    # Ensure image is contiguous array
    img = np.ascontiguousarray(img)
    
    # Apply blur preprocessing to improve detection on high-quality images
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Run inference
    results = model.predict(
        source=img,
        conf=0.25,  # Confidence threshold
        save=False,
        verbose=False
    )
    
    # Process results
    detections = []
    total_amount = 0
    denomination_counts = {}
    
    for result in results:
        # Convert boxes to numpy on CPU once
        # This returns a Boxes object where .xyxy, .conf, .cls are numpy arrays
        boxes = result.boxes.cpu().numpy()
        
        if boxes is None or len(boxes) == 0:
            continue
            
        # Iterate using index to access numpy arrays directly
        for i in range(len(boxes)):
            try:
                # Extract data using direct indexing
                # boxes.xyxy is (N, 4)
                box_coords = boxes.xyxy[i]
                x1 = int(box_coords[0])
                y1 = int(box_coords[1])
                x2 = int(box_coords[2])
                y2 = int(box_coords[3])
                
                # boxes.conf is (N,)
                confidence = float(boxes.conf[i])
                
                # boxes.cls is (N,)
                class_id = int(boxes.cls[i])
                
                class_name = model.names[class_id]
                
                # Extract denomination
                value = extract_denomination(class_name)
                total_amount += value
                
                # Count denominations
                if value not in denomination_counts:
                    denomination_counts[value] = 0
                denomination_counts[value] += 1
                
                # Draw bounding box on image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Add label
                label = f"{value} EGP ({confidence:.2f})"
                cv2.putText(img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                detections.append({
                    'class': str(class_name),
                    'confidence': confidence,
                    'value': value,
                    'bbox': [x1, y1, x2, y2]
                })
                
            except Exception as e:
                print(f"Error processing box {i}: {e}")
                traceback.print_exc()
                continue
    
    # Save annotated image
    filename = os.path.basename(image_path)
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{filename}')
    cv2.imwrite(result_path, img)
    
    return result_path, detections, total_amount, denomination_counts

@app.route('/')
def index():
    """Home page with upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and run detection"""
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Run detection
            result_path, detections, total_amount, denomination_counts = detect_and_calculate(filepath)
            
            # Prepare result data
            result_data = {
                'original_image': filename,
                'result_image': os.path.basename(result_path),
                'detections': detections,
                'total_amount': total_amount,
                'num_notes': len(detections),
                'denomination_counts': denomination_counts
            }
            
            return render_template('result.html', **result_data)
        
        except Exception as e:
            print("!!! ERROR IN UPLOAD_FILE !!!")
            traceback.print_exc()
            flash(f'Error processing image: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload PNG, JPG, or JPEG', 'error')
    return redirect(url_for('index'))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    # Load model on startup
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
