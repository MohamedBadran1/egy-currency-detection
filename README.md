# 🇪🇬 Egyptian Currency Detection System

A complete computer vision project for detecting, classifying, and counting Egyptian banknotes using YOLOv8. Includes both a Flask web application and a Jupyter notebook for training and experimentation.

## 📋 Project Overview

This project implements an automated Egyptian currency detection system capable of:
- Detecting all Egyptian denominations (5, 10, 20, 50, 100, 200 EGP)
- Recognizing both front and back sides of banknotes
- Calculating total monetary value in images
- Real-time detection via webcam
- Web-based deployment for easy access

## 🎯 Features

### Web Application
- 📤 **Upload & Detect** - Upload images containing Egyptian banknotes
- 💵 **Multi-Currency Support** - Detects all modern Egyptian denominations
- 🧮 **Auto Calculate** - Automatically calculates total amount
- 📊 **Visual Results** - Shows annotated images with bounding boxes
- 📋 **Detailed Breakdown** - Lists all detected notes with confidence scores
- 🔧 **Preprocessing** - Applies Gaussian blur to handle image quality variations

### Jupyter Notebook
- 📊 **Dataset Analysis** - Complete EDA with visualizations
- 🤖 **Model Training** - YOLOv8s training pipeline
- 📈 **Performance Metrics** - Training curves, confusion matrix
- 📷 **Real-time Camera** - Live webcam detection
- 🧪 **Preprocessing Experiments** - Compare different preprocessing techniques

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam (optional, for real-time detection)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/egyptian-currency-detection.git
cd egyptian-currency-detection

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Web App

```bash
python app.py
```
Then open your browser to `http://localhost:5000`

### Using the Notebook

```bash
jupyter notebook egy_currency_detection.ipynb
```

## 📁 Project Structure

```
egy-currency-detection-project/
├── app.py                          # Flask web application
├── egy_currency_detection.ipynb    # Complete training & analysis notebook
├── best.pt                         # Trained YOLOv8 model weights
├── requirements.txt                # Python dependencies
├── templates/
│   ├── index.html                  # Upload page
│   └── result.html                 # Results display page
├── static/
│   ├── css/
│   │   └── style.css              # Web app styling
│   └── uploads/                    # Uploaded images storage
└── egy_currency_model/
    └── weights/
        ├── best.pt                 # Best model checkpoint
        └── last.pt                 # Last epoch checkpoint
```

## 🧠 Model Performance

The YOLOv8s model was trained on 11,449 images with the following results:

| Metric | Value |
|--------|-------|
| **mAP50** | 99.5% |
| **mAP50-95** | 99.2% |
| **Precision** | 99.88% |
| **Recall** | 100% |

### Training Configuration
- **Model**: YOLOv8s (Small)
- **Epochs**: 30 (with early stopping)
- **Batch Size**: 64
- **Image Size**: 640x640
- **Classes**: 12 (6 denominations × 2 sides)

## 🔧 Technical Details

### Preprocessing Pipeline
To handle variations in image quality and camera focus, the system applies:
- **Gaussian Blur (5×5)** before inference
- Improves detection on low-quality or blurry images
- Simulates real-world camera conditions

### Currency Classes
The model detects 12 distinct classes:
- 5 EGP (Front & Back)
- 10 EGP (Front & Back)
- 20 EGP (Front & Back)
- 50 EGP (Front & Back)
- 100 EGP (Front & Back)
- 200 EGP (Front & Back)

## 📸 Usage Examples

### Web Application
1. Upload an image containing Egyptian banknotes
2. System detects and draws bounding boxes around each note
3. Calculates total value automatically
4. Displays confidence scores for each detection

### Real-time Camera (Notebook)
Run the camera detection cell to:
- Open webcam feed
- Detect currency in real-time
- Display total value overlay
- Press 'q' to quit, 's' to save screenshot

## 🛠️ Development

### Dataset
The model was trained using a curated dataset from Roboflow containing:
- 11,449 training images
- Multiple angles and lighting conditions
- Varied backgrounds (hands, tables, etc.)
- Real-world phone camera photos

### Training the Model
To retrain the model with your own data:
1. Open `egy_currency_detection.ipynb`
2. Follow the training pipeline cells
3. Update the dataset path
4. Run training cells
5. Model weights saved to `runs/detect/`

## 📝 Requirements

Core dependencies:
```
Flask==3.0.0
ultralytics==8.1.0
opencv-python==4.8.1.78
numpy==1.23.5
```

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Created as part of the NTI AI Training Program.

## 🙏 Acknowledgments

- **Ultralytics YOLOv8** for the detection framework
- **Roboflow** for dataset hosting and preprocessing tools
- **OpenCV** for image processing capabilities

---

**Built with ❤️ for Egyptian Currency Detection | NTI AI Training Program 2025**
