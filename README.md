# 🫁 Chest X-ray Pneumonia Predictor

A Deep Learning-based web application for detecting Pneumonia from Chest X-ray images using a Vision Transformer (ViT) model and Streamlit.
## 📸 Application Screenshot

![Application Screenshot](screenshots/app_outpu1t.png)
---

## Project Overview

This project predicts whether a patient has Pneumonia from a Chest X-ray image using a trained Vision Transformer (ViT) model.

The application provides:
- Uploading Chest X-ray images
- Loading custom trained `.pth` model files
- Pneumonia prediction with confidence score
- Interactive Streamlit web interface

---

##  Technologies Used

- Python
- Streamlit
- PyTorch
- TIMM (Vision Transformer)
- NumPy
- Pillow (PIL)
- Scikit-learn

---

##  Project Structure

```bash
PES-Project/
│
├── app.py                  # Streamlit frontend application
├── inference.py            # Model loading & prediction logic
├── requirements.txt        # Required Python libraries
├── models/
│   └── best_vit.pth        # Trained model weights
│
├── data/
│   └── Chest X-ray images
│
└── README.md

## Model Information

The project uses:
Vision Transformer (ViT Base Patch16 224)
Binary Classification
Output:
Yes → Pneumonia detected
No → Normal

Input image size:
224 x 224

** Installation
Step 1: Clone Repository
git clone <your-github-repo-link>
cd PES-Project

Step 2: Create Virtual Environment (Optional)
python -m venv venv

Activate environment:
Windows
venv\Scripts\activate
Linux / Mac
source venv/bin/activate

Step 3: Install Requirements
pip install -r requirements.txt
▶️ Run the Application
streamlit run app.py

After running successfully, open:
http://localhost:8501

📸 Application Workflow
Step 1
Load trained model:

Upload .pth model file
OR
Provide model path

Step 2
Upload Chest X-ray image:
.jpg
.jpeg
.png

Step 3
Click:
Predict

Step 4
View:
Pneumonia Prediction
Confidence Score
🖼️ Sample Output
Prediction: Yes
Confidence: 94.25%

OR

Prediction: No
Confidence: 12.45%

Dataset:
Chest X-ray image dataset

Images contain:
Normal Chest X-rays
Pneumonia infected Chest X-rays
🔍 Features

✅ Streamlit Web Interface
✅ Upload Custom Model
✅ Real-time Prediction
✅ Confidence Score
✅ GPU/CPU Support
✅ Vision Transformer (ViT) Architecture

🛠️ Requirements
streamlit==1.33.0
numpy==1.24.4
pandas==1.5.3
scikit-learn==1.3.2
pillow==9.5.0
tqdm==4.66.4
timm==0.9.12
