# -Deepfake-Detection-using-CNN-CLIP-Hybrid-Architecture-
Deepfake detection system using a hybrid CNN (Xception) and CLIP (ViT-B/16) model with feature fusion, trained on FaceForensics++ for robust real vs fake video classification.
##Overview

This project implements a deepfake detection system using a hybrid architecture:

->Xception (CNN) for spatial features
->CLIP (ViT-B/16) for semantic understanding
->Feature fusion for improved classification

The model processes videos and classifies them as REAL or FAKE.

## Key Features
1.End-to-end pipeline: Video → Faces → Model → Prediction
2.Hybrid CNN + Transformer architecture
3.Transfer learning with partial fine-tuning
4.Frame-level + video-level prediction
5.Evaluated using Accuracy & ROC-AUC


##  Model Architecture
- CNN (Xception): 2048 → 512  
- CLIP (ViT-B/16): semantic features  
- Fusion: `F = F_cnn + F_clip`  
- Final layer → Real / Fake classification  

## Dataset
- **FaceForensics++ (FF++)**
  - `original` → Real  
  - `FaceSwap` → Fake  

Split:
- Train: 70%  
- Validation: 15%  
- Test: 15%  


## Pipeline
1. Video splitting  
2. Face extraction using Dlib (~200 frames per video)  
3. Face refinement (224×224 images)  
4. Data loading with PyTorch  


## Training
- Epochs: 20  
- Batch Size: 32  
- Optimizer: AdamW  
- Scheduler: CosineAnnealingLR  


## Inference
- Sample frames from video  
- Detect faces  
- Predict per frame  
- Aggregate results  
- Output: **REAL / FAKE with confidence**
  
## 📊 Results on deepfake data

### 🔹 Performance Metrics
- **Test Accuracy:** 97.60%  
- **ROC-AUC Score:** 0.9968  
- **F1 Score:** 0.98  


### 🔹 Classification Report
| Class | Precision | Recall | F1-score |
|------|----------|--------|----------|
| Fake |   0.99   |  0.96  |   0.98   |
| Real |   0.96   |  0.99  |   0.98   |

## Installation
git clone https://github.com/your-username/deepfake-detection-cnn-clip.git
cd deepfake-detection-cnn-clip
pip install -r requirements.txt
