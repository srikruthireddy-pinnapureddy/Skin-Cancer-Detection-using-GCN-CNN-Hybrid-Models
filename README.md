Skin Cancer Detection using GCN-CNN Hybrid Models
Overview
This repository presents a deep learning approach for skin cancer detection employing a hybrid model integrating Graph Convolutional Networks (GCN) and Convolutional Neural Networks (CNN). The objective is to improve classification accuracy and generalizability for clinical diagnosis by leveraging both local and structural features of dermatoscopic skin lesion images.[2][3][4][1]
Features
•	Hybrid GCN-CNN architecture for enhanced skin lesion classification.
•	Model designed to distinguish between benign and malignant skin lesions.
•	Utilizes publicly available dermatoscopic image datasets (e.g., HAM10000).
•	Includes preprocessing, augmentation, training, and evaluation scripts.
•	Detailed performance analysis across multiple metrics.
Project Structure
Skin-Cancer-Detection-using-GCN-CNN-Hybrid-Models/
│
├── data/                   # Dataset storage (add instructions for download)
├── models/                 # Model definitions for GCN, CNN, and hybrid
├── preprocessing/          # Data preprocessing and augmentation scripts
├── train.py                # Main training code
├── evaluate.py             # Evaluation and metrics calculation
├── utils/                  # Utility functions, visualization, etc.
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation (this file)

Requirements
•	Python 3.10+
•	TensorFlow or PyTorch (specify version in requirements.txt)
•	NumPy, Pandas, Matplotlib, scikit-learn
•	Additional dependencies as listed in requirements.txt
Install all dependencies via pip:
pip install -r requirements.txt

Dataset
The model is evaluated primarily on the HAM10000 dataset containing over 10,000 dermatoscopic images labeled as benign or malignant. Please download the dataset from the official source (ISIC Archive) and place image files in the /data folder.[3][2]
Model Architecture
This approach integrates CNN layers for feature extraction with GCN layers to capture relational structures among lesion regions or superpixels. The combined feature map is then used for classification using fully connected layers.
•	CNN Block: Local feature extraction.
•	GCN Block: Graph-based global context modeling.
•	Fusion Layer: Integration of spatial and structural features.
•	Classifier Head: Produces benign/malignant prediction.
Training
Train the hybrid model by running:
python train.py

Key arguments and hyperparameters are configurable within the script. Training logs and checkpoints are automatically saved.
Evaluation
To evaluate on the test set and report metrics (accuracy, precision, recall, F1-score):
python evaluate.py

Results are stored in /results with visualizations of ROC curves and confusion matrices.
Results
•	Test accuracy: (insert results after running experiments)
•	Precision, Recall, F1-score: (update with your metrics)
•	Comparative analysis with standalone CNN and GCN models.
Applications
•	Early and accurate detection of skin cancer from medical images.[4][5][1][2][3]
•	Supports clinical decision-making with automated diagnosis tools.
 
