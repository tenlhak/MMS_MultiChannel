# Multi-Channel 1D CNN for Rotatory Machine Fault Classification. 
This repository contains code for training and evaluating a multi-channel 1D convolutional neural network (CNN) to classify time series data into multiple classes. The project is organized for clarity and ease of use, making it suitable for both beginners and experienced practitioners in deep learning and time series analysis.

## Project Overview
Mechanical fault classification of rotatory machines by vibration analysis is critical in various domains such as petrochemical, energy, automobile, and engineering. This project implements a multi-channel 1D CNN using PyTorch to classify time series vibration signals into four classes:
* Normal condition
* Misaligned condition
* Imbalanced condition
* Bearing Fault condition

The model is trained on multi-channel signal data, leveraging convolutional layers to capture temporal patterns and relationships across different channels.

## Features
* Modular Codebase: Organized into separate modules for data preparation, model definition, training, testing, and visualization.
* Reproducibility: Includes utilities to set random seeds for consistent results.
* Customizable: Easily adjust model architecture, hyperparameters, and data paths.
* Visualization: Generates plots for loss, accuracy, and per-class prediction results.
* Per-Class Analysis: Provides detailed performance metrics for each class.
## Project Structure 
```
MMS_MultiChannel/
├── data/                   
├── models/
│   └── multichannel_cnn.py 
├── scripts/
│   ├── data_preparation.py 
│   ├── train.py            
│   ├── test.py             
│   └── plot_results.py     
├── utils/
│   ├── seed.py             
│   └── helpers.py          
├── plots/                  
├── main.py                 
├── requirements.txt        
└── README.md               
```
## Getting Started
* Prerequisites
* Python 3.6+
* PyTorch 1.7+
* NumPy
* Matplotlib
## Installation
1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/MMS_MultiChannel.git
cd MMS_MultiChannel
```
2. **Set Up a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. **Install Dependencies**
```bash
pip install -r requirements.txt
```
## Data Preparation
* Data Files: The project expects two NumPy array files:
  * X_t.npy: Input data of shape (num_samples, channels, signal_length)
  * y_t.npy: Labels corresponding to the input data
* Data Directory: Place these files inside the data/ directory.
*Update Data Path: If your data directory is different, update the data_dir variable in main.py:

```python
data_dir = "path_to_your_data_directory"
```
# Usage 
Run the main script to train the model and evaluate its performance:
```bash
python main.py
```
This script will:

* Load and preprocess the data
* Split the data into training, validation, and test sets
* Initialize and train the multi-channel 1D CNN model
* Evaluate the model on the test set
* Generate and save plots for loss, accuracy, and per-class predictions

## Results
Upon successful execution, the following outputs will be available:

* **Model Checkpoint:** The best model saved as best_model_multichannel.pth
* **Plots:** Saved in the plots/ directory
  * loss_plot.png: Training and validation loss over epochs
  * accuracy_plot.png: Training and validation accuracy over epochs
  * per_class_predictions.png: Correct vs. incorrect predictions per class on the test set

* Console Output: Displays training progress, validation metrics, and per-class analysis
![ablation_importance](https://github.com/user-attachments/assets/bb003d59-e119-44aa-a56d-7f80661a50a5)
![channel_contributions](https://github.com/user-attachments/assets/55c83ffd-32c9-44b6-9495-844bac59c8c8)
![per_class_predictions](https://github.com/user-attachments/assets/23c21a39-48fc-4587-bc5b-e693cf47b722)


