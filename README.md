
# GNR 638 Assignment 3

## Overview
This repository contains the implementation of Assignment 3 for GNR 638. The notebook processes and analyzes the UCMerced_LandUse dataset, performing data extraction, visualization, preprocessing, and classification using deep learning models.

## Dataset
The UCMerced_LandUse dataset consists of high-resolution aerial images categorized into 21 different land-use classes, such as agricultural, buildings, dense residential, medium residential, parking lots, and more. The dataset is extracted and organized into labeled folders for efficient analysis and classification.

### Dataset Structure
- **Images/**: Contains subfolders for each land-use category.
- **Readme.txt**: Provides information about the dataset and labeling structure.

Each image in the dataset is a 256x256 TIFF file.

## Features Implemented
- **Dataset Extraction**: Automatically extracts the dataset and ensures proper directory structure.
- **Data Preprocessing**:
  - Image resizing and normalization.
  - Conversion of images to grayscale (if required for specific models).
  - Splitting into training, validation, and test sets (70-20-10 split).
- **Data Visualization**:
  - Displays random samples from each class.
  - Shows pixel intensity histograms to analyze image distributions.
- **Model Implementation**:
  - Uses a Convolutional Neural Network (CNN) for land-use classification.
  - Extracts features using a pre-trained ResNet-50 model.
  - Fine-tunes ResNet-50 for classification.
- **Training and Evaluation**:
  - Trains the model using the training set.
  - Validates performance using the validation set.
  - Evaluates accuracy and loss trends over epochs.
  - Plots confusion matrices and classification reports for analysis.

## Usage
1. Clone the repository:
2. Navigate to the project directory and open the Jupyter Notebook (`GNR_638_Assignment3.ipynb`).
3. Ensure that the dataset is correctly extracted and available at `/content/UCMerced_LandUse`.
4. Run all cells to execute the entire workflow, including data preprocessing, training, and evaluation.

## Dependencies
- Python 3
- TensorFlow
- Keras
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- Google Colab (for execution)

## Outputs
- **Dataset Extraction:** The dataset is extracted and verified.
- **File Listing:** Displays extracted files, ensuring dataset integrity.
- **Sample Images:** Shows representative images from different classes.
- **Model Training:** Prints training progress, loss, and accuracy per epoch.
- **Evaluation Metrics:**
  - Final accuracy on the test set.
  - Confusion matrix visualization.
  - Precision, recall, and F1-score for each class.

## Results
- The CNN model achieves a high classification accuracy on the test dataset.
- Class-wise performance is analyzed using confusion matrices and classification reports.

## Acknowledgments
- UCMerced_LandUse dataset
- Google Colab for execution
- TensorFlow and Keras for deep learning models

