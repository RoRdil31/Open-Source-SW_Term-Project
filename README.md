# Open-Source-SW_Term-Project

## Introduction

This project utilizes **OpenCV** and the **Caffe framework** to detect faces in real-time camera input and predict the gender and age of detected faces.  
It is designed for students interested in computer vision and deep learning, providing a foundational structure and code for learning and experimentation.

---

## Sources

1. **Link**: [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)  
   **Description**: A repository of various deep learning models for experimentation with the Caffe framework.  
   **Original Author**: BVLC (Berkeley Vision and Learning Center)

2. **Link**: [Age and Gender Classification](https://gist.github.com/GilLevi/c9e99062283c719c03de)  
   **Description**: CNN-based age and gender classification models and resources created by Gil Levi.  
   **Original Author**: Gil Levi

---

## Features

- **Real-time Face Detection**: Detect faces from live camera input using OpenCV.
- **Gender Prediction**: Predict the gender of detected faces using a pre-trained Caffe model.
- **Age Prediction**: Estimate the age of detected faces using a pre-trained Caffe model.

---

## File List

1. **Gender_age_detector.py**  
   - The main execution file for the project.  
   - Key functions:
     - Processes real-time camera input using OpenCV.
     - Detects faces and predicts their gender and age using Caffe models.
     - Displays the results in real-time on the screen.
   - Main libraries used: `OpenCV`, `NumPy`.

2. **age_train_val.prototxt**  
   - Defines the training and validation network structure for the age prediction model in Caffe.  
   - Key contents:
     - Specifies input data, network layers, and output data structure.
     - Includes hyperparameters and settings for training.

3. **deploy_age.prototxt**  
   - Defines the deployment network structure for the age prediction model in Caffe.  
   - Key contents:
     - A lightweight network structure used for testing or deploying the trained model.

4. **gender_train_val.prototxt**  
   - Defines the training and validation network structure for the gender prediction model in Caffe.  
   - Key contents:
     - Similar to the age prediction file but optimized for gender classification.

5. **deploy_gender.prototxt**  
   - Defines the deployment network structure for the gender prediction model in Caffe.  
   - Key contents:
     - A simplified network structure used for testing or deploying the trained gender prediction model.

6. **style.xml**  
   - GUI style configuration file.  
   - Key contents:
     - Specifies visual styles such as bounding box colors and text fonts for displaying results.
