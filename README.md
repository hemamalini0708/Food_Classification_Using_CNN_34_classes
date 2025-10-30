# Food_Classification_Using_CNN_34_classes

![WhatsApp Image 2025-10-30 at 15 54 05_ff270f0a](https://github.com/user-attachments/assets/d7061164-8c04-4f4a-a6cb-1fc9c67ea213)

# Project Overview
A complete web-based food classification system that can identify 34 different food items from images and display detailed nutritional information. This project combines deep learning with web technologies to create a practical tool for food recognition and health analysis.

### **Features**
Multi-Model Support: Choose between three trained models (Custom CNN, VGG16, ResNet)

Nutritional Analysis: Get detailed nutrition facts for identified food items

User-Friendly Interface: Clean, professional web interface with image upload

Performance Metrics: View model accuracy and confidence scores

Real-time Predictions: Instant classification results

### **Technical Stack**
**Backend Technologies**
Python Flask - Web framework for API endpoints

TensorFlow/Keras - Deep learning model training and inference

OpenCV/PIL - Image processing and preprocessing

NumPy - Numerical computations for image data

**Frontend Technologies**
HTML5 - Semantic structure with modern elements

CSS3 - Professional styling with gradients and responsive design

JavaScript - Client-side interactivity and API communication

Font Awesome - Icons for enhanced user interface

### **Models Implemented**

<img width="867" height="123" alt="image" src="https://github.com/user-attachments/assets/ed8725aa-037d-4621-bd21-55143b0c2db0" />


<img width="500" height="421" alt="image" src="https://github.com/user-attachments/assets/0461b66c-c157-4387-a91a-3306eff6ecd6" />







**1. Custom CNN Model**
Built from scratch with multiple convolutional layers

Trained specifically for food image classification

Optimized for 34 food classes


<img width="898" height="410" alt="image" src="https://github.com/user-attachments/assets/afd77147-3f33-45f5-b5ea-0e3d002432e0" />





**2. VGG16 Transfer Learning**
Pre-trained VGG16 model fine-tuned on food dataset

Leverages ImageNet weights for better feature extraction

Custom classification head for food categories


<img width="969" height="472" alt="image" src="https://github.com/user-attachments/assets/94b25d5d-81cd-4a41-bec5-bdf0994f6078" />






**3. ResNet Transfer Learning**
ResNet50 architecture with transfer learning

Handles complex food image patterns effectively

Residual connections for improved gradient flow


<img width="929" height="484" alt="image" src="https://github.com/user-attachments/assets/efce695c-2cfd-44f1-81b1-403d5e120f17" />





### **Dataset Information**
Source: Food Image Classification Dataset from Kaggle

Classes: 34 different food items

Samples: Balanced dataset with 200+ images per class

Food Items: Burger, Pizza, Samosa, Dosa, Idli, and 30+ more

### **Installation & Setup**
Prerequisites
Python 3.7+

TensorFlow 2.x

Flask

Required Python packages

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

pip install -r requirements.txt
Run the application:

bash
python app.py
Open browser and navigate to:

text
http://localhost:5000
Project Structure

<img width="884" height="362" alt="image" src="https://github.com/user-attachments/assets/6aac9722-6641-4def-8f16-86532b622a5e" />



### **How to Use**
Upload Image: Click the upload box to select a food image

Choose Model: Select one of the three available models

View Results: See predicted food class, confidence score, and nutrition facts

Analyze Metrics: Check model performance and accuracy metrics

Nutritional Information
The system provides detailed nutritional data for each food item:

Calories

Protein content

Carbohydrates

Fat content

Fiber content

### **Model Performance**
All models were trained for 50 epochs and evaluated on:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix metrics

### **Key Features**

Multiple Model Comparison: Test different architectures on same image

Confidence Scoring: See how confident the model is in its prediction

Nutrition Database: Comprehensive food nutrition information

Responsive Design: Works on desktop and mobile devices

Error Handling: Robust error management for smooth user experience

### **Training Details**
Platform: Kaggle (for reliable GPU access)

Epochs: 50 per model

Image Size: 224x224 pixels

Augmentation: Applied to balance dataset

Validation Split: 10% of training data

### **References**
Flask Documentation: https://flask.palletsprojects.com/

TensorFlow Documentation: https://www.tensorflow.org/

Kaggle Dataset: https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset

VGG16 Paper: https://arxiv.org/abs/1409.1556

ResNet Paper: https://arxiv.org/abs/1512.03385

### **Future Enhancements**

Expand to 100+ food classes

Mobile app development

Calorie estimation from food volume

User profiles and diet tracking

Integration with health apps

### **Developer**
Hema Malini
Data Science Intern
Vihara Tech

### **License**
This project is developed as part of Data Science Internship at Vihara Tech.

