## Parkinson's Disease Classification

This repository contains code and data for a machine learning project that aims to classify individuals with Parkinson's disease and healthy controls. The project focuses on developing and evaluating different machine learning models for accurate classification.

## Dataset

The dataset used in this project consists of various features extracted from voice recordings of individuals. It includes both individuals diagnosed with Parkinson's disease and healthy controls.

### Explanation of variables

- 'MDVP:Fo(Hz)': Mean fundamental frequency of the voice in Hertz. It represents the average rate at which the vocal folds vibrate during speech production.
- 'MDVP:Fhi(Hz)': Maximum fundamental frequency of the voice in Hertz. It indicates the highest frequency achieved by the vocal folds during speech.
- 'MDVP:Flo(Hz)': Minimum fundamental frequency of the voice in Hertz. It represents the lowest frequency reached by the vocal folds during speech.
- 'MDVP:Jitter(%)': Measures the variation in the vocal fold vibration cycle. It quantifies the frequency perturbation in the fundamental frequency, expressed as a percentage.
- 'MDVP:Jitter(Abs)': Represents the absolute value of the MDVP jitter, which is a measure of the absolute frequency perturbation in the vocal fold vibration cycle.
- 'MDVP:RAP': Represents the relative amplitude perturbation, which measures the variation in the vocal fold vibration amplitude.
- 'MDVP:PPQ': Represents the five-point period perturbation quotient, which is another measure of vocal fold vibration cycle irregularity.
- 'Jitter:DDP': Represents the difference between the absolute differences in consecutive periods of fundamental frequency.
- 'MDVP:Shimmer': Measures the variation in the vocal fold amplitude. It quantifies the cycle-to-cycle variation in amplitude of the vocal folds during sustained vowel phonation.
- 'MDVP:Shimmer(dB)': Represents the shimmer in decibels (dB), which is a logarithmic scale representation of vocal fold amplitude variation.
- 'Shimmer:APQ3': Represents the amplitude perturbation quotient calculated over 3 consecutive periods.
- 'Shimmer:APQ5': Represents the amplitude perturbation quotient calculated over 5 consecutive periods.
- 'MDVP:APQ': Represents the amplitude perturbation quotient, which quantifies the overall amplitude variation in the vocal fold vibration cycle.
- 'Shimmer:DDA': Represents the difference between the absolute differences in consecutive amplitudes of the vocal fold vibration.
- 'NHR': Represents the ratio of noise to harmonic components in the voice. It provides an indication of the degree of hoarseness or breathiness in the voice.
- 'HNR': Harmonic-to-noise ratio, which measures the ratio of energy in the harmonics to the energy in the non-harmonic or noise components of the voice.
- 'status': Binary variable indicating the presence (1) or absence (0) of Parkinson's disease.
- 'RPDE': Recurrence period density entropy, a measure of the complexity and irregularity of the speech signal.
- 'DFA': Detrended fluctuation analysis, which quantifies the fractal-like correlation properties of the voice signal.
- 'spread1', 'spread2', 'D2', 'PPE': Nonlinear dynamic features extracted from the speech signal, which capture various aspects of the vocal fold movement and voice quality.

## Project Structure

- `data/`: Contains the dataset used for raw data and preprocessed data.
- `notebooks/`: Contains Jupyter notebooks with the code for data exploration, preprocessing, model training, and evaluation.
- `models/`: Contains saved trained models.
- `figures/`: Contains visualizations generated during the data analysis.

## Key steps

### Read in data as pandas dataframe

### **Data Exploration and Analysis**

- Checking for missing values to ensure data completeness.
- Assessing the distribution of classes to determine data balance.

### **Data Visualization**

- Histograms to visualize the distribution of each dependent variable for Parkinson's patients and healthy controls, enabling the identification of patterns and differences.
- Box plots to identify extreme values or outliers that could impact model performance.
- Heatmap of pairwise correlations to explore relationships between independent variables.
- Hierarchical clustering to uncover potential clusters within the dataset.

### **Data Preprocessing**

- Outliers were detected and removed using the Interquartile Range (IQR) method.
- Principal Component Analysis (PCA) was applied to address multicollinearity among highly correlated variables. 9 principal components were retained, explaining over 95% of the total variance in the data.

### **Model Selection and Hyperparameter Tuning**

- Stratified splitting of the dataset into training and testing sets to maintain the original data distribution.
- Training different models on the training data.
- Employing grid search to tune hyperparameters and identify the best parameter combination for each model.
- Evaluating model performance using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

### **Models Tested**

- Logistic Linear Regression (PCA-processed data)
- Lasso Regression
- Ridge Regression
- Elastic Net Regression
- K-Nearest Neighbors (KNN)
- Decision Tree and Random Forest
- Support Vector Machine (SVM)
- Linear Discriminant Analysis (LDA)
- Naive Bayes
- Gradient Boosting
- AdaBoost
- Neural Network (NN)

### **Model Evaluation**

The performance of each model was assessed using the testing data. The two models with the highest performance are:

- K-Nearest Neighbors (KNN):
    - Accuracy: 0.967
    - Precision: 0.958
    - Recall: 1.000
    - F1-Score: 0.979
- Support Vector Machine (SVM):
    - Accuracy: 0.967
    - Precision: 1.000
    - Recall: 0.957
    - F1-Score: 0.978

Please refer to the individual code files and notebooks for a detailed implementation of each step.
