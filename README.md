# Detection of Food Adulteration Using Machine Learning
This project addresses food safety by leveraging machine learning algorithms to detect food adulteration. The system uses classification models to identify adulterants in food products based on a dataset. It also evaluates the performance of various algorithms such as Random Forest, Logistic Regression, SVM, KNN, and Decision Trees.

![Alt Text](https://github.com/imcalledaditi/Food-Adulteration-Detection/blob/944b42c7c6938d5a4c98bec8e6d5949b7324737d/img.jpg)

## Features
- Utilizes multiple machine learning models, including Random Forest, SVM, Logistic Regression, KNN, and Decision Trees, to detect adulteration.
- Provides predictions on adulteration severity levels (Low, Medium, High).
- Includes data visualization tools for better insight into adulteration trends.
- Offers an interactive interface for real-time analysis.

## Dataset
The dataset used in this project includes:
- **Product Details:** Name, brand, and category.
- **Adulterants:** Type of adulterant detected.
- **Severity Levels:** Low, Medium, and High.
- **Health Risks:** Potential impact on health.
- **Actions Taken:** Measures to address adulteration cases.

## Installation

### Prerequisites
- Python 3.7 or above.
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit.

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Food-Adulteration-Detection-ML.git
2. Navigate to the project directory:
   ```bash
   cd Food-Adulteration-Detection-ML
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
### Machine Learning Models
- Logistic Regression: Effective for binary and multi-class classification problems.
- Support Vector Machine (SVM): Handles high-dimensional data and non-linear boundaries.
- K-Nearest Neighbors (KNN): Simple algorithm based on proximity of data points.
- Random Forest: The ensemble model provides robust predictions with feature importance analysis.
- Decision Tree: Tree-based model for easy interpretability.

### Results
- Best Model: Random Forest achieved the highest accuracy and reliability in detecting adulteration severity.
- Performance Metrics: Models were evaluated using accuracy, precision, recall, and F1 score.
- Visualization: Bar graphs and decision boundary plots were included to illustrate model outputs and data trends.
