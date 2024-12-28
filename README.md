# Heart Disease Prediction Model

This project uses machine learning to predict heart disease based on various medical attributes. The model achieves approximately 88.5% accuracy using Logistic Regression.

## Project Overview

The project aims to predict whether a patient has heart disease based on clinical parameters. We use various Python-based machine learning and data science libraries to build and evaluate our model.

### Dataset Features

- `age`: Age of the patient in years
- `sex`: Gender (1 = male, 0 = female)
- `cp`: Chest pain type
  - 0: Typical angina
  - 1: Atypical angina
  - 2: Non-anginal pain
  - 3: Asymptomatic
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- `restecg`: Resting electrocardiographic results
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise induced angina (1 = yes, 0 = no)
- `oldpeak`: ST depression induced by exercise relative to rest
- `slope`: Slope of the peak exercise ST segment
- `ca`: Number of major vessels colored by fluoroscopy (0-3)
- `thal`: Thalassemia type
- `target`: Heart disease (1 = present, 0 = absent)

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
heart-disease-prediction/
├── data/
│   └── heart-disease.csv
├── notebooks/
│   └── heart_disease_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   └── evaluation.py
├── tests/
│   └── test_model.py
├── README.md
└── requirements.txt
```

## Model Performance

The Logistic Regression model achieves:
- Accuracy: 88.52%
- Precision: 82.16%
- Recall: 92.73%
- F1 Score: 87.05%

## Usage

1. Data preprocessing:
```python
from src.data_preprocessing import preprocess_data
X_train, X_test, y_train, y_test = preprocess_data('data/heart-disease.csv')
```

2. Train and evaluate the model:
```python
from src.model import train_model
from src.evaluation import evaluate_model

model = train_model(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original dataset from the UCI Machine Learning Repository
- Inspired by various heart disease prediction research papers