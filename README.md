# Loan Prediction Analysis

A Flask-based loan risk prediction system powered by a Gaussian Naive Bayes model. This project demonstrates end-to-end machine learning pipeline implementation with model training, evaluation, and deployment.

**Author:** Kishore SN

---

## Project Overview

This project implements a machine learning solution for loan approval prediction. It includes:

- **Data preprocessing** with outlier removal and categorical encoding
- **Model training** using Gaussian Naive Bayes with threshold tuning
- **Web interface** for making predictions on new loan applications
- **Model evaluation** with comprehensive performance metrics
- **Reusable model artifact** for consistent predictions

---

## Model Performance

The model was trained and evaluated using the included dataset: `loan_data.csv`.

| Metric | Value |
| --- | ---: |
| Original records | 45,000 |
| Records after outlier removal | 37,992 |
| Records removed as outliers | 7,008 |
| Validation accuracy | 87.88% |
| Test accuracy | 87.10% |
| Decision threshold | 0.61 |
| Model type | Gaussian Naive Bayes |

### Confusion Matrix (Test Set)

| Actual / Predicted | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | 5,439 | 493 |
| Actual 1 | 487 | 1,180 |

### Classification Summary

- Class `0` precision: 91.78%
- Class `0` recall: 91.69%
- Class `1` precision: 70.53%
- Class `1` recall: 70.79%
- Weighted F1-score: 87.11%

---

## Features Used

The model uses the following 13 features for prediction:

- Age
- Gender
- Education
- Annual income
- Employment experience
- Home ownership
- Loan amount
- Loan intent
- Loan interest rate
- Loan percent income
- Credit history length
- Credit score
- Previous loan defaults on file

---

## Project Structure

```
.
|-- app.py                          # Flask application
|-- loan_data.csv                   # Training dataset
|-- models/
|   `-- loan_status_new_model.pkl   # Trained model artifact
|-- scripts/
|   `-- train_model.py              # Model training script
|-- static/
|   `-- css/                        # Stylesheet files
|-- templates/
|   `-- index.html                  # Web interface
|-- test_app.py                     # Application tests
|-- requirements.txt                # Python dependencies
`-- README.md                       # This file
```

---

## Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the Application

Start the Flask development server:

```powershell
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000/
```

### Run Application Tests

```powershell
python test_app.py
```

### Production Deployment

For production-style serving, use Waitress:

```powershell
waitress-serve --host 0.0.0.0 --port 5000 app:app
```

---

## Model Training

### Retrain the Model

To retrain the model with new data:

```powershell
python scripts\train_model.py
```

### Training Pipeline

The training script performs the following steps:

1. **Data Loading:** Reads `loan_data.csv`
2. **Data Preprocessing:**
   - Encodes categorical variables
   - Removes outliers using IQR method
3. **Model Training:** Trains Gaussian Naive Bayes classifier
4. **Threshold Tuning:** Optimizes decision threshold for business outcomes
5. **Model Evaluation:** Generates performance metrics
6. **Model Persistence:** Saves trained model as `models/loan_status_new_model.pkl`

---

## How to Use

### Web Interface

1. Fill in the loan application form with borrower and loan details
2. Click "Predict" to get the model's assessment
3. View the prediction result and supporting probability score

### Prediction Output

The application returns:

- **Predicted Status:** Approved or Rejected classification
- **Probability Score:** Confidence level of the prediction
- **Model Metadata:** Decision threshold and model performance metrics

---

## Technical Details

### Model Algorithm

**Gaussian Naive Bayes** - A probabilistic classifier based on Bayes' theorem that assumes feature independence. It is:

- Fast to train and predict
- Computationally efficient
- Suitable for real-time decision support
- Interpretable through probability scores

### Preprocessing

- **Outlier Removal:** IQR (Interquartile Range) method applied to numeric features
- **Categorical Encoding:** One-hot encoding for categorical variables
- **Feature Scaling:** Standardization for numeric features

### Decision Threshold

The model uses a tuned threshold of **0.61** instead of the default 0.50 to optimize for business outcomes and balance precision/recall tradeoffs.

---

## Development & Improvements

### Current Limitations

- Class imbalance affects minority class performance
- Prototype-quality; requires additional hardening for production
- Limited to 13 predefined features
- No audit logging or monitoring

### Recommended Enhancements

- Add ROC-AUC and precision-recall analysis
- Implement model monitoring and drift detection
- Compare with alternative algorithms (Logistic Regression, Random Forest, Gradient Boosting)
- Add database storage for prediction history
- Implement authentication and role-based access control
- Add comprehensive audit logging
- Build analytics dashboard for model performance tracking
- Conduct fairness and bias analysis across demographic groups

---

## Dependencies

Key Python packages required:

- Flask - Web framework
- scikit-learn - Machine learning library
- pandas - Data manipulation
- numpy - Numerical computing
- joblib - Model serialization
- pytest - Testing framework

See `requirements.txt` for complete list and versions.

---

## License

This project is open source. Please see the LICENSE file for details.

---

## Contact

**Author:** Kishore SN  
**Project:** Loan Prediction Analysis

For questions or contributions, please open an issue or submit a pull request.
