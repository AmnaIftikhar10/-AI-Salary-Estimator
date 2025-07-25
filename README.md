🤖 AI Salary Estimator

A machine learning project to predict employee salaries based on job position and experience level. This project was developed as part of an AI course and demonstrates how to build, train, and deploy a regression model using Python, scikit-learn, and Flask.

📌 Features

- Predicts employee salary based on job position
- Trained using regression algorithms
- Flask-based backend for model deployment (`app.py`)
- Interactive development and analysis in Jupyter Notebook
- Clean and modular code (`model.py`)

🧠 Technologies Used

- Python
- scikit-learn
- Flask
- pandas, numpy, matplotlib
- Jupyter Notebook

📂 Files Overview

| File/Folder             | Description                                        |
|-------------------------|----------------------------------------------------|
| `app.py`                | Flask web application to serve predictions         |
| `model.py`              | Contains machine learning model logic              |
| `AI_model.ipynb`        | Jupyter Notebook for EDA, training, and testing    |
| `Position_Salaries.csv` | Dataset used for training and testing              |
| `requirements.txt`      | List of required Python libraries                  |

⚙️ Installation

```bash
git clone https://github.com/yourusername/ai-salary-estimator.git
cd ai-salary-estimator
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
