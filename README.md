#india vs afghanistan winner prediction according to head to head dataset

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange?style=for-the-badge)
![XGBoost](https://img.shields.io/badge/XGBoost-172434?style=for-the-badge&logo=xgboost)
![LightGBM](https://img.shields.io/badge/LightGBM-ff69b4?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-yellow?style=for-the-badge)

A comprehensive, Machine Learning project that predicts the outcomes of India vs Afghanistan ODI matches.

##  Features
- **Match Winner Prediction:** Predicts the winner based on venue, toss, form, and team strength.
- **Winning Probability:** Outputs confidence scores.
- **Top Performers:** Predicts the Highest Run Scorer and Highest Wicket Taker.
- **Expected Score:** Estimates the team scores.
- **Explainable AI (SHAP):** Visualizes exactly *why* a specific team is predicted to win.
- **Interactive UI:** A modern, dark-themed cricket dashboard built with Streamlit.

##  Project Structure
```text
├── app/
│   └── main.py                  # Streamlit Web Application
├── data/
│   └── synthetic_match_data.csv # Generated dataset of 2000 matches
├── models/
│   └── *.pkl                    # Saved ML models, scalers, and encoders
├── src/
│   ├── data_pipeline.py         # Data preprocessing & Feature Engineering
│   └── train_pipeline.py        # Model Training, Evaluation, and SHAP generation
├── generate_synthetic_data.py   # Script to expand small CSV into robust dataset
├── config.yaml                  # Project config
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

##  Machine Learning Pipeline
- **Models Trained:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, and a Soft Voting Ensemble.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- **Feature Engineering:** Venue advantage, toss influence, recent form indexes, H2H record, average scores, and batting/bowling strength indicators.
4. Select the repository, branch, and set the Main file path to `app/main.py`.
5. Click **Deploy!** The app will automatically read `requirements.txt`, install dependencies, and launch.
