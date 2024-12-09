import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import time


data = pd.read_csv("data-cleaned.csv")

categorical = ['fire_position_on_slope']
numerical = ['wind_speed', 'relative_humidity', 'temperature', 'fire_spread_rate', 'current_size', 'assessment_hectares']

for var in categorical:
    data[var] = data[var].fillna("N/A")
    print(f"{var} value counts after filling missing values:")
    print(data[var].value_counts())

for var in numerical:
    data[var] = data[var].fillna(data[var].median())

data['fire_position_on_slope'] = data['fire_position_on_slope'].astype('category').cat.codes

features = ['fire_position_on_slope', 'wind_speed', 'relative_humidity', 'temperature',
            'fire_spread_rate', 'current_size', 'assessment_hectares']
target = 'isNaturalCaused'


x = data[features]
y = data[target]



scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(x_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
gb_model = GradientBoostingClassifier(random_state=42, n_estimators=100)


rf_model.fit(X_train, y_train)

start_time = time.time()
gb_model.fit(X_train, y_train)
end_time = time.time()
print(f"Fitting time: {end_time - start_time}")

rf_predictions = rf_model.predict(X_test)
gb_predictions = gb_model.predict(X_test)


voting_model = VotingClassifier(estimators=[
    ('random_forest', rf_model),
    ('gradient_boosting', gb_model)
], voting='soft')


voting_model.fit(X_train, y_train)


voting_predictions = voting_model.predict(X_test)

def evaluate_model(name, y_test, y_pred):
    print(f"--- {name} ---")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\nTP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")


evaluate_model("Random Forest", y_test, rf_predictions)


evaluate_model("Gradient Boosting", y_test, gb_predictions)


evaluate_model("Voting Classifier", y_test, voting_predictions)
