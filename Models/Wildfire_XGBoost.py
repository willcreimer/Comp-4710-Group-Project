import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load the data
data = pd.read_csv("data-cleaned.csv")


data['fire_position_on_slope'] = data['fire_position_on_slope'].astype('category').cat.codes


features = ['fire_position_on_slope', 'wind_speed', 'relative_humidity', 'temperature',
            'fire_spread_rate', 'current_size', 'assessment_hectares']
target = 'isNaturalCaused'


data = data.dropna(subset=features + [target])


X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_smote, y_train_smote)

# Predictions
y_pred = xgb_model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

print("Confusion Matrix:")
print(conf_matrix)
print(f"\nRaw Rates:\n - True Positives (TP): {tp}\n - True Negatives (TN): {tn}\n - False Positives (FP): {fp}\n - False Negatives (FN): {fn}\n")

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}\n")

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)


precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1_score}")
