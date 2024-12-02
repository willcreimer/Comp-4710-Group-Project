import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import time

start_time = time.time()


data = pd.read_csv("data-cleaned.csv")

categorical = ['fire_position_on_slope']  
numerical = ['wind_speed', 'relative_humidity', 'temperature', 'fire_spread_rate', 'current_size', 'assessment_hectares']  


categorical_start_time = time.time()
for var in categorical:
    data[var] = data[var].fillna("N/A")
    print(f"{var} value counts after filling missing values:")
    print(data[var].value_counts())
categorical_end_time = time.time()
categorical_fill_time = categorical_end_time - categorical_start_time
print(f"\nTime to fill categorical missing values: {categorical_fill_time:.2f} seconds")


numerical_start_time = time.time()
for var in numerical:
    data[var] = data[var].fillna(data[var].median())
numerical_end_time = time.time()
numerical_fill_time = numerical_end_time - numerical_start_time
print(f"\nTime to fill numerical missing values: {numerical_fill_time:.2f} seconds")


print("\nNull values after cleanup:")
print(data.isnull().sum())


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

fit_start_time = time.time()
xgb_model.fit(X_train_smote, y_train_smote)
fit_end_time = time.time()

fit_time = fit_end_time - fit_start_time
print(f"\nModel Fitting Time: {fit_time:.2f} seconds")


y_pred = xgb_model.predict(X_test)


conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

print("Confusion Matrix:")
print(conf_matrix)
print(f"\nRaw Rates:\n - True Positives (TP): {tp}\n - True Negatives (TN): {tn}\n - False Positives (FP): {fp}\n - False Negatives (FN): {fn}\n")


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

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal Execution Time: {execution_time:.2f} seconds")
