import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import time

start_time = time.time()

data = pd.read_csv("data-cleaned.csv")

categorical = ['fire_position_on_slope']
numerical = ['wind_speed', 'relative_humidity', 'temperature', 'fire_spread_rate', 'current_size', 'assessment_hectares']

for var in categorical:
    data[var] = data[var].fillna("N/A")
    print(f"{var} value counts after filling missing values:")
    print(data[var].value_counts())

for var in numerical:
    data[var] = data[var].fillna(data[var].median())

print("\nNull values after cleanup:")
print(data.isnull().sum())

data['fire_position_on_slope'] = data['fire_position_on_slope'].astype('category').cat.codes

features = ['fire_position_on_slope', 'wind_speed', 'relative_humidity', 'temperature',
            'fire_spread_rate', 'current_size', 'assessment_hectares']
target = 'isNaturalCaused'

x = data[features]
y = data[target]



scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(x_scaled, y)

X_train_smote, X_test, y_train_smote, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print(X_train.shape)
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
}

grid_search = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_smote, y_train_smote)

best_params = grid_search.best_params_
print("Best Hyperparameters from GridSearchCV:", best_params)

print("\nGridSearchCV Results:")
results = pd.DataFrame(grid_search.cv_results_)
print(results)

best_model = LogisticRegression(C=best_params['C'], solver=best_params['solver'], max_iter=500)

fit_start_time = time.time()
best_model.fit(X_train_smote, y_train_smote)
fit_end_time = time.time()

fit_time = fit_end_time - fit_start_time
print(f"\nModel Fitting Time: {fit_time} seconds")

y_pred = best_model.predict(X_test)

report_dict = classification_report(y_test, y_pred, output_dict=True)
precision_weighted_avg = report_dict['weighted avg']['precision']
print(f"Precision for the whole model: {precision_weighted_avg}")

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual Human Caused", "Actual Natural Caused"], columns=["Predicted Human Caused", "Predicted Natural Caused"])
print(cm_df)
print(f"\nBest Accuracy Score after Tuning: {accuracy}")
print("Best Classification Report:")
print(classification_report(y_test, y_pred))

coefficients = best_model.coef_
print(f"Best Model Coefficients: {coefficients}")

end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal Execution Time: {execution_time:.2f} seconds")
