import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Import SMOTE

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


param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularization strength
    'solver': ['liblinear', 'saga'],  # Solvers for optimization
}


grid_search = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_smote, y_train_smote)


best_params = grid_search.best_params_
print("Best Hyperparameters from GridSearchCV:", best_params)


print("\nGridSearchCV Results:")
results = pd.DataFrame(grid_search.cv_results_)
print(results)


best_model = LogisticRegression(C=best_params['C'], solver=best_params['solver'], max_iter=500)
best_model.fit(X_train_smote, y_train_smote)


y_pred = best_model.predict(X_test)
report_dict = classification_report(y_test, y_pred, output_dict=True)
precision_weighted_avg = report_dict['weighted avg']['precision']
print(f"Precision for the whole model: {precision_weighted_avg}")


accuracy = accuracy_score(y_test, y_pred)
print(f"\nBest Accuracy Score after Tuning: {accuracy}")

report = classification_report(y_test, y_pred)
print("Best Classification Report:")
print(report)


coefficients = best_model.coef_
print(f"Best Model Coefficients: {coefficients}")
