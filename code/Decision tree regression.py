import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df = pd.read_excel('C:\\Users\\Kabir\\jvenv2\\MLprojDSdata2.xlsx')

actual_angles = df['Actual/Real Angle']
reported_angles = df['Reported Angle by cv2.minAreaRect()']
point_A_X = df['Point_A_X']
point_A_Y = df['Point_A_Y']
point_B_X = df['Point_B_X']
point_B_Y = df['Point_B_Y']
point_C_X = df['Point_C_X']
point_C_Y = df['Point_C_Y']
point_D_X = df['Point_D_X']
point_D_Y = df['Point_D_Y']


X = np.column_stack((reported_angles, point_A_X, point_A_Y, point_B_X, point_B_Y, point_C_X, point_C_Y, point_D_X, point_D_Y))
y = actual_angles  
validation_size = 0.2
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, random_state=42)

from sklearn.model_selection import GridSearchCV


model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
# Create a grid search object
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best parameter values and the corresponding mean cross-validated score
best_max_depth = grid_search.best_params_['max_depth']
best_score = grid_search.best_score_
print("Best maximum depth:", best_max_depth)
print("Best negative mean squared error:", best_score)
print(model)

mae_test = mean_absolute_error(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)

print("Mean Absolute Error (MAE) on Test Set:", mae_test)
print("Mean Squared Error (MSE) on Test Set:", mse_test)
print("Root Mean Squared Error (RMSE) on Test Set:", rmse_test)
print("R-squared (R2) Score on Test Set:", r2_test)




# Visualize predictions
plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual Output Angle")
plt.ylabel("Predicted Output Angle")
plt.title("Actual vs Predicted Output Angles (Testing Set)")
plt.savefig("actual_vs_predicted_angles.png")  # Save the figure with a specific name
plt.show()

# Decision tree visualization
plt.figure(figsize=(15,10))
plot_tree(model, filled=True, feature_names=df.columns[:-1])  # Assuming the last column is the target variable
plt.savefig("decision_tree.png")  # Save the figure with a specific name
plt.show()

from sklearn.tree import export_text

input = [[-23,0,0,5.151243682,9.922937495,-11.02909621,-1.832767551,-5.151243682,-9.922937495]]
output = model.predict(input)
print(output)