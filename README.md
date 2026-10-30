# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries such as NumPy, Pandas, Matplotlib, and the SGDRegressor model from Scikit-learn.

2. Load and preprocess the dataset by separating the input features (independent variables) and the target variables (house price and number of occupants). Split the dataset into training and testing sets.
   
3. Train the model by creating separate SGDRegressor models for each target variable (house price and number of occupants) and fit them using the training data.

4. Predict and evaluate the results using the testing data, compare the predicted values with the actual values, and calculate performance metrics such as Mean Squared Error (MSE) and R² Score. 


## Program:
```
/*
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Sarmasaran m
RegisterNumber:  212224060239
*/

# Code cell
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#Code cell
data = fetch_california_housing()

# Select first 3 features (for demonstration)
X = data.data[:, :3]   # shape (n_samples, 3)

# Create a multi-output target: [median_house_value, some_other_numeric_column]
# Here we use column index 6 (for demonstration) as the second output
Y = np.column_stack((data.target, data.data[:, 6]))

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Example X (first row):", X[0])
print("Example Y (first row):", Y[0])

# Code cell
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Train shapes:", X_train.shape, Y_train.shape)
print("Test shapes: ", X_test.shape, Y_test.shape)

# Code cell
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

# Fit on training data and transform both train and test
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

print("Scaled X_train mean (approx):", X_train_scaled.mean(axis=0))
print("Scaled Y_train mean (approx):", Y_train_scaled.mean(axis=0))

# Code cell
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)  # you can also set alpha, eta0, penalty etc.
multi_output_sgd = MultiOutputRegressor(sgd)

# Fit on scaled training data
multi_output_sgd.fit(X_train_scaled, Y_train_scaled)

# Code cell
Y_pred_scaled = multi_output_sgd.predict(X_test_scaled)   # predicted in scaled space
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)         # back to original units
Y_test_orig = scaler_Y.inverse_transform(Y_test_scaled)    # ground-truth back to original

print("First 5 predictions (original units):")
print(Y_pred[:5])

# Code cell
mse = mean_squared_error(Y_test_orig, Y_pred)
print("Mean Squared Error (multi-output):", mse)

# Per-output MSE (optional, helpful for debugging)
mse_per_output = np.mean((Y_test_orig - Y_pred) ** 2, axis=0)
print("MSE per output:", mse_per_output)

# Code cell
for i in range(5):
    print(f"Example {i+1}")
    print("Inputs (raw):", X_test[i])
    print("True outputs:", Y_test_orig[i])
    print("Predicted   :", Y_pred[i])
    print("-" * 40)
 
*/
*/
```

## Output:
## 2.Load dataset and select features/targets:

<img width="485" height="70" alt="image" src="https://github.com/user-attachments/assets/6580629c-ce5d-48ab-9d12-d87ce3c419eb" />

## 3.Train/test split:

<img width="263" height="47" alt="image" src="https://github.com/user-attachments/assets/fab5d7e1-ad2c-4cfe-8c4d-e4cf42dadb29" />

## 4.Feature scaling:

<img width="547" height="42" alt="image" src="https://github.com/user-attachments/assets/d60e8e9b-3b47-4905-866b-4eb06f1bdf32" />

## 5.Initialize and train the model:

<img width="217" height="140" alt="image" src="https://github.com/user-attachments/assets/ee06d96d-286c-49f4-b6f0-1b97eda06a37" />

## 6.Predict and inverse-transform:

<img width="296" height="102" alt="image" src="https://github.com/user-attachments/assets/08e7bdd0-e913-4703-932f-923e46246e58" />

## 7.Evaluate model performance:

<img width="371" height="43" alt="image" src="https://github.com/user-attachments/assets/ad7fc3a7-8bd0-48e9-9156-b5ab2faef751" />
## 8.Sample predictions vs true values:

<img width="421" height="372" alt="image" src="https://github.com/user-attachments/assets/806fe231-63bc-4305-a4e3-11c05ba99ca3" />

<img width="287" height="37" alt="image" src="https://github.com/user-attachments/assets/2728f01d-05f9-418a-bf6d-5a02376b3967" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
