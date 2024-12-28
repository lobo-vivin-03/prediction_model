import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import joblib 

# Step 1: Load Data
df = pd.read_csv('dataset.csv')

# Separate features (q1, q2, q3, time_spent) and target (final_exam)
X = df[['Quiz1', 'Quiz2', 'Quiz3', 'HoursSpent']].values
y = df['final_exam'].values

# Step 2: Scale Inputs and Outputs
scaler_X = MinMaxScaler()  # Scaler for features
scaler_y = MinMaxScaler()  # Scaler for target (final exam marks)

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Step 3: Build the Model
model = Sequential([
    Dense(8, activation='relu', input_shape=(X.shape[1],)),  # Input Layer
    Dense(4, activation='relu'),  # Hidden Layer
    Dense(1, activation='linear')  # Output Layer
])

# Compile the Model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Step 4: Train the Model
model.fit(X_scaled, y_scaled, epochs=200, batch_size=2, verbose=1)

# Step 5: Save the trained model
model.save('updated_model/final_exam_model.h5')
print("Model saved as 'final_exam_model.h5'")

joblib.dump(scaler_X, 'updated_model/scaler_X.pkl')
joblib.dump(scaler_y, 'updated_model/scaler_y.pkl')

# Step 6: Predict Final Exam Marks
new_quiz_scores = np.array([[10, 14, 13, 4.6]])  # New input: q1, q2, q3, time_spent
new_quiz_scaled = scaler_X.transform(new_quiz_scores)

# Predict using the trained model
predicted_scaled = model.predict(new_quiz_scaled)
predicted_final_score = scaler_y.inverse_transform(predicted_scaled)

print("Predicted Final Exam Marks:", round(predicted_final_score[0][0], 2))
