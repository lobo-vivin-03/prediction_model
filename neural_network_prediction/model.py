from tensorflow.keras.models import load_model
import joblib
import numpy as np
import tensorflow as tf

# Load the trained model
model = load_model('final_exam_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError(), 'mae': tf.keras.metrics.MeanAbsoluteError()})

# Load the scalers
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Example data: [q1, q2, q3, time_spent]
new_data = np.array([[15, 21, 10, 4]])

# Scale the new data
new_data_scaled = scaler_X.transform(new_data)

# Predict the scaled final exam marks
predicted_scaled = model.predict(new_data_scaled)

# Inverse scale the prediction to get the final exam score
predicted_final_exam = scaler_y.inverse_transform(predicted_scaled)

print("Predicted Final Exam Marks:", int(predicted_final_exam[0][0]))
