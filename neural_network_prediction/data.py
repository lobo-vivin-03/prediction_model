import numpy as np
import pandas as pd

# Parameters
num_rows = 10000  # Number of rows of data
quiz_score_range = (0, 30)  # Range for quiz scores (0 to 30)

# Generating quiz scores
q1_scores = np.random.randint(quiz_score_range[0], quiz_score_range[1] + 1, size=num_rows)  # +1 to include 30
q2_scores = np.random.randint(quiz_score_range[0], quiz_score_range[1] + 1, size=num_rows)
q3_scores = np.random.randint(quiz_score_range[0], quiz_score_range[1] + 1, size=num_rows)

# Generating time spent on the platform (in hours)
time_spent = np.round(np.random.uniform(1, 10, size=num_rows), 1)  # Round to one decimal place

# Difficulty factor for each quiz
difficulty_factor_q1 = 1.1  # Higher weight on quiz 1 (e.g., harder)
difficulty_factor_q2 = 1.0  # Normal difficulty for quiz 2
difficulty_factor_q3 = 0.9  # Lower weight on quiz 3 (e.g., easier)

# Calculate weighted average of quiz scores
final_exam_scores = (q1_scores * difficulty_factor_q1 + q2_scores * difficulty_factor_q2 + q3_scores * difficulty_factor_q3) / (difficulty_factor_q1 + difficulty_factor_q2 + difficulty_factor_q3)

# Introduce a performance modifier based on time spent on platform
performance_modifier = np.round(time_spent * 0.5, 1)  # Round to one decimal place

# Calculate the final score with the modifier
final_exam_scores_with_modifier = final_exam_scores + performance_modifier

# Add some noise for randomness (optional)
noise = np.round(np.random.normal(0, 2, size=num_rows), 1)  # Round noise to one decimal place
final_exam_scores_with_noise = final_exam_scores_with_modifier + noise

# Ensure the score is within 0-30 range and round it to one decimal
final_exam_scores_with_noise = np.clip(np.round(final_exam_scores_with_noise, 1), 0, 30)

# Update the final exam score to be out of 100 and apply error if score < 75
final_exam_scores_out_of_100 = final_exam_scores_with_noise * (100 / 30)  # Scale score to out of 100

# Apply 10% error if final exam score is below 75
final_exam_scores_updated = []
for score in final_exam_scores_out_of_100:
    if score < 60:
        variation = np.random.uniform(-0.05, 0.05)  # Random variation between -5% and +5%
        updated_score = score + (score * variation)
        final_exam_scores_updated.append(round(updated_score, 1))
    else:
        final_exam_scores_updated.append(round(score, 1))

# Ensure no student gets more than 95 marks
final_exam_scores_final = []
for score in final_exam_scores_updated:
    if score > 93:
        score -= score * 0.05  # Apply 5% reduction
    final_exam_scores_final.append(round(score, 1))

# Create a DataFrame
data = {
    'q1': q1_scores,
    'q2': q2_scores,
    'q3': q3_scores,
    'time_spent': time_spent,
    'final_exam': final_exam_scores_final
}

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('enhanced_data.csv', index=False)

print("CSV file 'enhanced_data_with_error.csv' has been generated successfully.")
