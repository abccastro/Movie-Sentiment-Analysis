# app.py
import pandas as pd
import numpy as np
import random
import datetime
from faker import Faker

# Set seed for reproducibility
np.random.seed(42)

# Create Faker instance for generating names
fake = Faker()

# Generate a dummy dataset with 10,000 records
data = {
    'ID': [f'ID_{i}' for i in range(1, 10001)],
    'First_Name': [fake.first_name() for _ in range(10000)],
    'Last_Name': [fake.last_name() for _ in range(10000)],
    'Age': np.random.randint(18, 65, size=10000),
    'Gender': [random.choice(['Male', 'Female']) for _ in range(10000)],
    'Salary': np.random.uniform(30000, 100000, size=10000),
    'Department': [random.choice(['HR', 'IT', 'Finance', 'Marketing']) for _ in range(10000)],
    'Joining_Date': [datetime.date(2022, np.random.randint(1, 13), np.random.randint(1, 29)) for _ in range(10000)],
    'Has_Health_Insurance': [random.choice([True, False]) for _ in range(10000)],
    'Work_Hours': np.random.randint(6, 10, size=10000),
    'Performance_Rating': np.random.choice([1, 2, 3, 4, 5], size=10000)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file for reference
df.to_csv('dummy_dataset_10000_records.csv', index=False)

# Display the DataFrame
print(df.head())
