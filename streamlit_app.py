# app.py
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import datetime

def generate_report(filtered_df):
    # Calculate some hypothetical metrics using the numerical columns
    total_salary = filtered_df['Salary'].sum()
    average_work_hours = filtered_df['Work_Hours'].mean()

    # Create a bar chart
    bar_chart = alt.Chart(filtered_df).mark_bar().encode(
        x='Department',
        y='Salary',
        color='Gender',
        tooltip=['Department', 'Salary']
    ).properties(
        title='Salary by Department (colored by Gender)',
        width=300
    )

    # Create a line chart
    line_chart = alt.Chart(filtered_df).mark_line().encode(
        x='Department',
        y='Work_Hours',
        color='Gender',
        tooltip=['Department', 'Work_Hours']
    ).properties(
        title='Work Hours by Department (colored by Gender)',
        width=300
    )

    # Create a bar chart for Age Distribution by Gender
    age_distribution_chart = alt.Chart(filtered_df).mark_bar().encode(
        x='Age:Q',
        y='count()',
        color='Gender:N'
    ).properties(
        title='Age Distribution by Gender',
        width=300
    )

    # Create a bar chart for Salary Distribution by Department
    salary_distribution_chart = alt.Chart(filtered_df).mark_bar().encode(
        x='Department:N',
        y='average(Salary):Q',
        color='Gender:N'
    ).properties(
        title='Salary Distribution by Department',
        width=300
    )

    # Display the charts in a single row
    st.header("Report:")
    st.write(f"Total Salary: {total_salary}")
    st.write(f"Average Work Hours: {average_work_hours}")

    charts_row1, charts_row2 = st.columns(2)

    with charts_row1:
        st.subheader("Bar Chart:")
        st.altair_chart(bar_chart)

    with charts_row1:
        st.subheader("Line Chart:")
        st.altair_chart(line_chart)

    with charts_row2:
        st.subheader("Age Distribution by Gender:")
        st.altair_chart(age_distribution_chart)

    with charts_row2:
        st.subheader("Salary Distribution by Department:")
        st.altair_chart(salary_distribution_chart)

def main():
    st.title("User Input Streamlit App")

    # Read the data from the CSV file
    file_path = 'dummy_dataset_10000_records.csv'  # Update with your file path
    df = pd.read_csv(file_path)

    # Input 1: Textbox for Department (case-insensitive)
    input_department = st.text_input("Input 1: Enter Department", '').lower()

    # Input 2: Dropdown for Gender
    options_gender = ["Male", "Female"]
    input_gender = st.selectbox("Input 2: Select Gender", options_gender)

    # Input 3: Date selection for Joining Date
    input_joining_date = st.date_input("Input 3: Select Joining Date", datetime.date.today())

    # Button to trigger the display
    if st.button("Filter and Generate Report"):
        # Convert input_joining_date to Pandas Timestamp
        input_joining_date = pd.to_datetime(input_joining_date)

        # Filter DataFrame based on selected values
        filtered_df = df[
            (df['Department'].str.lower() == input_department) &
            (pd.to_datetime(df['Joining_Date']) > input_joining_date) &
            (df['Gender'] == input_gender)
        ]

        # Display filtered records and generate graphical report
        st.header("Filtered Records:")
        if filtered_df.empty:
            st.warning("No records found with the selected values.")
        else:
            st.dataframe(filtered_df)

            # Generate and display the report
            generate_report(filtered_df)

if __name__ == "__main__":
    main()
