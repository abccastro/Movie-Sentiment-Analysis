# app.py
import streamlit as st
import pandas as pd
import datetime
import numpy as np
import altair as alt

# Create a dummy DataFrame
data = {
    'Input1': ['Text_A', 'Text_B', 'Text_C', 'Text_D', 'Text_E', 'Text_A', 'Text_B', 'Text_C'],
    'Input2': ['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5', 'Option 1', 'Option 2', 'Option 3'],
    'Input3': [datetime.date(2022, 1, 1), datetime.date(2022, 2, 2), datetime.date(2022, 3, 3),
               datetime.date(2022, 4, 4), datetime.date(2022, 5, 5), datetime.date(2022, 6, 6),
               datetime.date(2022, 7, 7), datetime.date(2022, 8, 8)],
    'Numeric1': [10, 20, 15, 25, 30, 12, 18, 22],
    'Numeric2': [50, 40, 30, 20, 10, 45, 35, 25]
}

df = pd.DataFrame(data)

def generate_report(filtered_df):
    # Calculate some hypothetical metrics using the numerical columns
    total_numeric1 = filtered_df['Numeric1'].sum()
    average_numeric2 = filtered_df['Numeric2'].mean()

    # Create a bar chart
    bar_chart = alt.Chart(filtered_df).mark_bar().encode(
        x='Input1',
        y='Numeric1',
        color='Input2',
        tooltip=['Input1', 'Numeric1']
    ).properties(
        title='Numeric1 by Input1 (colored by Input2)',
        width=500
    )

    # Create a line chart
    line_chart = alt.Chart(filtered_df).mark_line().encode(
        x='Input1',
        y='Numeric2',
        color='Input2',
        tooltip=['Input1', 'Numeric2']
    ).properties(
        title='Numeric2 by Input1 (colored by Input2)',
        width=500
    )

    # Display the charts
    st.header("Report:")
    st.write(f"Total Numeric1: {total_numeric1}")
    st.write(f"Average Numeric2: {average_numeric2}")

    st.subheader("Bar Chart:")
    st.altair_chart(bar_chart)

    st.subheader("Line Chart:")
    st.altair_chart(line_chart)

def main():
    st.title("User Input Streamlit App")

    # Input 1: Textbox
    input_text = st.text_input("Input 1: Enter some text")

    # Input 2: Dropdown
    options = ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"]
    input_dropdown = st.selectbox("Input 2: Select an option", options)

    # Input 3: Date selection
    input_date = st.date_input("Input 3: Select a date", datetime.date.today())

    # Button to trigger the display
    if st.button("Search Records"):
        # Filter DataFrame based on selected values
        filtered_df = df[(df['Input1'] == input_text) & (df['Input2'] == input_dropdown) & (df['Input3'] == input_date)]

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
