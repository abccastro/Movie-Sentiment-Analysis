# app.py
import streamlit as st
import pandas as pd
import datetime

# Create a dummy DataFrame
data = {
    'Input1': ['Text_A', 'Text_B', 'Text_C', 'Text_D', 'Text_E'],
    'Input2': ['Option 1', 'Option 2', 'Option 3', 'Option 4', 'Option 5'],
    'Input3': [datetime.date(2022, 1, 1), datetime.date(2022, 2, 2), datetime.date(2022, 3, 3),
               datetime.date(2022, 4, 4), datetime.date(2022, 5, 5)]
}

df = pd.DataFrame(data)

def main():
    st.title("User Input Streamlit App")

    # Input 1: Textbox
    input_text = st.text_input("Input 1: Enter some text")

    # Input 2: Dropdown
    options = data['Input2']
    input_dropdown = st.selectbox("Input 2: Select an option", options)

    # Input 3: Date selection
    input_date = st.date_input("Input 3: Select a date", datetime.date.today())

    # Button to trigger the display
    if st.button("Search Records"):
        # Filter DataFrame based on selected values
        filtered_df = df[(df['Input1'] == input_text) & (df['Input2'] == input_dropdown) & (df['Input3'] == input_date)]

        # Display filtered records
        st.header("Filtered Records:")
        if filtered_df.empty:
            st.warning("No records found with the selected values.")
        else:
            st.dataframe(filtered_df)

if __name__ == "__main__":
    main()
