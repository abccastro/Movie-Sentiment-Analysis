# app.py
import streamlit as st
import datetime

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
    if st.button("Show Selected Values"):
        # Display selected values
        st.header("Selected Values:")
        st.subheader(f"Input 1 (Text): {input_text}")
        st.subheader(f"Input 2 (Dropdown): {input_dropdown}")
        st.subheader(f"Input 3 (Date): {input_date}")

if __name__ == "__main__":
    main()
