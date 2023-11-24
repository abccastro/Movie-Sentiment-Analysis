# app.py
import streamlit as st

def main():
    st.title("Streamlit App with Dropdown")

    # Add a dropdown
    selected_option = st.selectbox("Select an option", ["Option 1", "Option 2", "Option 3"])

    st.write(f"You selected: {selected_option}")

if __name__ == "__main__":
    main()
