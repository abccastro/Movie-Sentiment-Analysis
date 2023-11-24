# app.py
import streamlit as st

def main():
    st.title("Interactive Streamlit App")
    
    # Add a slider
    age = st.slider("Select your age", 0, 100, 25)

    # Add a button
    if st.button("Click me"):
        st.write(f"You clicked the button! Your age is {age}.")

if __name__ == "__main__":
    main()
