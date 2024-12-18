import streamlit as st

st.set_page_config(page_title="LLM Tools")
st.title("LLM Tools")
st.write(
    "Welcome! LLM Tools is an open-source project designed to provide essential tools for developing and running large language models (LLMs)."
)

if st.button("Memory Requirements Calculator"):
    st.switch_page("pages/1_Memory.py")
elif st.button("Costs Calculator"):
    st.switch_page("pages/2_Costs.py")

st.write(
    "This project is open-source and actively welcomes contributions from developers and enthusiasts worldwide. Feel free to explore, use, and improve it!"
)

# Add a box for the GitHub repository
st.markdown(
    """
    <div style="
        border: 2px solid #4CAF50; 
        border-radius: 10px; 
        padding: 10px; 
        background-color: #f9f9f9; 
        width: fit-content;
    ">
        <a href="https://github.com/manuelescobar-dev/LLM-Tools" 
           style="text-decoration: none; 
                  color: #4CAF50; 
                  font-weight: bold; 
                  font-size: 16px;" 
           target="_blank">
            Visit GitHub Repository
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
