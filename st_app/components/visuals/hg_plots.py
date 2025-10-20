import sys
import streamlit as st
import matplotlib.pyplot as plt
sys.path.append("../data_sets/")
from rfm import rfm_data

def create_histogram(col):

    data = rfm_data()
    fig, ax = plt.subplots()
    data[col].hist(ax=ax, bins=5)

    return st.pyplot(fig)

