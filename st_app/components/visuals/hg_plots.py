import sys
import streamlit as st
import matplotlib.pyplot as plt
sys.path.append("../data_sets/")
from rfm import rfm_data

def create_histogram(col):

    data = rfm_data()
    fig, ax = plt.subplots()
    plt.title(col)
    data[col].hist(
        ax=ax,
        bins=5,
        grid=False,
        rwidth=0.8,
        edgecolor='black',
        ylabelsize=0
    )

    return st.pyplot(fig)

