import sys
import streamlit as st
import matplotlib.pyplot as plt
sys.path.append("../data_sets/")
from rfm import rfm_data

def create_box_plot(col):

    data = rfm_data()
    fig, ax = plt.subplots()
    ax.boxplot(data[col], vert=False)
    ax.set_title(col)

    return st.pyplot(fig)

