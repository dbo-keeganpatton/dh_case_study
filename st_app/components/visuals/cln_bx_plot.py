import sys
import streamlit as st
import matplotlib.pyplot as plt
sys.path.append("../data_sets/")
from rfm_outlier_removed import clean_rfm_data_outliers 

def create_clean_box_plot(col):

    data = clean_rfm_data_outliers()
    fig, ax = plt.subplots()
    ax.boxplot(data[col], vert=False)
    ax.set_title(col)

    return st.pyplot(fig)

