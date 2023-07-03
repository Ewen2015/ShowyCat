#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""
import os
import pandas as pd
import numpy as np 

import plotly.express as px
import streamlit as st

from sklearn.metrics import mean_squared_error

st.set_page_config(
     page_title="MLOps",
     page_icon="🍁",
     layout="wide",
     initial_sidebar_state=st.session_state.get('sidebar_state', 'expanded')
     )

st.session_state.sidebar_state = 'expanded'
st.sidebar.header("🪵 Let's set it up!")
uploaded_file = st.sidebar.file_uploader("#### Upload 🖖🏼", type=["csv"])


st.header("☘️ MLOps: Visualize your ML models' results!")

col1, col2 = st.columns(spec=[2, 3], gap='medium')

with col1: 
    st.write("## 🪴 Data preview")

    if uploaded_file :
        df = pd.read_csv(uploaded_file)

        col11, col12 = st.columns(spec=2, gap='medium')
       
        with col11: 
            col_index = st.selectbox("Select the index column", [None] + list(df.columns))
        with col12:
            col_target = st.selectbox("Select the target column", [None] + list(df.columns))

        if col_index:
            df.set_index(col_index, inplace=True)

        col21, col22, col23 = st.columns(spec=3, gap='small')

        with col21:
            col_model1 = st.selectbox("Select the model 1", [None] + list(df.columns))
        with col22:
            col_model2 = st.selectbox("Select the model 2", [None] + list(df.columns))
        with col23:
            col_model3 = st.selectbox("Select the model 3", [None] + list(df.columns))

        list_models = []
        list_errors = []
        list_metrics = []

        if col_target:
            list_models.append(col_target)

            if col_model1:
                col_error1 = "ERROR_" + col_model1
                df[col_error1] = df[col_model1] - df[col_target]
                list_models.append(col_model1)
                list_errors.append(col_error1)

                mse_model1 = mean_squared_error(df[col_target], df[col_model1])
                rmse_model1 = round(np.sqrt(mse_model1), 2)
                list_metrics.append(rmse_model1)

            if col_model2:
                col_error2 = "ERROR_" + col_model2
                df[col_error2] = df[col_model2] - df[col_target]
                list_models.append(col_model2)
                list_errors.append(col_error2)

                mse_model2 = mean_squared_error(df[col_target], df[col_model2])
                rmse_model2 = round(np.sqrt(mse_model2), 2)
                list_metrics.append(rmse_model2)

            if col_model3:
                col_error3 = "ERROR_" + col_model3
                df[col_error3] = df[col_model3] - df[col_target]
                list_models.append(col_model3)
                list_errors.append(col_error3)

                mse_model3 = mean_squared_error(df[col_target], df[col_model3])
                rmse_model3 = round(np.sqrt(mse_model3), 2)
                list_metrics.append(rmse_model3)

        st.dataframe(df)

with col2:
    st.write("## 🎍 Performance analysis")

    # colors = ["#DB444B", "#006BA2", "#379A8B", "#3EBCD2", "#EBB434", "#B4BA39", "#9A607F", "#D1B07"]

    if uploaded_file :
        st.write("### 🌼 Key Performance Index")
        st.write("**RMSE**: (the root mean square error) measures the average difference between a statistical model's predicted values and the actual values.")
        col31, col32, col33 = st.columns(spec=3, gap="medium")
        
        if col_model1:
            with col31:
                st.metric("RMSE of "+col_model1, list_metrics[0])
        if col_model2:
            with col32:
                st.metric("RMSE of "+col_model2, list_metrics[1], round(list_metrics[1]-list_metrics[0], 2))
        if col_model3:
            with col33:
                st.metric("RMSE of "+col_model3, list_metrics[2], round(list_metrics[2]-list_metrics[0], 2))

        st.write("### 🌸 Prediction chart")
        df_model = df[list_models]

        # fig_model = px.line(df_model, color_discrete_sequence=colors)
        # fig_model.update_layout(
        #     xaxis_title="Date",
        #     yaxis_title="Value",
        #     legend_title="Models",
        # )
        # st.plotly_chart(fig_model, use_container_width=True)
        st.line_chart(df_model)

        st.write("### 🌻 Model error analysis")
        df_error = df[list_errors]

        # fig_error = px.area(df_error, color_discrete_sequence=colors)
        # fig_error.update_layout(
        #     xaxis_title="Date",
        #     yaxis_title="Error",
        #     legend_title="Models",
        # )
        # st.plotly_chart(fig_error, use_container_width=True)

        st.area_chart(df_error)


















