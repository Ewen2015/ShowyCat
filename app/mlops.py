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
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

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

        col11, col12, col13 = st.columns(spec=3, gap='medium')
       
        with col11: 
            col_index = st.selectbox("Select the index column", [None] + list(df.columns))
        with col12:
            col_target = st.selectbox("Select the target column", [None] + list(df.columns))
        with col13:
            prob_type = st.selectbox("Select the ML problem type", ["regression", "classfication"])

        if col_index:
            df.set_index(col_index, inplace=True)

        col21, col22, col23 = st.columns(spec=3, gap='medium')

        if prob_type == "classfication":
            st.info("Please choose probabilities as your model output.", icon="⚠️")

        with col21:
            col_model1 = st.selectbox("Select the model 1", [None] + list(df.columns))
        with col22:
            col_model2 = st.selectbox("Select the model 2", [None] + list(df.columns))
        with col23:
            col_model3 = st.selectbox("Select the model 3", [None] + list(df.columns))

        list_models = []

        if col_target:
            list_models.append(col_target)

            if prob_type == "regression":
                list_errors = []
                list_rmse = []

                if col_model1:
                    col_model = col_model1

                    col_error = "ERROR_" + col_model
                    df[col_error] = df[col_model] - df[col_target]
                    list_models.append(col_model)
                    list_errors.append(col_error)

                    mse_model = mean_squared_error(df[col_target], df[col_model])
                    rmse_model = round(np.sqrt(mse_model), 2)
                    list_rmse.append(rmse_model)

                if col_model2:
                    col_model = col_model2

                    col_error = "ERROR_" + col_model
                    df[col_error] = df[col_model] - df[col_target]
                    list_models.append(col_model)
                    list_errors.append(col_error)

                    mse_model = mean_squared_error(df[col_target], df[col_model])
                    rmse_model = round(np.sqrt(mse_model), 2)
                    list_rmse.append(rmse_model)

                if col_model3:
                    col_model = col_model3

                    col_error = "ERROR_" + col_model
                    df[col_error] = df[col_model] - df[col_target]
                    list_models.append(col_model)
                    list_errors.append(col_error)

                    mse_model = mean_squared_error(df[col_target], df[col_model])
                    rmse_model = round(np.sqrt(mse_model), 2)
                    list_rmse.append(rmse_model)
            else:
                list_ap = []
                list_cm = []

                if col_model1:
                    col_model = col_model1

                    col_pred = "PRED_" + col_model
                    col_tf = "TRUEFALSE_" + col_model
                    df[col_pred] = df[col_model].apply(lambda x: 1 if x >= 0.5 else 0)
                    df[col_tf] = df[col_target] == df[col_pred]

                    ap = average_precision_score(df[col_target], df[col_model])
                    list_ap.append(round(ap, 2))

                    cm = confusion_matrix(df[col_target], df[col_pred])
                    list_cm.append(cm)

                if col_model2:
                    col_model = col_model2

                    col_pred = "PRED_" + col_model
                    col_tf = "TRUEFALSE_" + col_model
                    df[col_pred] = df[col_model].apply(lambda x: 1 if x >= 0.5 else 0)
                    df[col_tf] = df[col_target] == df[col_pred]

                    ap = average_precision_score(df[col_target], df[col_pred])
                    list_ap.append(round(ap, 2))

                    cm = confusion_matrix(df[col_target], df[col_pred])
                    list_cm.append(cm)

                if col_model3:
                    col_model = col_model3

                    col_pred = "PRED_" + col_model
                    col_tf = "TRUEFALSE_" + col_model
                    df[col_pred] = df[col_model].apply(lambda x: 1 if x >= 0.5 else 0)
                    df[col_tf] = df[col_target] == df[col_pred]

                    ap = average_precision_score(df[col_target], df[col_pred])
                    list_ap.append(round(ap, 2))

                    cm = confusion_matrix(df[col_target], df[col_pred])
                    list_cm.append(cm)

        st.dataframe(df)

with col2:
    st.write("## 🎍 Performance analysis")

    colors = ["#DB444B", "#006BA2", "#379A8B", "#3EBCD2", "#EBB434", "#B4BA39", "#9A607F", "#D1B07"]

    if uploaded_file :
        st.write("### 🌱 Key performance index")

        if prob_type == "regression":
            st.write("**RMSE**: (the root mean square error) measures the average difference between a statistical model's predicted values and the actual values.")
            col31, col32, col33 = st.columns(spec=3, gap="medium")
            
            if col_model1:
                with col31:
                    st.metric("RMSE of "+col_model1, list_rmse[0])
            if col_model2:
                with col32:
                    st.metric("RMSE of "+col_model2, list_rmse[1], round(list_rmse[1]-list_rmse[0], 2))
            if col_model3:
                with col33:
                    st.metric("RMSE of "+col_model3, list_rmse[2], round(list_rmse[2]-list_rmse[0], 2))

            st.write("### 🌿 Prediction chart")
            df_model = df[list_models]

            # fig_model = px.line(df_model, color_discrete_sequence=colors)
            # fig_model.update_layout(
            #     xaxis_title="Date",
            #     yaxis_title="Value",
            #     legend_title="Models",
            # )
            # st.plotly_chart(fig_model, use_container_width=True)
            st.line_chart(df_model)

            st.write("### 🍁 Model error analysis")
            df_error = df[list_errors]

            # fig_error = px.area(df_error, color_discrete_sequence=colors)
            # fig_error.update_layout(
            #     xaxis_title="Date",
            #     yaxis_title="Error",
            #     legend_title="Models",
            # )
            # st.plotly_chart(fig_error, use_container_width=True)

            st.area_chart(df_error)
        else:
            st.write("**AP**: (average precision score) measures the quality of a ranked list of results, taking into account both precision and recall.")
            col31, col32, col33 = st.columns(spec=3, gap="medium")
            
            if col_model1:
                with col31:
                    st.metric("AP of "+col_model1, list_ap[0])
            if col_model2:
                with col32:
                    st.metric("AP of "+col_model2, list_ap[1], round(list_ap[1]-list_ap[0], 2))
            if col_model3:
                with col33:
                    st.metric("AP of "+col_model3, list_ap[2], round(list_ap[2]-list_ap[0], 2))

            st.write("### 🌿 Prediction chart")
            col41, col42, col43 = st.columns(spec=3, gap="medium")
            
            if col_model1:
                with col41:
                    col_model = col_model1

                    col_pred = "PRED_" + col_model
                    df_model = df[[col_model, col_pred]]

                    hist_data = [df_model[df_model[col_pred] == pred][col_model] for pred in df_model[col_pred].unique()]
                    group_labels = df_model[col_pred].unique().astype(str)

                    fig_model = ff.create_distplot(hist_data=hist_data, group_labels=group_labels, colors=colors, show_hist=False)
                    fig_model.update_xaxes(range=[0, 1])
                    fig_model.update_layout(title=col_model)
                    fig_model.update_layout(height=360, width=600)

                    st.plotly_chart(fig_model, use_container_width=True)
            if col_model2:
                with col42:
                    col_model = col_model2

                    col_pred = "PRED_" + col_model
                    df_model = df[[col_model, col_pred]]

                    hist_data = [df_model[df_model[col_pred] == pred][col_model] for pred in df_model[col_pred].unique()]
                    group_labels = df_model[col_pred].unique().astype(str)

                    fig_model = ff.create_distplot(hist_data=hist_data, group_labels=group_labels, colors=colors, show_hist=False)
                    fig_model.update_xaxes(range=[0, 1])
                    fig_model.update_layout(title=col_model)
                    fig_model.update_layout(height=360, width=600)

                    st.plotly_chart(fig_model, use_container_width=True)
            if col_model3:
                with col43:
                    col_model = col_model3

                    col_pred = "PRED_" + col_model
                    df_model = df[[col_model, col_pred]]

                    hist_data = [df_model[df_model[col_pred] == pred][col_model] for pred in df_model[col_pred].unique()]
                    group_labels = df_model[col_pred].unique().astype(str)

                    fig_model = ff.create_distplot(hist_data=hist_data, group_labels=group_labels, colors=colors, show_hist=False)
                    fig_model.update_xaxes(range=[0, 1])
                    fig_model.update_layout(title=col_model)
                    fig_model.update_layout(height=360, width=600)

                    st.plotly_chart(fig_model, use_container_width=True)
           
            st.write("### 🍁 Model error analysis")            
            col51, col52, col53 = st.columns(spec=3, gap="medium")
            
            if col_model1:
                with col51:
                    col_model = col_model1
                    cm = list_cm[0]

                    TP = cm[1, 1]
                    FP = cm[0, 1]
                    TN = cm[0, 0]
                    FN = cm[1, 0]

                    fig_cm = go.Figure(data=go.Heatmap(
                        z=[[TP, FN], [FP, TN]],
                        x=['1', '0'],
                        y=['1', '0'],
                        colorscale='Blues',
                        showscale=True
                    ))

                    fig_cm.update_layout(
                        title=col_model,
                        xaxis_title='Predicted label',
                        yaxis_title='True label'
                    )
                    fig_cm.update_layout(height=360, width=600)

                    st.plotly_chart(fig_cm, use_container_width=True)
            if col_model2:
                with col52:
                    col_model = col_model1
                    cm = list_cm[1]

                    TP = cm[1, 1]
                    FP = cm[0, 1]
                    TN = cm[0, 0]
                    FN = cm[1, 0]

                    fig_cm = go.Figure(data=go.Heatmap(
                        z=[[TP, FN], [FP, TN]],
                        x=['1', '0'],
                        y=['1', '0'],
                        colorscale='Blues',
                        showscale=True
                    ))

                    fig_cm.update_layout(
                        title=col_model,
                        xaxis_title='Predicted label',
                        yaxis_title='True label'
                    )
                    fig_cm.update_layout(height=360, width=600)

                    st.plotly_chart(fig_cm, use_container_width=True)
            if col_model3:
                with col53:
                    col_model = col_model1
                    cm = list_cm[2]

                    TP = cm[1, 1]
                    FP = cm[0, 1]
                    TN = cm[0, 0]
                    FN = cm[1, 0]

                    fig_cm = go.Figure(data=go.Heatmap(
                        z=[[TP, FN], [FP, TN]],
                        x=['1', '0'],
                        y=['1', '0'],
                        colorscale='Blues',
                        showscale=True
                    ))

                    fig_cm.update_layout(
                        title=col_model,
                        xaxis_title='Predicted label',
                        yaxis_title='True label'
                    )
                    fig_cm.update_layout(height=360, width=600)

                    st.plotly_chart(fig_cm, use_container_width=True)



                

