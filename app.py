import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer

st.set_page_config(layout='wide')
st.title('Life Expectancy Data Analysis')
st.write('The dataset is downloaded from [Kaggle](https://www.kaggle.com/mmattson/who-national-life-expectancy) for use in this analysis.')
st.write('This dataset is related to WHO National Life Expectancy and related parameters to predict the life expectancy and analysis the related parameters.')
st.write()
st.write('Before you continue, please read the [terms and conditions](https://www.gnu.org/licenses/gpl-3.0.en.html)')

show = st.checkbox('I agree the terms and conditions')

if show : 
    # Read Data
    df = pd.read_csv("who_life_exp.csv")

    # Preprocessing - Select X,y
    st.subheader('Raw Data Explorer')
    st.sidebar.header('Variable Selection')
    feature_list = list(df.iloc[:,0:].columns.values)
    feature_option = st.sidebar.multiselect('Select Feature Varable (X)',feature_list,default=list(df.iloc[:,0:1].columns.values))
    label_list = [ele for ele in feature_list if ele not in feature_option]
    label_option = st.sidebar.selectbox('Select Label for predicton (y)',label_list)
    selected_list = feature_option+[label_option]
    df1 = df[selected_list]
    
    #Missing Value
    df_na = df1.isna().sum().rename('No_of_Missing_Value',axis=1).to_frame().reset_index()
    df_na = df_na[df_na.No_of_Missing_Value != 0]
    df_na = df_na.sort_values('No_of_Missing_Value')

    st.sidebar.header('Setup')
    #Encoding Catergorical variable
    cat_check = df1.select_dtypes(include='object').any(axis=None)
    if cat_check == True :
        X1 = df1.select_dtypes(include=['object'])
        X2 = df1.select_dtypes(exclude=['object'])
        X3 = X1.apply(LabelEncoder().fit_transform)
        df1 = pd.concat([X3,X2],axis=1)
    
    #Missing Value Handling
    imp_sel = st.sidebar.radio('Handling with missing data', ['Remove row contain missing values','Impute Missing Value'])
    if imp_sel == 'Remove row contain missing values':
        na_select = st.sidebar.multiselect('Drop missing value column (if required)',df_na['index'])
        df1 = df1.drop(na_select, axis=1)
        df1 = df1.dropna()
    elif imp_sel == 'Impute Missing Value':
        imputetype = st.sidebar.selectbox('Select Scaler Method', ('Zero Value','Mean Imputation','KNN Imputation'))
        if imputetype == 'Zero Value' :
            df1= df1.fillna(0)
        elif imputetype == 'Mean Imputation' :
            df1= df1.fillna(df1.mean())
        elif imputetype == 'KNN Imputation' :
            neighbors = st.sidebar.slider('Select kNN no. of neighbors', min_value = 1, max_value = 20)
            imputer = KNNImputer(missing_values=np.nan,n_neighbors=neighbors)
            df1[:]=imputer.fit_transform(df1)
    
    #Final X and y  
    X_final = df1.drop([label_option],axis=1)   
    y = df1[label_option]

    #Visualize raw data
    scalefig2 = st.checkbox('Scale Data Overview Y-Axis')
    if scalefig2:
        df2 = df1 - df1.min()
        df2 = df2 / df1.max()*100
        fig2data = df2
    else :
        fig2data = df1
        
    col1, col2 = st.beta_columns(2)
    fig1, ax = plt.subplots(figsize=(10, 5))
    hbar = ax.barh(df_na['index'],df_na['No_of_Missing_Value'])
    ax.set_title('Missing Value')
    ax.set_xlabel('No.of missing value')
    col1.pyplot(fig1)

    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('Data Overview')
    ax.boxplot(fig2data)
    ax.set_xticklabels(fig2data.columns.values, rotation=90) 
    ax.set_ylabel('Observed values')
    col2.pyplot(fig2)
    
    #Model Setup - Model type
    st.sidebar.subheader('Model Setup')
    regresstype = st.sidebar.selectbox('Select Model', ('LinearRegression','RandomForestRegressor'))
    testsize = st.sidebar.slider('Select size of y_test split', min_value = 0.1, max_value = 0.3)
    if regresstype == 'LinearRegression' :
        model = LinearRegression(fit_intercept=True)
    elif regresstype == 'RandomForestRegressor'  :
        ntree = st.sidebar.slider('Select number of trees in the forest.', min_value = 1, max_value = 200)
        model = RandomForestRegressor(n_estimators=ntree,random_state=1)
    X_train,X_test,y_train,y_test=train_test_split(X_final,y,test_size=testsize,random_state=0)

   #Model Setup - apply scaler
    scaler_check = st.sidebar.checkbox('Apply MinMaxScaler')
    if scaler_check :
        scalertype = st.sidebar.selectbox('Select Scaler Method', ('MinMaxScaler','RobustScaler','QuantileTransformer'))
        if scalertype == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scalertype == 'RobustScaler':
            scaler = RobustScaler()
        elif scalertype == 'QuantileTransformer' :
            scaler = QuantileTransformer()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_train = X_train_scaled
        X_test =  X_test_scaled
    X_train_df = pd.DataFrame(X_train,columns = X_final.columns.values)
    X_test_df = pd.DataFrame(X_test,columns = X_final.columns.values)
    y_train_df = y_train.to_frame()
    y_test_df = y_test.reset_index()
    st.write('===============================================================================')
    st.subheader('Preprocessed Training Dataset Overview')

    #Visualize raw and preprocessed data
    fig3, ax = plt.subplots(1,2, figsize=(15, 3))
    ax[0].set_title('X_train')
    ax[0].boxplot(X_train_df)
    ax[0].set_xticklabels(X_train_df.columns.values, rotation=90) 
    ax[0].set_ylabel('Observed values')
    ax[1].set_title('y_train')
    ax[1].boxplot(y_train_df)
    ax[1].set_xticklabels(y_train_df.columns.values, rotation=90) 
    ax[1].set_ylabel('Observed values')
    st.pyplot(fig3)
    
    # Model Running
    if st.sidebar.button('Click To Run') : 
        st.write('===============================================================================')
        st.subheader('Calculation Result')
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.05)
        my_bar.progress(percent_complete + 1)
        data_load = st.text('Calculating .....')
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        resultdf1 = pd.Series(y_test).to_frame()
        resultdf2 = pd.Series(y_pred, name = 'Predict Value').to_frame()
        time.sleep(2)
        data_load.text('Calculating Done!!')
        
        #Model Result
        st.write('===============================================================================')
        st.write('R2 score(coefficient of determination) : {}'.format(r2_score(y_test,y_pred)))
        st.write('MSE (Mean Square Error) : {}'.format(mean_squared_error(y_test,y_pred)))
        st.write('RMSE (Mean Square Error) : {}'.format(mean_squared_error(y_test,y_pred,squared=False)))
        st.write('MAE (Mean Absolute Error) : {}'.format(mean_absolute_error(y_test,y_pred)))
        st.write('===============================================================================')
        col3, col4, col5 = st.beta_columns([1,1,2])
        if regresstype == 'LinearRegression'  :
            col3.write('Linear Regression Coefficients')
            coef = pd.DataFrame({'Factor': list(X_final.columns), 'Coefficient': model.coef_})
            col3.write(coef)
        elif regresstype == 'RandomForestRegressor'  :
            source = pd.DataFrame({'Factor': list(X_final.columns), 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)
            col3.write('RandomForestRegressor Importance')
            col3.write(source)
        y_pred_df = pd.Series(y_pred, name='prediction')
        y_compare = pd.concat([y_test_df,y_pred_df],axis=1)
        col4.write('Prediction Value')
        col4.write(y_compare)
        fig4, ax = plt.subplots(figsize=(10,5))
        ax.set_title('y_test & y_pred comparison')
        ax.scatter(y_compare.iloc[:,0],y_compare.iloc[:,1], label ='y_test')
        ax.scatter(y_compare.iloc[:,0],y_compare.iloc[:,2], label ='y_pred')
        ax.legend()
        col5.write('Comparison Chart between Real value and predicted value')
        col5.pyplot(fig4)