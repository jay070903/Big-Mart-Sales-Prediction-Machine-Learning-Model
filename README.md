import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
df_Train = pd.read_csv(r'C:\Users\Lenovo\Downloads\archive (2)\Train.csv') 
df_Test = pd.read_csv(r'C:\Users\Lenovo\Downloads\archive (2)\Test.csv')
df_Train.head()
df_Train.shape
df_Train.isnull().sum()
df_Test.isnull().sum()
df_Train.info()
df_Train.describe()
df_Train['Item_Weight'].describe()
df_Train['Item_Weight'].fillna(df_Train['Item_Weight'].mean(),inplace=True)
df_Test['Item_Weight'].fillna(df_Test['Item_Weight'].mean(),inplace=True)
df_Train.isnull().sum()
df_Train['Item_Weight'].describe()
df_Train['Outlet_Size'].value_counts()
df_Train['Outlet_Size'].mode()
df_Train['Outlet_Size'].fillna(df_Train['Outlet_Size'].mode()[0],inplace=True)
df_Test['Outlet_Size'].fillna(df_Test['Outlet_Size'].mode()[0],inplace=True)
df_Train.isnull().sum()
df_Test.isnull().sum()
df_Train.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
df_Test.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
df_Train
import klib
klib.cat_plot(df_Train) 
klib.missingval_plot(df_Train)
klib.dist_plot(df_Train) 
klib.data_cleaning(df_Train)
klib.clean_column_names(df_Train)
df_Train.info()
df_Train=klib.convert_datatypes(df_Train)
df_Train.info()
klib.mv_col_handling(df_Train)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_Train['item_fat_content']= le.fit_transform(df_Train['item_fat_content'])
df_Train['item_type']= le.fit_transform(df_Train['item_type'])
df_Train['outlet_size']= le.fit_transform(df_Train['outlet_size'])
df_Train['outlet_location_type']= le.fit_transform(df_Train['outlet_location_type'])
df_Train['outlet_type']= le.fit_transform(df_Train['outlet_type'])
df_Train
X=df_Train.drop('item_outlet_sales',axis=1)
Y=df_Train['item_outlet_sales']
from sklearn.model_selection import train_test_split

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y, random_state=101, test_size=0.2)
X.describe()
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_Train_std= sc.fit_transform(X_Train)
X_Test_std= sc.transform(X_Test)
X_Train_std
X_Test_std
Y_Train
Y_Test
from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(X_Train_std,Y_Train)
X_Test.head()
Y_pred_lr=lr.predict(X_Test_std)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print(r2_score(Y_Test,Y_pred_lr))
print(mean_absolute_error(Y_Test,Y_pred_lr))
print(np.sqrt(mean_squared_error(Y_Test,Y_pred_lr)))
from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor(n_estimators=1000)
rf.fit(X_Train_std,Y_Train)
Y_pred_rf= rf.predict(X_Test_std)
print(r2_score(Y_Test,Y_pred_rf))
print(mean_absolute_error(Y_Test,Y_pred_rf))
print(np.sqrt(mean_squared_error(Y_Test,Y_pred_rf)))
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# define models and parameters
model = RandomForestRegressor()
n_estimators = [10, 100, 1000]
max_depth=range(1,31)
min_samples_leaf=np.linspace(0.1, 1.0)
max_features=["auto", "sqrt", "log2"]
min_samples_split=np.linspace(0.1, 1.0, 10)

# define grid search
grid = dict(n_estimators=n_estimators)

#cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=101)

grid_search_forest = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, 
                           scoring='r2',error_score=0,verbose=2,cv=2)

grid_search_forest.fit(X_Train_std, Y_Train)

# summarize results
print(f"Best: {grid_search_forest.best_score_:.3f} using {grid_search_forest.best_params_}")
means = grid_search_forest.cv_results_['mean_test_score']
stds = grid_search_forest.cv_results_['std_test_score']
params = grid_search_forest.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print(f"{mean:.3f} ({stdev:.3f}) with: {param}")
    grid_search_forest.best_params_
    grid_search_forest.best_score_
    Y_pred_rf_grid=grid_search_forest.predict(X_Test_std)
    r2_score(Y_Test,Y_pred_rf_grid)
    import joblib
    joblib.dump(grid_search_forest,r'C:\Users\Lenovo\Downloads\final model\models\random_forest_grid.sav')
    model=joblib.load(r'C:\Users\Lenovo\Downloads\final model\models\random_forest_grid.sav')
