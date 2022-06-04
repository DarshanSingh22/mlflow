import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import hashlib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
import mlflow


DOWNLOAD_ROOT="https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH=os.path.join("datasets","housing")
HOUSING_URL = DOWNLOAD_ROOT+"datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL,housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz= tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    


fetch_housing_data()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)


# Stratified data split

def split_data(test_size,housing):
    split = StratifiedShuffleSplit(n_splits=1,test_size=test_size,random_state=42)
    for train_index,test_index in split.split(housing,housing['income_cat']):
        train_set=housing.loc[train_index]
        test_set = housing.loc[test_index]
    housing.drop('income_cat',axis=1,inplace=True)    
    train_set_labels = train_set['median_house_value']
    test_set_labels = test_set['median_house_value']
    test_set.drop({'income_cat','median_house_value'},axis=1,inplace=True)
    train_set.drop({'income_cat','median_house_value'},axis=1,inplace=True) 

    return train_set,test_set,train_set_labels,test_set_labels
    


# How to define our own transformer  
# BaseEstimator gives two functions get_params(),set_params()
# TransformerMaxin can be used to mix two functions like fit() and transform() into fit_tranform()
rooms_ix,bedrooms_ix,population_ix,household_ix = 3,4,5,6

class CombinedAttributeAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=True):     #no *args or **kargs
        self.add_bedrooms_per_room=add_bedrooms_per_room
    def fit(self,X,y=None):
        return self 
    def transform(self,X,y=None):
        rooms_per_household = X[:,rooms_ix]/X[:,household_ix]
        population_per_household=X[:,population_ix]/X[:,population_ix]
        if self.add_bedrooms_per_room :
            bedrooms_per_room =X[:,rooms_ix]/X[:,household_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else :
            return np.c_[X,rooms_per_household,population_per_household]
        
        


#  Transformation Pipelines
num_attr = list()
cat_attr = ['ocean_proximity']

num_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('attr_adder',CombinedAttributeAdder()),
    ('std_scaler',StandardScaler())
])

# calling both the pipline simultaneously or joining the pipelines

full_pipeline = ColumnTransformer([
    ('num_pipeline',num_pipeline,num_attr),
    ("cat",OneHotEncoder(),cat_attr)
])


# ### Training and Evaluating the Model

def eval_metrics(actual, pred):
    # compute relevant metrics
    rmse = np.sqrt(metrics.mean_squared_error(actual, pred))
    mae = metrics.mean_absolute_error(actual, pred)
    r2 = metrics.r2_score(actual, pred)
    return rmse, mae, r2


# mlflow server configurations

remote_server_uri = "http://localhost:5000/" # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)

mlflow.get_tracking_uri()

exp_name = "House_price_pred"
mlflow.set_experiment(exp_name)

def train(test_size):
  
    with mlflow.start_run(run_name='Parent_run') as parent_run:
        #fetch the data
        housing = load_housing_data()
        
        #data preparation
        with mlflow.start_run(run_name='Child_run_1',nested=True) as child_run:
            #creating a categorical median income column
            housing['income_cat']=np.ceil(housing['median_income']/1.5)
            housing['income_cat'].where(housing['median_income']<5,5.0,inplace=True)  #replace where the condition is false   
            #split the data into train and test set
            train_X,test_X,train_y,test_y = split_data(test_size,housing)
            train_X_num = train_X.drop('ocean_proximity',axis=1)
            num_attr = list(train_X_num)
           
        # Model training
        with mlflow.start_run(run_name='Child_run_2',nested=True) as child_run:
                train_set_prep=full_pipeline.fit_transform(train_X)
                lin_reg = LinearRegression()
                lin_reg = lin_reg.fit(train_set_prep,train_y)
                
        #Model Prediction and Evaluation 
        with mlflow.start_run(run_name='Child_run_3',nested=True) as child_run:
                test_set_prep = full_pipeline.fit_transform(test_X)
                predicted_ = lin_reg.predict(test_set_prep)
                (rmse, mae, r2) = eval_metrics(test_y, predicted_)

        # Print out metrics
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        

        # Log parameter, metrics, and model to MLflow
        mlflow.log_metric(key = 'test_size', value = test_size)
        mlflow.log_metric(key="rmse", value=rmse)
        mlflow.log_metrics({"mae": mae, "r2": r2})
       # mlflow.log_artifact(HOUSING_URL)
         
        mlflow.sklearn.log_model(lin_reg, "model")

train(0.3)


