# VS_Codes

**Project - Student Performance Indicator(Prediction)**

**Reason is Dataset has every features like Categorical,NaN,etc..; Same Jupyter Notebook codes we will write and we will convert it to Modular Programming**

**Important Note: Both for EDA and for pipeline - Include Object - Categorical Feature; Exclude Object - Numerical Feature**

**Reusable Functions like save_obj for Pickle file, can be used in Utils and can be called in Data Transformation**

#### Life cycle of Machine learning Project

1. Understanding the Problem Statement

2. Data Collection

3. Data Checks to perform

4. Exploratory data analysis

5. Data Pre-Processing

6. Model Training
   
7. Choose best model

creating virtual environment - conda create venv python==3.8 -y

activating venv - conda activate venv/

**If Conda not recognized, activate it through Anaconda Prompt (Go to the Folder Location and use "code."**

clone and sync vs code with github - git init


**To setup Username**
git config --global user.name "Ashwath Bala S"
git config --global user.email "ashwathbala0510@gmail.com"

**To make sure to be sync with Github Repository - git remote add origin https://github.com/Lambda123-design/End_to_End_ML_Project.git**

**To check sync status - git remote -v**

**To run notebooks from src - python src/components/data_ingestion.py**


**Steps done on ML Project:**

1. Used Anaconda Prompt to go inside the folder, and used "code ."

2. Created a Virtual Environment

3. Created "Git init"

4. Added a "Readme.md" file (It can be created in VS Code or both in git command too) (git add README.md)

5. Commiting it to GitHub (git commit -m "first commit") (Before that we can add it to, git add README.md)

6. Add Main Branch - git branch -M main and sync with GitHub -git remote add origin https://github.com/Lambda123-design/End_to_End_ML_Project.git

7. In Github - Created a GitIgnore File (To ignore the packages that won't be required - E.g. Python Version, venv (That is the Virtual Environment and we don't want to have it in GitHub (Command - Create .gitignore)

8. GitPull In VS Code, to make it synced with Github Repository

**We can also Automate these tasks, we will learn how we can automate these things and all later **

9. Create New Files - setup.py, requirements.txt (Will have all the packages that we need while we need to run the project)

**Need for setup.py --> Pypi may have many libraries (say seaborn, we do pip install, say for that also, setup.py is required); Same way we create setup.py which will be responsible for creating Machine Learning applications as a package that can be used in any projects**

**It can be creating as a package and can even be deployed in pypi (PythonPi), say from there anybody can use it and install in their system**

10. setup.py codes - from setuptools import find_packages, setup 

from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)-->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements=[]
    with open file_path as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n"," ") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Ashwath',
    author_email='ashwathbala510@gmail.com',
    packages=find_packages,
    install_requires=get_requirements('requirements.txt')
)

**In this code, we gave find_packages, how it is going to find is, we created "src" folder and created __init.py. Whenever the find_packages runs, it will go and see, in how many folders __init__.py is there.**

**Once it finds, we can import it anywhere, like where we import seaborn, pandas, Like how we see from Pypi**

**Entire project creation will be happening in that folder, whenever new folder is created, it will also be pointing to this**

**Whenever we install from requirements.txt, at that time setup.py should also be triggered, so we use "-e." at the end of "requirements.txt" file to do that, to build the packages**

11. **After that, installing the requirements.txt and setup.py** --> pip install -r requirements.txt

12. **Once done --> git add., git status**

Shows whatever files got added, say requirements.txt, setup.py, src /__init__.py

13. **Commiting Now - git commit -m "setup; And push to main branch - git push -u origin main**

14. **Components folder is created in SRC, along with __init__.py** (Reason for __init__.py, it can be created as a package and can be imported in some another file location

**Components - All are process that we will be doing such as Data Ingestion (Loading the Data - Reading the data from specific database; It is the process of module)**

**Similarly, next we will be having Data Transformation (Like we do for One-Hot Encoding, Label Encoding**

15. **Data Ingestion, Data Transformation, Model Trainer.py files has been created in Components**

**Data Ingestion might be having splitting into Training, Test, Validation data; Those codes we will be writing there**

**Model Trainer might be having the models we develop, evaluation metrics such as Confusion Matrix, RMSE.**

**From Model Trainer itself we will try to push the pickle file into the cloud**

16. **Pipeline Folder is created in SRC - Train Pipeline, Predict Pipeline, __init__.py is created; Train Pipeline will be used to trigger the components; Predict Pipeline will be used to do the predictions**

17. **logger.py and exception.py,utils.py created in src**

**utils.py - Any functionalities that is written in an common way that can be used in entire application (E.g. Reading Data, Save my model in Cloud; Utils code we will try to call inside the components itself**

18. **Exception Handling is created using sys library**

_,_,exc_tb=error_detail.exc_info() - Gives all error info 

**If exc_info we got exc_tb; In exc_tb we got the code file_name=exc_tb.tb_frame.f_code.co_filename**

**We created a function and then defined a class and then inherited it**

**Defined one more function inside the class after inheriting**     def __str__(self):return self.error_message

**Whenever we did trycache and raised exception, this exception message will be coming**

import sys
import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occurred in Python Script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
    def __str__(self):
        return self.error_message

**19. Created Logging - Logging is for the purpose of any execution that happens, we need to log all the information, execution in some files; If any error also, we need to track it**

Whatever log is created it will be with respect to the current working directory; In SRC, every file will start with logs and whatever log is coming

os.makedirs(logs_path,exist_ok=True) - Even though there is file, folder, keep on appending to it 

Whenver we want to create a log and want to succeed, we want to set it up in basic config

import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

**Whenever we get a new exception, log it with Logging file, and use logging.info to put it in Logging File**

**[2025-08-03 09:58:23,644 ] 18 root - INFO - Logging has started** - Output we got; First is TIme in Ascending; 18 is LineNumber

**20. We will commit it to GitHub**

**git status, git add ., git commit -m "logging and exception"**

**21. To connect Exception with Logging; from src.logger import logging, Use this code in Exception**

**Project Started here**

**22. Created a folder called Notebook; Inside it added "data" folder and inside it added "stud.csv" dataset; Inside the notebook folder, added EDA and Model Training Notebooks**

**Please refer to Notebook for Further Coding**

**Till Now we have learnt end-to-end in Jupyter Notebook. Next we will learn everything in form of Modular Coding**

**23. Pushed the Dataset, EDA and Model Training Notebook to GitHub**

## In Real companies, there will be Big Data Engineers, who will make sure that they collect data from different data sources and store it in Databases, Hadoop, MongoDB. I as a Data Scientist, make sure to read data from the data source; First we will try to load from Local and then try to load from MongoDB. We will read data, split it into Train-Test and then transformations will happen

## We did: Created a @dataClass to create artifacts, created a DataIngestionConfig to join Train,Test,Raw Data and created a DataIngestion class to initiate process using try and then an exception class

**Data Ingestion Component Starts Here**

24. **Started writing code into Data Ingestion in "Components"**

Created - @dataclass class DataIngestionConfig and Class DataIngestion to store the three artifacts inside the DataIngestion Class

Importing OS and sys because we will be using Custom Exception

**New Learning - from dataclasses import dataclass; This used to create class variables**

**We created DataIngestionConfig class - Data Ingestion needs some Inputs, say like where I need to save Train, Test, Raw Data; Output can be NumPy array or file saved in some other places; We used a decorator called dataclass (@dataclass) [We don't need to define init, directly we can define class variable]**

**Outputs we saved in this artifact - train_data_path: str=os.path.join('artifacts',"train.csv"); Same we did for Test and Raw Data**

**We can also create a file inside component and do, but as a starter this is easy to do**

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion.config=DataIngestionConfig()

**The three paths will get saved inside the class "DataIngestion**

25. **initiate_data_ingestion Class**

First we will try to read data from CSV; Later we will try to hold the MongoDB database in UTILS and try to read it

**In this class, we created a Logging "Entered the Data Ingestion Method or Component and then started the Data Ingestion using Exception Handling**

**In exception handling, Try - Read the data from CSV; Then created a folder to check for the artifacts "os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)". exist_ok=True, because if the folder is already there, we don't want to delete it**

**df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True); Next saved the Raw data into the Raw Data CSV**

**26. Created a Log to initiate Train Test Split and splitted into train test split train_set, test_set=train_test_split(df,test_size=0.2,random_state=42); Saved the train and test data into the artificats as did for the Raw Data and created a Log that "Ingestion of Data is completed"**

train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True); test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True); logging.info("Ingestion of Data is completed")

**27. Returned the Output in return and gave except case; And initiated using __init__**

return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()

**28. Run in Command Prompt to run the Data_Ingestion.py - python src\components\data_ingestion.py**

Artifacts will be created; Logs will be noted

**We can change only "df=pd.read_csv('notebook\data\stud.csv')" and read the data from any Databases or API's**

**Artifacts has been removed by adding it in "git.ignore"**

## We can also write train_test split in Utils to make the code clean

## Data Transformation Starts Here

Data Transformation - To do feature engineering, Data Cleaning, Convert Categorical to Numerical Features

**29. Started to write code at Data Transfomer**

from sklear.compose import ColumnTransformer - Used to create Pipeline which has StandardScaler, Encoding

from sklearn.impute import SimpleImputer - Used to Impute Missing Values

from sklearn.pipeline import Pipeline - Used to create the Pipeline

**DataIngestionConfig - Is like a Input used to give for the Data Ingestion**

preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl") - If any model we are trying to export using a Pickle file, we will save it as the same way in the artifacts; It is also like how we have pickle files which has Categorical converted to Numerical, StandardScalar done files

30. **Created a getDataTransformerobject and gave try and except class in it**

In try: Created two Pipelines - Numerical Pipeline and Categorical Pipeline.

Numerical Pipeline - Imputed Missing Values with Median (As there were Outliers), did Standard Scaling

Categorical Pipeline - Imputer Missing Values using Mode, Did One-Hot Encoding for Categorical Columns, Standard Scaling is done 

Created Logs for Numerical and Categorical Pipelines

**31. Created a Pipeline to include both Numerical and Categorical Pipelines**

preprocessor=ColumnTransformer(
                ("num_pipeline",num_pipeline,numerical_columns)
                ("cat_pipeline",cat_pipeline,categorical_columns)
            )

**32. Started the Data Transformation Object**:

Initiated Try, and then loaded train and test data as df; Logged for "Reading train and test data is completed", "Obtained Pre-Processing Object"

**Loaded Preprocessor from the above created function**

Gave Target column and Numerical Column 

**Whatever we gave for the Train Data, we did for the same for the test data too**

**Gave logging that applying preprocessing object on training dataframe and testing dataframe**

**33. Next Did Fit_transform for Train Data and Transform for test data**

Created input feature Array for Train and Test

input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df) and input_feature_test_arr=preprocessing_obj.transform(input_feature_train_df)

**34.  Merged features and labels into one array - Convinient for Model Training, Evaluation; Did for both train and test data** (train_arr=np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

**35. Logged for Saved Preprocessing Object**

**36. Created a save_object in Utils and called the save_object in DataTransformation to utilize the Pickle File**

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open (file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

**37. Returned train_array, test_array, self.data_transformation_config.preprocessor_obj_file_path**

**38. Imported DataTransformation and DataTransformationConfig in Data_Ingestion to check if everything is working fine**

## **To test wrote the below code in Data Ingestion:**

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)

### We have combined Data Ingestion and after than Data Transformation

## Krish didn't write the except block, so he didn't be able to see the error 

**39. Pushed the Data Transformation code to the GitHub Repository**

**Model Training Startes Here:**

**We will try to train different different models and evaluate it; We have to try with every algorithm and find the best model**

### For Every step we tried to create a Config file and from there we created artifacts, say like to save as a pickle file

**40. Created a Model TrainerConfig and saved model.pkl**

@dataclass
    class ModelTrainerConfig:
        trained_model_file_path=os.path.join("artifacts","model.pkl")

**41. Created the Model Trainer class and initialized the above created config**

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

**42. Inside the same class created a try class, Initialized Model Training using a function and started using getting logs for the same**

**43. After that Started Train, Test Split; and then created a dictionary of models that we will going to test**

**44. Created a Model Report to evaluate using X_train,X_test, y_train, y_test**

model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
            models=models,param=params)

**evaludate_model - We will create as a function in utils and loaded in the from statement in the model_trainer notebook**

**45. Wrote code for finding the best model and gave a condition that if the best model score is less than 0.6, then we say "No best model found"**

**After that we logged that Best model found on both training and test set (If <0.6, we said not found, if not we said it is found in the logs)**

**46. Next we saved the Model**

save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

**47. Next calculated R2 Score of best model**

**48. At the end raised a custom exception**

**49. To test it went to Data Ingestion and loaded from src.components.model_trainer import ModelTrainer and ModelTrainerConfig**

**50. Added train_arr, test_arr in Data Ingestion last line**

train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

**Added below, to print the R2 score in Data Ingestion**

modeltrainer=ModelTrainer(); print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

**51. Ran it from the Terminal (python src/components/data_ingestion.py)**

**Pushed it to GitHub**

## Next we will learn about Deployment using Dockers; In AWS using pushing the model, called as Model Pusher
