# VS_Codes
creating virtual environment - conda create venv python==3.8 -y

activating venv - conda activate venv/

**If Conda not recognized, activate it through Anaconda Prompt (Go to the Folder Location and use "code."**

clone and sync vs code with github - git init


**To setup Username**
git config --global user.name "Ashwath Bala S"
git config --global user.email "ashwathbala0510@gmail.com"

**To make sure to be sync with Github Repository - git remote add origin https://github.com/Lambda123-design/End_to_End_ML_Project.git**

**To check sync status - git remote -v**



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
