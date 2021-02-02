# yellowcab_analysis
Analysis based on the task given to me during the one of the recruitment processes that I participated in.

As this project uses variety of packages, check the req.txt file to ensure that Your enviroment will handle runnning 
the notebooks.

## setting-up the enviroment

### using pip
$ pip install -r requirements.txt

### using Conda
$ conda create --name <env_name> --file requirements.txt

If You wish to export Your enviroment information to the YAML file, please use:
$ conda env create -f environment.yml

While experimenting, keep on mind that this is high-volume data, trigerring computation on dask dataframe from one month 
will consume up to 2.8 GB of Your RAM. 

## Project structure
Project tree is shown below:

.
├── ├── .gitignore

├── README.md 

├── src

├── ├── stored_model

│   ├── models.py

├── ├── ├── drivetime_model.py

│   ├── generators

├── ├── ├── drivetime_generator.py

│   ├── toolkit

├── ├── ├── analysis_toolkit.py

├── ├── ├── etl_toolkit.py

├── report.pdf

├── yellowcab_data_domain_understanding.ipnyb

├── yellowcab_eda.ipynb

├── yellowcab_models.ipynb

├── environment.yml

├── requirements.txt
