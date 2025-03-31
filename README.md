
# ML Analytics Utilities

## Overview

The ML Analytics Utilities package offers tools to facilitate data processing and database connections, and manage the lifecycle of machine learning models with MLflow. The idea is to make our work easily reproducible. It includes features for registering, logging, and deleting models, simplifying managing machine learning models in a production environment. 

It also includes functionalities for executing SQL queries, saving data to Redshift and S3, and managing S3 files.

## Objectives

### Data Connector
-  **Database Connection**: Establish and manage connections to Redshift databases.
-  **SQL Execution**: Execute SQL queries and retrieve results as pandas DataFrames.
-  **Data Storage**: Save pandas DataFrames to Redshift tables and S3 buckets.
-  **File Management**: List, save, and delete files in S3.
-  **Path Management**: Create local file paths for data storage and retrieval.

### Model Manager (with MLflow)
-  **Model Registration**: Register new models and update existing models with new tags or descriptions.
-  **Model Logging**: Log machine learning models using various MLflow flavors, capturing model signatures and input examples.
-  **Model Versioning**: Manage different versions of models, including setting aliases for easy reference.
-  **Model Deletion**: Remove specific versions of models from the MLflow registry.
-  **Experiment Setup**: Automatically set up MLflow experiments for models.


### Setup

Ensure that your environment variables are set in a `.env` file. This file should include your database and AWS credentials, and environment variables for MLflow tracking. You can use the `.env_template` file as reference.

## Data connector


### Initialization

```python
from utils.data_connector import DataConnector

# Initialize the DataConnector
dc = DataConnector()
```

### Executing SQL Queries

```python
query = "SELECT * FROM farming.health_score_modelling_churn_base"
df = dc.sql(query)
print(df.head())
```

### Saving Data to Reshift

```python
dc.save_table_redshift(df, table_name="test", schema="test_bi")
```

### Saving Data to S3

```python
data_connector.save_dataframe_to_s3(
    directory='ml-bi-projects/test', filename='test', file_format='parquet')
```

### Listing and delete S3 files

```python
dc.list_s3_files(prefix='ml-bi-projects')
dc.delete_s3_file(key='ml-bi-projects/test/test.parquet')
```

## Model Manager

Below is a step-by-step guide on how to use the `ModelManager` class to manage your ML models.

### Initialization

```python
from utils.model_manager import ModelManager

# Initialize the ModelManager
model_manager = ModelManager(
    model_name="example_model",
    task="classification",
    project="Project XYZ",
    description="An example classification model",
    team="bi",
    user="sebastian.daza"
)
```

### Logging a model

```python
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Sample data
input_data = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
predictions = np.array([0, 1, 0])

# Train a sample model
model = RandomForestClassifier()
model.fit(input_data, predictions)

# Log the model
model_manager.log_model(
    model=model,
    input_data=input_data,
    predictions=predictions,
    register_model=True,
    flavor="sklearn",
    description="A RandomForest model for classification",
    tags={'status':'testing'}
)
```

### Managing model versions

```python
# Set an alias for the latest model version
model_manager.set_model_alias(alias="test", version=5)

# Retrieving latest model version
my_model = model_manager.get_latest_model()
my_model.predict(input_data)

# Retrieving model by alias
my_model = model_manager.get_model_by_alias(alias="test")
my_model.predict(input_data)

# Deleting model using a specific version
model_manager.delete_model(version=4)

# Delete model using alias
model_manager.delete_model(alias="test")
```