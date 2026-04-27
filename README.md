# Smart Logistics Delay Prediction

Final project for Machine Learning (Spring 2026).

**Important:** The original dataset is uploaded as a ZIP file 
(`DataCoSupplyChainDataset.zip`) due to GitHub size limits. 
Before running preprocessing, please unzip it first:

## Overview

This project predicts whether a shipment will be delivered late or on time using machine learning models.

Dataset used: DataCo Smart Supply Chain dataset (~180,000 orders)

Models tested:
- Logistic Regression
- Decision Tree
- Random Forest

Target variable:

`Late_delivery_risk`

- 0 = On-Time  
- 1 = Late


## Data and Features

Some features used in the project include:

- Shipping Mode  
- Product Category  
- Payment Type  
- Order Region  
- Sales related variables  
- Engineered feature: `ship_gap`

Some columns were removed during preprocessing because of leakage, missing values, or redundancy.


## Project Structure

```text
smart-logistics-delay-prediction/

data/              
notebooks/          
src/                
results/plots/      
report/             

README.md
requirements.txt
```

Files in `src/`:

```text
preprocessing.py
model_training.py
```

Notebook:

```text
EDA.ipynb
```


## Run

Install packages:

```bash
pip install -r requirements.txt
```

Run preprocessing:

```bash
python src/preprocessing.py
```

Run model training:

```bash
python src/model_training.py
```

Open EDA notebook:

```bash
jupyter notebook notebooks/EDA.ipynb
```


## Current Result

Best model (current):

Random Forest — F1 Score around 0.98


## Author

Sachin Kirtibhai Patel  
U01124876  
Wright State University