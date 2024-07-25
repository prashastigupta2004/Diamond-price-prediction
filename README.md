# Diamond Price Prediction

A machine learning project to predict the price of diamonds based on various attributes using regression models.

## Overview

This project aims to predict diamond prices based on attributes such as carat, cut, color, clarity, and dimensions using machine learning techniques. The objective is to build a model that can accurately estimate the price given the features.

## Contents

1. [Project Structure](#project-structure)
2. [Getting Started](#getting-started)
3. [Usage](#usage)
4. [Model Information](#model-information)
5. [Results](#results)
   
## Project Structure

- **data/**: Contains raw and processed data files.
- **notebooks/**: Jupyter notebooks for exploration and model development.
- **src/**: Source code for data preprocessing, training, and prediction.
- **application.py**: Script to run the web application.
- **requirements.txt**: List of dependencies.
- **README.md**: Project documentation.
- **LICENSE**: License information.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- pip

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/prashastigupta2004/Diamond-price-prediction.git
    cd Diamond-price-prediction
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preprocessing

Prepare the data for training:
```bash
python src/data_preprocessing.py

Train the machine learning model:
python src/train_model.py

Predict the price of diamonds:
python src/predict.py

Running the Web Application
python application.py

## Results
The model's performance on the test set:

R-squared: 93.68 %
MAE: 674.0
RMSE: 1013.90
