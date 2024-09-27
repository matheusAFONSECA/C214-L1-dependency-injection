# C214-L1-dependency-injection

Repository dedicated to applying dependency injection in a Python code - lab activity for C214.

## Index
1. [Overview](#overview)
2. [How to Run Via Terminal](#how-to-run-via-terminal)
3. [Explanation of Injection](#explanation-of-injection)
4. [Project Structure](#project-structure)
5. [Author](#author)

## Overview

This project is a simple implementation of dependency injection using Python. The code is modularized and contains a machine learning model training pipeline. The RandomForest and LogisticRegression models are dynamically injected into the training pipeline, along with the injection of a data preprocessor (StandardScaler).

## How to Run Via Terminal

1. **Create and activate a virtual environment:**

    - **Using commands via terminal**

    1. **Creating a virtual enviroment**
        ```bash
        python -m venv C214venv
        ```

    2. **Activate the virtual environment:**

        - On Windows:

        ```bash
        C214venv\Scripts\activate
        ```

        - On Linux/MacOS:

        ```bash
        source venv/bin/activate
        ```
    
    - **Using a script**

        ```bash
        ./scripts/create_and_activate_venv.sh
        ```
    
3. **Install the project dependencies:**

    ```bash 
    pip install -r requirements.txt
    ```

4. **Run the main code:**

    ```bash 
    python main.py
    ```

