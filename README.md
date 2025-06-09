# California Housing Price Prediction Project

## Table of Contents

* [1. Overview](#1-overview)
* [2. Project Goal](#2-project-goal)
* [3. Repository Structure](#3-repository-structure)
* [4. Key Features and Techniques](#4-key-features-and-techniques)
* [5. How to Run the Notebook](#5-how-to-run-the-notebook)
* [6. Results and Conclusion](#6-results-and-conclusion)
* [7. Blog Post](#7-blog-post)
* [8. Contributing](#8-contributing)
* [9. License](#9-license)

---

## 1. Overview

This repository contains a comprehensive machine learning project focused on predicting median housing prices in California. The analysis leverages the well-known California Housing dataset, accessible via scikit-learn. The project demonstrates a full data science pipeline, from initial data loading and exploratory analysis to advanced preprocessing, feature engineering, model training, and evaluation.

## 2. Project Goal

The primary objective of this project is to build a robust and accurate predictive model for housing prices. This involves:
* Understanding the underlying patterns and relationships within the housing data.
* Handling data quality issues, such as outliers, to improve model performance.
* Creating new, more informative features from existing data.
* Training and comparing various regression algorithms.
* Identifying the best-performing model for the task.

## 3. Repository Structure

```
.
├── notebook/
|    |── housing_price_prediction.ipynb
|── docs/
|    ├── blog.md
|    ├── index.md
├── README.md

```

* `housing_price_prediction.ipynb`: The core Jupyter Notebook containing all the analytical steps, code, and detailed explanations.
* `blog.md`: A summary blog post that provides a narrative walkthrough of the project, its methodology, and key findings.
* `index.md`: The main landing page for the GitHub Pages site associated with this repository.
* `README.md`: This file, providing an overview of the project.
* `assets/images/housing_prediction/`: Directory to store all image outputs from the notebook, embedded in the `blog.md` and potentially here.

## 4. Key Features and Techniques

* **Data Loading & Splitting**: Using `sklearn.datasets` and `train_test_split`.
* **Exploratory Data Analysis (EDA)**:
    * Visualizing feature distributions with histograms.
    * Analyzing geospatial patterns of house values using scatter plots.
    * Understanding relationships between features via correlation matrices and heatmaps.
* **Data Preprocessing**:
    * **Outlier Handling**: Implementing a quantile-based clipping strategy (5th and 95th percentiles) to manage extreme values, crucial for linear models.
    * **Feature Engineering**: Creating new insightful features like `AveBedrms_AveRooms_ratio` and `AveRooms_AveOccup_ratio`.
    * **Feature Scaling**: Applying `StandardScaler` to normalize feature magnitudes, improving convergence and performance for distance-based and gradient-based models.
* **Model Training & Evaluation**:
    * Implementation and comparison of various regression models:
        * Linear Regression
        * Decision Tree Regressor (with `GridSearchCV` for hyperparameter tuning)
        * Random Forest Regressor (with `GridSearchCV` for hyperparameter tuning)
        * XGBoost Regressor
    * Evaluation metrics used: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared ($R^2$).

## 5. How to Run the Notebook

To run this project locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
3.  **Install the required libraries:**
    ```bash
    pip install pandas numpy scikit-learn seaborn matplotlib xgboost jupyter
    ```
4.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook housing_price_prediction.ipynb
    ```
    This will open the notebook in your web browser, where you can run all cells.

## 6. Results and Conclusion

After training and evaluating multiple models, the **XGBoost Regressor** emerged as the top performer, achieving an **R-squared ($R^2$) score of 0.83**. This indicates that the model explains 83% of the variance in the median house values, showcasing strong predictive capability. The visualization of actual vs. predicted values confirms this, with points closely aligning to a 45-degree line.

## 7. Blog Post

For a more narrative and less code-intensive walkthrough of this project, check out the accompanying [blog post](blog.md).

## 8. Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or find any bugs, please open an issue or submit a pull request.

## 9. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: You should create a `LICENSE` file in your repository with the MIT License text if you choose this option.)*
```