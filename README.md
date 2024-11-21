# Titanic Data Analysis Project

## Project Overview

The **Titanic Data Analysis Project** aims to predict survival outcomes for passengers aboard the Titanic using machine learning techniques. By analyzing historical data such as age, class, gender, and fare, the project provides insights into the factors that influenced survival rates. The analysis leverages popular Python libraries such as Pandas, Matplotlib, Seaborn, and Scikit-learn to explore and visualize data, build predictive models, and evaluate their performance.

## Key Features

- **Data Exploration and Preprocessing:** Cleaned and transformed raw Titanic dataset, handling missing values, encoding categorical variables, and scaling features.
- **Data Visualization:** Used Matplotlib and Seaborn to create visualizations such as bar charts, histograms, and correlation matrices to identify key patterns and trends.
- **Modeling:** Applied various machine learning algorithms, including Logistic Regression, Random Forest, and Support Vector Machines (SVM), to predict survival.
- **Model Evaluation:** Assessed model performance using metrics like accuracy, precision, recall, F1 score, and the confusion matrix.
- **Feature Engineering:** Extracted and created new features to improve model performance, including titles (e.g., Mr., Mrs.) and family size (combining siblings/spouses and parents/children).

## Getting Started

To run the project locally, follow these steps:

1. **Clone the repository:**
   ```
   git clone https://github.com/your-username/titanic-data-analysis.git
   ```

2. **Install required libraries:**
   ```
   pip install -r requirements.txt
   ```

3. **Run the Jupyter notebook:**
   - Open the `Titanic_Data_Analysis.ipynb` file in Jupyter Notebook or Google Colab.
   - Execute the notebook cells sequentially to perform data analysis and model building.

## Dataset

The dataset used for this analysis is the **Titanic: Machine Learning from Disaster** dataset, available from [Kaggle](https://www.kaggle.com/c/titanic/data). It contains the following features:

- **PassengerId:** Unique identifier for each passenger.
- **Pclass:** Passenger class (1st, 2nd, 3rd).
- **Name:** Name of the passenger.
- **Sex:** Gender of the passenger.
- **Age:** Age of the passenger.
- **SibSp:** Number of siblings/spouses aboard the Titanic.
- **Parch:** Number of parents/children aboard the Titanic.
- **Fare:** Fare paid for the ticket.
- **Embarked:** Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton).

## Results

- **Key Findings:** Gender, class, and age significantly influenced the survival rates.
- **Best Performing Model:** Random Forest Classifier achieved the highest accuracy in predicting survival.

## Conclusion

This project provides a comprehensive analysis of Titanic passenger data, offering valuable insights into survival factors. By applying machine learning techniques, we were able to build predictive models with strong performance, shedding light on the characteristics that influenced survival on the Titanic.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

Feel free to modify this README based on your specific findings and contributions!
