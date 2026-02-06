# Titanic-Survival-Prediction
This project analyzes the classic Titanic dataset to predict passenger survival based on demographic and trip-related features. It demonstrates a data science workflow, including data cleaning, exploratory analysis, and machine learning.

Key Insights
Gender: Female passengers had a significantly higher survival rate than male passengers.
Socioeconomic Status: Passengers in higher classes (1st class) were more likely to survive than those in lower classes.
Age: Missing values were handled using median imputation to maintain dataset integrity.

Technologies Used
Python: Core programming.
Pandas: Data manipulation and cleaning.
Seaborn & Matplotlib: Data visualization.
Scikit-Learn: Machine learning (Logistic Regression) and model evaluation.

How It Works
Data Cleaning: Dropped the deck column due to excessive missing values and imputed missing values for age and embark_town.
Feature Engineering: Converted categorical variables (sex, class) into numeric formats using one-hot encoding.
Modeling: Implemented a Logistic Regression model using an 80/20 train-test split.
Evaluation: Achieved classification accuracy based on the testing set.

Visualizations
The script generates bar plots showing survival rates across different categories to validate the correlation between features and survival outcomes.
