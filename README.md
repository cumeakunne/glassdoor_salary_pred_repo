### Glassdoor Data Analyst Estimator: Project Overview
- Created: A tool that predicts salary estimates
- Purpose: To support firm recruiters in making competitive offers and data analysts in negotiating their salaries.
- Performance: Mean Absolute Error of roughly $11,000
- Best Model: Ensemble of Random Forest Regressor(50%) and Lasso Regression(50%). Paramter tuned using Gridsearch CV.
- Data: Scraped 500 job postings off Glassdoor.com using selenium in python
- Features: Engineered features from job description details including job title, seniority, location, company, industry, and tools used like excel and tableau.
- Wrapped model in Flask API

### Resources Used
- Language: Python Version - 3.7
- IDEs: Spyder and Jupyter Notebook
- Packages: Flask, JSON, Matplotlib, Numpy, Pickle, Pandas, Seaborn, Selenium 
- For Web Framework: pip install -r requirements.txt
- Project Inspiration and Walkthrough: https://github.com/PlayingNumbers/ds_salary_proj
- Webscraper Github: https://github.com/arapfaik/scraping-glassdoor-selenium
- Webscraper Blog: https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905
- Deployment with Flask API: https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

### Data Collection
Tweaked the github webscraper by arapfaik to scrape 500 job postings off glassdoor.com.
With each post we extracted the following features of data;
- Job Title  "The name of the job"
- Salary Estimate "The estimated salary range the for the job"
- Job Description "All the content within the section introducing the company and describing requeirement skills and duties the ideal prospective candidate should have."
- Rating   "The companies rating as an employer"
- Company Name "The name of the company"
- Location "The city and state the company is located"
- Headquarters "The city and state the HQ is located" 
- Size "A range of the number of employees the company has"
- Founded "The year the company was founded"
- Type of ownership "Publically traded, privately owned, and other types of ownership
- Industry "The industry the company belongs in"
- Sector "The secotr the company belongs in"
- Revenue "The range of annual revenue the company brings in"
- Competitors "A list of competitors the copmany has"

### Data Cleaning and Feature Engineering
The raw data scraped from Glassdoor needed to be cleaned for use in our analysis. In addition, some features were engineered to boost our models performance.
The following lists the actions taken during this step;
1. Remove NULLS: The records with null values from 'Salary Estimate'
2. Salary parsing: The numeric estimates were parsed from the 'Salary Estimate' string. 
  - Added Feature: An average salary estimate 'avg_salary'  was computed from the min and max salary estimates
  - This 'avg_salary' feature became the outcome/target variable of interest for our analysis and modeling efforts
3. Fixed spliced data: 'Company Name' included the 'Rating' value, removed
4. Seperated Location contents into a State feature ('job_state') and city feature ('job_city')
5. Same Location Dummy: Created a dummy variable capturing when the job was at the headquarters 
  - 'same_location' has 1 for at headquarters and 0 for not at headquarters
6. Company age: used 'Founded' year to compute the age of company from this year '2020'
7. Parsed Job Description: Created dummy variables capturing the presence of analysis tools requested in 'Job Description'
  - These include 'python_yn', 'R_yn', tableau_yn', 'excel_yn', SAS_yn'. The _yn stands for whether the tool is present ( yes or no) which corresponds to values of 1 or 0 for    each dummy variables

### Data Exploration
Once the data was cleaned and some preliminary features were created, it was now time to explore the data.
- I first used .describe() to see the basic descriptive stats for central tendency and the overall dispersion of each fature.
- Each continuous variables was explored using histograms, box-plots, and a correlation matrix/heatmap
- Then the categorical variables using bar plots
- Once I had a better sense of the distribution of each feature, I used pivot tables to compare them with the outcome variable 'avg_salary'
![EDA Barplot!](https://github.com/cumeakunne/glassdoor_salary_pred_repo/blob/master/eda_barplot%20-%20Copy.jpg)
![EDA Heatmap!](https://github.com/cumeakunne/glassdoor_salary_pred_repo/blob/master/eda_heatmap.jpg)
![EDA_Pivot Table!](https://github.com/cumeakunne/glassdoor_salary_pred_repo/blob/master/eda_pivot.jpg)

### Data Model Building
From the exploration, I selected the features to include in the training models. Split the data set into a training set (80%) and test set (20%).
Used pandas get_dummies() function to create dummy variables for all my categorical variables. This exponentially increased dimension size from 14 to 406.
The evaluation method was chosen to be Mean Absolute Error for ease of interpretation, can tell how many $ our estimate is off directly from MAE.

Three Models were built:
1. Multiple Linear Regression : Served as Baseline Model
  - Performance: score = 247630661127.73917 | excessively off due to sparsity data, with no meaningful interpretation
2. Lasso Regression: A regularized regression was expected to perform better with such a small data set
  - Performance: #Score = 11.47 | Interprets as 'the predicted estimate off by $11,470 on average'
3. Random Forest Regressor: This model aslo performs well with small data
  - Performance: #Score = 11.30 | Interprets as 'the predicted estimate off by $11,300 on average'

Ensemblling: Combining Lasso and the Random Forest model produced a slighly better model. 
  - Best Score = #11.10 at a 50% weight per component model | Predicted estimate is off by $11,100 on average.

### Data Model Deployment
Built a Flask API virtual environment on my local client following the Ken Jee walkthrough and the article above in the Resources Section. The API takes in a list of parameters for a job listing, runs it through the model, and returns a salary estimate.
