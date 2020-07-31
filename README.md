### Glassdoor Data Analyst Estimator: Project Overview
- Created: A tool that predicts salary estimates
- Purpose: To support firm recruiters in making competitive offers and data analysts in negotiating their salaries.
- Performance: Mean Absolute Error of roughly $11,000
- Best Model: Ensemble of Random Forest Regressor(50%) and Lasso Regression(50%). Paramter tuned using Gridsearch CV.
- Data: Scraped over 500 job postings off Glassdoor.com using selenium in python
- Features: Engineered features from job description details including job title, seniority, location, company, industry, and tools used like excel and tableau.
- Wrapped model in Flask API

## Resources Used
Language: Python Version - 3.7
IDEs: Spyder and Jupyter Notebook
Packages: Flask, JSON, Matplotlib, Numpy, Pickle, Pandas, Seaborn, Selenium 
For Web Framework: pip install -r requirements.txt
Project Inspiration and Walkthrough: https://github.com/PlayingNumbers/ds_salary_proj
Webscraper Github: https://github.com/arapfaik/scraping-glassdoor-selenium
Webscraper Blog: https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905
Deployment with Flask API: https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

### 
