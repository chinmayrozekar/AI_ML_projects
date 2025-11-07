# FoodHub Data Science — Project README

## Project Summary
FoodHub is an online food-ordering platform. This project analyzes FoodHub order data to answer business questions about customer behavior, operational performance (preparation & delivery times), cuisine preferences, and rating patterns. The goal is to provide concise, recruiter-friendly insights, highlight technologies used, and point to the code for reproducibility.

## Business Context & Objective
- Business context: rapid growth in online food delivery; FoodHub needs to optimize operations and marketing.
- Objective: analyze order patterns to: reduce delivery/prep delays, improve customer satisfaction, and identify high-value segments to target for promotions.

## Data
- Source: `foodhub_order.csv` (included in this project folder)
- Rows: 1898 | Columns: 9
- Key columns: `order_id`, `customer_id`, `restaurant_name`, `cuisine_type`, `cost_of_the_order`, `day_of_the_week`, `rating`, `food_preparation_time`, `delivery_time`

## What I did (approach)
1. Data ingestion and sanity checks (duplicates, missing values)
2. Data cleaning and type conversions (convert 'Not given' ratings to NaN)
3. Exploratory Data Analysis (univariate & bivariate): distributions, boxplots, histograms, correlation checks
4. Simple aggregations to identify peak days, cuisine popularity, and long-prep restaurants
5. Business-focused conclusions and recommendations

## Key Findings & Conclusions
- Data quality: No duplicate rows; 736 ratings are recorded as 'Not given' and are treated as missing.
- Preparation times: Min = 20 min, Avg ≈ 27.4 min, Max = 35 min — indicates restaurants are fairly consistent, but those above the mean are candidates for operational review.
- Delivery times: Generally under 30 minutes for majority of orders — overall delivery performance is acceptable.
- Ratings: A significant fraction of orders have no rating (736 of 1898). Among rated orders, most ratings are positive (many 5s).
- Order volume: Weekends have higher order volume and show different cuisine mixes — plan promotions and staffing around weekend peaks.

## Recommendations (business actionable)
- Targeted operations: Audit restaurants with prep times > 30 minutes to reduce variability (training, menu simplification, kitchen workflow).
- Rating capture: Add in-app nudges or incentives (coupon on next order) to increase rating completion — will improve feedback and model training for sentiment analysis.
- Weekend campaigns: Run promotions for high-margin cuisines on weekends and provision more delivery partners during weekend peaks.
- Monitor KPIs: Track daily/weekly averages of `food_preparation_time` and `delivery_time` per restaurant and set SLAs.

## Technologies & Libraries
- Environment: Jupyter Notebook / Google Colab
- Core libraries: Python, Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Reproducibility: notebook included (`Chinmay_Learner_Notebook_Full_Code.ipynb`) and an HTML render is present

## Where the code is
- Notebook (analysis + code): `Chinmay_Learner_Notebook_Full_Code.ipynb`
  - Raw notebook: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_1_Food_Hub_DataScience/Chinmay_Learner_Notebook_Full_Code.ipynb
  - Rendered HTML: https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_1_Food_Hub_DataScience/Chinmay_Learner_Notebook_Full_Code.html
- Data file: `foodhub_order.csv` — https://github.com/chinmayrozekar/AI_ML_projects/blob/main/Project_1_Food_Hub_DataScience/foodhub_order.csv

## Quick code snippets (from notebook)
```python
# load data
import pandas as pd
data = pd.read_csv('foodhub_order.csv', na_values=['Not given'])

# shape and datatypes
print(data.shape)
print(data.dtypes)

# basic prep time stats
min_time = data['food_preparation_time'].min()
avg_time = data['food_preparation_time'].mean()
max_time = data['food_preparation_time'].max()
print(min_time, avg_time, max_time)
```

## How this looks in a recruiter portfolio
- Short summary and clear business recommendations upfront
- Code + notebook link for reproducibility
- Key KPIs and recommended next steps clearly listed

---

*Author: Chinmay Rozekar*