import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

def load_data():
    fpath='/home/adityapandey/Downloads/project-sem5/Data Sets/'
    indian_students = pd.read_csv(fpath+"IndianStudentsAbroad.csv")
    cost_living = pd.read_csv(fpath+"Cost_of_Living_Index_by_Country_2024.csv")
    tuition = pd.read_csv(fpath+"International_Education_Costs.csv")
    reputation = pd.read_csv(fpath+"QS World University Rankings 2025 (Top global universities).csv",encoding='latin1')
    return indian_students, cost_living, tuition, reputation

indian_students, cost_living, tuition, reputation = load_data()
indian_students.drop(columns=['index'], inplace=True)
print(indian_students.head())

order = indian_students.sort_values('No of Indian Students')['Country'].unique().tolist()
encoder = OrdinalEncoder(categories=[order])
indian_students['Country_Encoded'] = encoder.fit_transform(indian_students[['Country']])

num_cols = ['Country_Encoded','No of Indian Students','Percentage','Unnamed: 3']
scaler = MinMaxScaler()
indian_students[num_cols] = scaler.fit_transform(indian_students[num_cols])
indian_students.drop(columns=['Unnamed: 3'], inplace=True)
print(indian_students)

tuition.columns = tuition.columns.str.strip().str.lower().str.replace(" ", "_")

tuition["total_cost"] = (tuition["tuition_usd"] * tuition["duration_years"]) + \
                   (tuition["rent_usd"] *tuition["duration_years"]) + \
                tuition["visa_fee_usd"] +tuition["insurance_usd"]
nums_col2=['exchange_rate','living_cost_index','tuition_usd','duration_years','rent_usd','visa_fee_usd','insurance_usd','total_cost']  
tuition[nums_col2]=scaler.fit_transform(tuition[nums_col2]) 
print(tuition.head())

cost_living.columns = cost_living.columns.str.strip().str.lower().str.replace(" ", "_")
nums_col3=['cost_of_living_index','rent_index','cost_of_living_plus_rent_index','groceries_index','restaurant_price_index','local_purchasing_power_index']
cost_living[nums_col3]=scaler.fit_transform(cost_living[nums_col3]) 
print(cost_living.head())


reputation.columns = reputation.columns.str.strip()
nums_col4=['RANK_2024','Region','SIZE','FOCUS','RES.','STATUS','Academic_Reputation_Score', 'Academic_Reputation_Rank', 'Employer_Reputation_Score', 'Employer_Reputation_Rank', 'Faculty_Student_Score', 'Faculty_Student_Rank', 'Citations_per_Faculty_Score', 'Citations_per_Faculty_Rank', 'International_Faculty_Score', 'International_Faculty_Rank', 'International_Students_Score', 'International_Students_Rank', 'International_Research_Network_Score', 'International_Research_Network_Rank', 'Employment_Outcomes_Score', 'Employment_Outcomes_Rank', 'Sustainability_Score', 'Sustainability_Rank']
reputation.drop(columns=nums_col4, inplace=True)
nums_col5=['Overall_Score']
# Ensure numeric dtype, then scale
reputation[nums_col5] = scaler.fit_transform(
    reputation[nums_col5].apply(pd.to_numeric, errors='coerce')
)
print(reputation.head())
print(reputation.columns.tolist())

reputation.columns = reputation.columns.str.strip().str.lower().str.replace(" ", "_")
indian_students.columns = indian_students.columns.str.strip().str.lower().str.replace(" ", "_")
df = tuition.merge(cost_living, on="country", how="left") \
            .merge(indian_students, on="country", how="left") \
            .merge(reputation, left_on="university", right_on="institution_name", how="left")
# Example: Drop rows where 'rank' is NaN (missing)
df_cleaned = df.dropna(subset=['rank_2025'])

print(df_cleaned.head())
df_cleaned.to_csv('/home/adityapandey/Downloads/project-sem5/Data Sets/merged_indian_students_abroad.csv', index=False)

