
import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler

def data_report(file_name):
    fpath='/home/adityapandey/Downloads/project-sem5/Data Sets/'
    from ydata_profiling import ProfileReport
    df = pd.read_csv(fpath+file_name)
    profile = ProfileReport(df, title="final report1", explorative=True)
    profile.to_file("final report1.html")

def load_data():
    fpath='/home/adityapandey/Downloads/project-sem5/Data Sets/'
    indian_students = pd.read_csv(fpath+"IndianStudentsAbroad.csv")
    cost_living = pd.read_csv(fpath+"Cost_of_Living_Index_by_Country_2024.csv")
    tuition = pd.read_csv(fpath+"International_Education_Costs.csv")
    reputation = pd.read_csv(fpath+"QS World University Rankings 2025 (Top global universities).csv",encoding='latin1')
    return indian_students, cost_living, tuition, reputation

indian_students, cost_living, tuition, reputation = load_data()
indian_students.drop(columns=['index'], inplace=True)
indian_students.loc[indian_students['Country']=='US(2015-16)','Country']='United States'
indian_students.loc[indian_students['Country']=='UK(2015-16)','Country']='United Kingdom'
tuition.columns = tuition.columns.str.strip().str.lower().str.replace(" ", "_")
tuition.loc[tuition['country']=='USA','country']='United States'
tuition.loc[tuition['country']=='UK','country']='United Kingdom'
print(tuition.head())
tuition["total_cost"] = (tuition["tuition_usd"] * tuition["duration_years"]) + \
                   (tuition["rent_usd"] *tuition["duration_years"]) + \
                tuition["visa_fee_usd"] +tuition["insurance_usd"]
cost_living.columns = cost_living.columns.str.strip().str.lower().str.replace(" ", "_")
print(cost_living.head())
reputation.columns = reputation.columns.str.strip()
print(reputation.head())
print(reputation.columns.tolist())

reputation.columns = reputation.columns.str.strip().str.lower().str.replace(" ", "_")
indian_students.columns = indian_students.columns.str.strip().str.lower().str.replace(" ", "_")
df = tuition.merge(cost_living, on="country", how="left") \
            .merge(indian_students, on="country", how="left") \
            .merge(reputation, left_on="university", right_on="institution_name", how="left")
# Example: Drop rows where 'rank' is NaN (missing)
df_cleaned = df.dropna(subset=['rank_2025'])
df_cleaned = df_cleaned.drop(df_cleaned.loc[df_cleaned['country'] == 'UAE'].index)
df_cleaned = df_cleaned.drop(df_cleaned.loc[df_cleaned['country'] == 'South Korea'].index)
df_cleaned=df_cleaned.drop(columns=['institution_name','location'])
print(df_cleaned.head())
df_cleaned=df_cleaned.drop(columns=['unnamed:_3','rank_2024','region','size','focus','res.','status'])
fpath='/home/adityapandey/Downloads/project-sem5/Data Sets/'
df_cleaned.to_csv(fpath+"merged_indian_students_abroad.csv", index=False)


