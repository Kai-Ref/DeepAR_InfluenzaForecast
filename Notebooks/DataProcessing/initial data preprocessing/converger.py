import pandas as pd
import os
from zipfile import ZipFile 
from epiweeks import Week
import sys
#print(sys.path)
os.chdir(sys.path[0])
complete_df=pd.DataFrame()
for root, dirs, files in os.walk('Dateien/Meningococken'):
    for filename in files:
        if ".zip" in filename:
            path=os.path.join(root, filename)
            with ZipFile(path) as myzip:
                with myzip.open('Data.csv') as myfile:
                    df1=pd.read_csv(myfile,encoding="utf-16",sep = "\t",header=[1]).fillna(0).rename(columns={'Unnamed: 0':'week'})
                    df1['year']=int(filename[-8:-4])
                    print(filename[-8:-4])
                    df1['age_group']="00+"
                    df=pd.DataFrame()
                    df['week']=df1.week
                    df['year']=df1.year
                    df1['date']=df.apply(lambda x: Week(x.year, x.week, system='iso').enddate(), axis=1)
                    #df1['date'] = df1.apply(lambda x: Week(x.year, x.week, system='iso').enddate(), axis=1)
                    df1=df1.melt(id_vars=['year','date','week','age_group'], value_vars=df1.columns.difference(['year','date','week']).to_list()).rename(columns={'variable':'location'})
                    complete_df=pd.concat([complete_df,df1])
if "Unbekannt" in complete_df.location.unique():
    complete_df = complete_df[complete_df.location != "Unbekannt"]
complete_df.to_csv('meningococcal.csv')
print(complete_df)