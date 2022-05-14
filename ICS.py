import pandas as pd
import datetime as dt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
from matplotlib import pyplot as plt

sensor_data = pd.read_csv(r"D:\ICS\sensor.csv", encoding = "ISO-8859-1")
manual_data_raw = pd.read_csv(r"D:\ICS\log21.csv", encoding = "ISO-8859-1")

#change column names to match sensor.csv
manual_data = manual_data_raw.rename(columns={"Date:" : "Date", "Time:" : "Time", "pH:" : "pH"})

#Add AM to times where AM/PM value missing
manual_data['Time'] = pd.to_datetime(manual_data['Time'], format='%I:%M %p', errors='coerce').fillna(pd.to_datetime(manual_data['Time'], format='%I:%M', errors='coerce'))

#Convert AM/PM values to 24 hour values
manual_data['Time'] = pd.to_datetime(manual_data['Time'], format='%I:%M %p').dt.strftime('%H:%M')

#forward-fill missing values in 'Time' column
manual_data['Time'].fillna(method='ffill', inplace=True)

#combine 'Date' and 'Time' into 'DateTime' column for manual_data dataframe
manual_data["DateTime"] = pd.to_datetime(manual_data['Date'] + manual_data['Time'], format='%m/%d/%Y%H:%M')

#Drop seconds from 'Time' column in sensor_data to match manual_data
sensor_data['Time'] = pd.to_datetime(sensor_data['Time'], format='%H:%M:%S').dt.strftime('%H:%M')

#combine 'Date' and 'Time' into 'DateTime' column for manual_data dataframe
sensor_data["DateTime"] = pd.to_datetime(sensor_data['Date'] + sensor_data['Time'], errors='ignore', format='%m/%d/%Y%H:%M')

#Set indexes for both dataframes to 'DateTime'
left = sensor_data.set_index(['DateTime'])
right = manual_data.set_index(['DateTime'])

#Merge the two dataframes
water_data_combined = left.join(right, lsuffix=' Sensor', rsuffix=' Manual')
#water_data_combined=pd.merge(left,right, how='inner', left_index=True, right_index=True, lsuffix=' Sensor', rsuffix=' Manual')

water_data_combined.reset_index()

#Remove rows with missing values for either 'pH Manual' or 'pH Sensor'
water_data_combined = water_data_combined[water_data_combined['pH Manual'].notna()]
water_data_combined = water_data_combined[water_data_combined['pH Sensor'].notna()]

#Remove columns except 'pH Sensor' and 'pH Manual'
water_data_aov = water_data_combined.loc[:, water_data_combined.columns.intersection(['pH Sensor','pH Manual'])]

#Rename columns because I couldn't figure out how to get them to work with a space
water_data_aov = water_data_aov.rename(columns={"pH Sensor" : "Sensor", "pH Manual" : "Manual"})

#change datatype of 'Sensor' to float from object
water_data_aov['Sensor'] = water_data_aov['Sensor'].astype(float, errors = 'raise')

#ordinary least squares
mod = ols('Sensor ~ Manual',data=water_data_aov).fit()
aov_table = sm.stats.anova_lm(mod, typ=2)
print(aov_table)

p_S = stats.shapiro(mod.resid)
print(p_S)

res = mod.resid
fig = sm.qqplot(res, line= 's')
plt.title("Q-Q Plot for ANOVA Assumption of Normality")
plt.show()

