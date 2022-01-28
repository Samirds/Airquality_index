import pandas as pd
import numpy as np

# ################## Import Dataset #################################

# ----> AirQuality Dataset--------->
aq_df = pd.read_csv("city_day.csv.zip", parse_dates=["Date"])
aq_df = aq_df[aq_df["City"] == "Delhi"]
aq_df = aq_df[aq_df["Date"] <= "2020-05-12"]
aq_df = aq_df.reset_index()
aq_df.drop("index", axis=1, inplace=True)

# ----> Temperature Dataset--------->

temp_df = pd.read_csv("delhi_temp.csv", parse_dates=[["month", "day", "year"]])
temp_df = temp_df.rename(columns={"month_day_year": "date"})
temp_df = temp_df[temp_df["date"] >= "2015-01-01"]

# ---------> Confirming That Both dataset has same date ----->

temp_df.drop(7669, axis=0, inplace=True)
temp_df = temp_df.reset_index()
temp_df.drop("index", axis=1, inplace=True)

# ---------> Combine Datasets -------------------------->

combine_df = pd.concat([aq_df, temp_df["temp"]], axis=1)

# -----------> Filling Up Missing Values ------>

combine_df_nan_fr = combine_df.fillna(method="ffill")

# --------------------> Fixed outliers in temp ---------->

combine_df_nan_fr["temp"] = np.where(combine_df_nan_fr["temp"] == -99, np.NaN, combine_df_nan_fr["temp"])
combine_df_nan_fr = combine_df_nan_fr.set_index("Date")
combine_df_nan_fr = combine_df_nan_fr.interpolate(method="time")

# ---------------> Dropping unnecessary columns ------------->

combine_df_nan_fr = combine_df_nan_fr.drop(["City", "AQI_Bucket"], axis=1)

# -----------------> Making a rolling dataset -------------------->

for i in combine_df_nan_fr.columns:
    combine_df_nan_fr["roll_" + i] = combine_df_nan_fr[i].rolling(window=2).mean()

combine_df_nan_fr_roll = combine_df_nan_fr[['roll_PM2.5',
                                            'roll_PM10', 'roll_NO', 'roll_NO2', 'roll_NOx', 'roll_NH3', 'roll_CO',
                                            'roll_SO2', 'roll_O3', 'roll_Benzene', 'roll_Toluene', 'roll_Xylene',
                                            'roll_AQI', 'roll_temp']]

combine_df_nan_fr_roll.dropna(inplace=True)

# ---------------- df --------------------->

df = combine_df_nan_fr.drop(['roll_PM2.5',
                             'roll_PM10', 'roll_NO', 'roll_NO2', 'roll_NOx', 'roll_NH3', 'roll_CO',
                             'roll_SO2', 'roll_O3', 'roll_Benzene', 'roll_Toluene', 'roll_Xylene',
                             'roll_AQI', 'roll_temp'], axis=1)

# ------------------- Check for Stationarity ------------------->

# for i in df.columns:
#     result = adfuller(df[i])
#
#     print("\n\n\nFeature Name-->{}\n".format(i))
#     print(f'Test Stats: {result[0]}')
#     print(f"P-Value: {result[1]}")
#     print(f"Critical Value: {result[4]}")
#
#     # if result[1] > 0.05:
#     #     print("\nSeries is not Stationary")
#     # else:
#     #     print("\nseries is stationary")

# test stats always prefer to be good if it's lower

# -------------------- Make dataset Stationary --------------------->

# We are removing the seasonality by substracting by 12 month
df["temp_st"] = df["temp"] - df["temp"].shift(12)
df = df.dropna()

# ------------------------ Checking Adf test again ----------------->

# result = adfuller(df["temp_st"])
#
# print(f'Test Stats: {result[0]}')
# print(f"P-Value: {result[1]}")
# print(f"Critical Value: {result[4]}")
#
# if result[1] > 0.05:
#     print("\nSeries is not Stationary")
# else:
#     print("\nseries is stationary")

# -----------------------------  Check for Granger Causality Test ----------------->

# df = df[['AQI', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
#          'Benzene', 'Toluene', 'Xylene', 'temp', 'temp_st']]
#
# max_lags=8
# y='AQI'
#
# for i in range(len(df.columns)-1):
#   results=grangercausalitytests(df[[y,df.columns[i+1]]], max_lags, verbose=False)
#   p_values=[round(results[i+1][0]['ssr_ftest'][1],4) for i in range(max_lags)]
#   print('Column - {} : P_Values - {}'.format(df.columns[i+1],p_values))

# ------------------- Final DF ------------------------------------>

final_df = df[["PM2.5", "PM10", "NOx", "temp_st", "AQI"]]
final_df = final_df.reset_index()

final_df = final_df.rename({"Date": "ds",
                            "AQI": "y"}, axis='columns')

print(final_df.head())