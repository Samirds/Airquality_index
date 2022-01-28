# ############ Import Libraries ################################

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from fbprophet import Prophet
import pickle

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

# --------------------- Split Dataset ------------------------------------>

train = final_df[(final_df['ds'] >= '2015-01-13') & (final_df['ds'] <= '2020-04-01')]
test = final_df[(final_df['ds'] > '2020-04-01')]

# ------------------------  FbPropher for Multivariate --------------------->


model = Prophet(
    growth='linear',
    # interval_width=0.80,
    seasonality_mode='multiplicative',
    daily_seasonality=False,
    weekly_seasonality=False,
    yearly_seasonality=True,
).add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=50,  # 25
    prior_scale=20
).add_seasonality(
    name='daily',
    period=1,
    fourier_order=70,  # 25
    prior_scale=20
).add_seasonality(
    name='weekly',
    period=7,
    fourier_order=50,
    prior_scale=60
).add_seasonality(
    name='yearly',
    period=365.25,
    fourier_order=30)

# ----------------> Added Multivariate in series------------------------------->
model.add_regressor('PM2.5', standardize=True, mode='multiplicative')
model.add_regressor('PM10', standardize=True, mode='multiplicative')
model.add_regressor('NOx', standardize=True, mode='multiplicative')
model.add_regressor('temp_st', standardize=True, mode='multiplicative')

model.fit(train)

# ---------------- Prediction ----------------------------------->

future = model.make_future_dataframe(periods=41)

future['PM2.5'] = final_df['PM2.5']
future['PM10'] = final_df['PM10']
future['NOx'] = final_df['NOx']
future['temp_st'] = final_df['temp_st']

forecast = model.predict(future)

# print(forecast[["ds", "yhat"]].tail())
# # print("\n\n")
# # print(test[["ds", "y"]].tail())

# # -------------------------------> Detect and Removal of Anomaly--------------------------->
#
# forecasted = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
# forecasted['anomaly'] = 0
# forecasted['fact'] = final_df['y']
# forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
# forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1
#
#
# new_forecasted = forecasted[['ds', 'yhat', 'yhat_lower', 'yhat_upper', "anomaly", "fact"]]
# new_forecasted['new_fact'] =new_forecasted["fact"]
# new_forecasted.loc[new_forecasted['anomaly'] == 1, 'new_fact'] = new_forecasted["yhat_upper"]
# new_forecasted.loc[new_forecasted['anomaly'] == -1, 'new_fact'] = new_forecasted["yhat_lower"]
#
# # ---------------------------> Anomaly Modified Dataset -------------------------------->
#
# anml_fr_df = new_forecasted[["ds", "yhat"]]
# anml_fr_df = anml_fr_df.rename(columns={"yhat" : "y"})
#
# # --------------------------> New Model ----------------------------->
#
# new_model = Prophet(
#     growth='linear',
#     # interval_width=0.80,
#     seasonality_mode='multiplicative',
#     daily_seasonality=False,
#     weekly_seasonality=False,
#     yearly_seasonality=True,
# ).add_seasonality(
#     name='monthly',
#     period=30.5,
#     fourier_order=50,  # 25
#     prior_scale=20
# ).add_seasonality(
#     name='daily',
#     period=1,
#     fourier_order=70,  # 25
#     prior_scale=20
# ).add_seasonality(
#     name='weekly',
#     period=7,
#     fourier_order=50,
#     prior_scale=60
# ).add_seasonality(
#     name='yearly',
#     period=365.25,
#     fourier_order=30)
#
#
# # --------------------- Split  Anomaly Free Dataset ------------------------------------>
#
# anml_fr_train = anml_fr_df[(anml_fr_df['ds'] >= '2015-01-13') & (anml_fr_df['ds'] <= '2020-04-01')]
# anml_fr_test = anml_fr_df[(anml_fr_df['ds'] > '2020-04-01')]
#
# new_model.fit(anml_fr_train)
#
# # ---------------- Final Prediction ----------------------------------->
#
# final_future = new_model.make_future_dataframe(periods=41)
# final_forecast = new_model.predict(final_future)
#
# print(final_forecast[["ds", "yhat"]].tail())
# # # print("\n\n")
# # # print(test[["ds", "y"]].tail())

# ========================================= Save the model ================================================>

# open a file, where you ant to store the data
file = open('AirQuality.pkl', 'wb')

# dump information to that file
pickle.dump(model, file)


