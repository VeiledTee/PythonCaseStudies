import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

birddata = pd.read_csv("birdTrackingHomework.csv", index_col=0)
# birddata.head()

# ----- Exercise 1 ----- #
# First, use `groupby()` to group the data by "bird_name".
grouped_birds = birddata.groupby('bird_name')
# Now calculate the mean of `speed_2d` using the `mean()` function.
mean_speeds = grouped_birds['speed_2d'].mean()
# Find the mean `altitude` for each bird.
mean_altitudes = grouped_birds['altitude'].mean()


# ----- Exercise 2 ----- #
# Convert birddata.date_time to the `pd.datetime` format.
birddata.date_time = pd.to_datetime(birddata.date_time)
# Create a new column of day of observation
birddata["date"] = birddata['date_time'].dt.date
# Use `groupby()` to group the data by date.
grouped_bydates = birddata.groupby('date')
# Find the mean `altitude` for each date.
mean_altitudes_perday = grouped_bydates['altitude'].mean()
date = pd.to_datetime("2013-09-12")
print(mean_altitudes_perday[pd.to_datetime('2013-09-12').date()])  # 75.64609053497942

# -----Exercise 3 ----- #
# Use `groupby()` to group the data by bird and date.
grouped_birdday = birddata.groupby(['bird_name', 'date'])
# Find the mean `altitude` for each bird and date.
mean_altitudes_perday = grouped_birdday['altitude'].mean()
print(mean_altitudes_perday['Eric', pd.to_datetime('2013-08-18').date()])  # 121.35365853658537

# ----- Exercise 4 ----- #
mean_speeds_perday = grouped_birdday['speed_2d'].mean()
eric_daily_speed = pd.Series(mean_speeds_perday["Eric"])
sanne_daily_speed = pd.Series(mean_speeds_perday["Sanne"])
nico_daily_speed = pd.Series(mean_speeds_perday["Nico"])
print(mean_speeds_perday["Nico", pd.to_datetime('2014-04-04').date()])  # 2.8324654508684057

eric_daily_speed.plot(label="Eric")
sanne_daily_speed.plot(label="Sanne")
nico_daily_speed.plot(label="Nico")
plt.legend(loc="upper left")
plt.show()
