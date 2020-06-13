import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -- 4.2.1 intro to GPS Tracking of Birds
birdData = pd.read_csv("bird_tracking.csv")
# print(birdData.info())
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 61920 entries, 0 to 61919
Data columns (total 9 columns):
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   Unnamed: 0          61920 non-null  int64  
 1   altitude            61920 non-null  int64  
 2   date_time           61920 non-null  object 
 3   device_info_serial  61920 non-null  int64  
 4   direction           61477 non-null  float64
 5   latitude            61920 non-null  float64
 6   longitude           61920 non-null  float64
 7   speed_2d            61477 non-null  float64
 8   bird_name           61920 non-null  object 
dtypes: float64(4), int64(3), object(2)
memory usage: 4.3+ MB
"""
# print(birdData.head())
"""
   Unnamed: 0  altitude               date_time  ...  longitude  speed_2d  bird_name
0           0        71  2013-08-15 00:18:08+00  ...   2.120733  0.150000       Eric
1           1        68  2013-08-15 00:48:07+00  ...   2.120746  2.438360       Eric
2           2        68  2013-08-15 01:17:58+00  ...   2.120885  0.596657       Eric
3           3        73  2013-08-15 01:47:51+00  ...   2.120859  0.310161       Eric
4           4        69  2013-08-15 02:17:42+00  ...   2.120887  0.193132       Eric

[5 rows x 9 columns]
"""

# -- 4.2.2 Simple Data Visualizations
# finding Eric's trajectory in 2D
"""
ix = birdData.bird_name == "Eric"
x, y = birdData.longitude[ix], birdData.latitude[ix]

plt.figure(figsize=(7, 7))
plt.plot(x, y, "bo")
plt.show()
"""
# finding every bird's trajectory in 2D

birdNames = pd.unique(birdData.bird_name)
"""
plt.figure(figsize=(7, 7))
for birdName in birdNames:
    ix = birdData.bird_name == birdName
    x, y = birdData.longitude[ix], birdData.latitude[ix]
    plt.plot(x, y, ".", label=birdName)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc="lower right")
# plt.savefig("3traj.pdf")
# plt.show()
"""

# -- 4.2.3 Examine Flight Speed
# USING MATPLOTLIB
ix = birdData.bird_name == "Eric"
speed = birdData.speed_2d[ix]
# plt.hist(speed)  # WONT WORK
# print(np.isnan(speed))  # True if not a number, False if is a number
"""
0        False
1        False
2        False
3        False
4        False
         ...  
19790    False
19791    False
19792    False
19793    False
19794    False
"""
# print(np.isnan(speed).any())  # much more elegant, returns True
# print(np.sum(np.isnan(speed)))  # 85 non number values
ind = np.isnan(speed)
# print(ind)
"""0        False
1        False
2        False
3        False
4        False
         ...  
19790    False
19791    False
19792    False
19793    False
19794    False
"""
# print(~ind)  # '~' operator reverses
"""
0        True
1        True
2        True
3        True
4        True
         ... 
19790    True
19791    True
19792    True
19793    True
19794    True
"""
# plt.hist(speed[~ind], bins=np.linspace(0, 30, 20), density=True)
# plt.xlabel("2D Speed (m/s)")
# plt.ylabel("Frequency")
# plt.savefig("MPLHist.pdf")

# USING PANDAS
# birdData.speed_2d.plot(kind='hist', range=[0, 30])
# plt.xlabel("2D Speed")
# plt.savefig("PDHist.pdf")

# -- 4.2.4 Using Datetime
# print(birdData.date_time[0:3])
"""
0    2013-08-15 00:18:08+00
1    2013-08-15 00:48:07+00
2    2013-08-15 01:17:58+00
"""
time1 = datetime.datetime.today()
time2 = datetime.datetime.today()
# print(time2 - time1)  # yields a datetime 'time delta' object with the difference between the two times
# ^^ Have to do in console to see difference-> datetime.timedelta(seconds=8, microseconds=68672)

dateStr = birdData.date_time[0]
# print(dateStr)  # 2013-08-15 00:18:08+00
# print(dateStr[:-3])  # 2013-08-15 00:18:08 -> removes '+00'
datetime.datetime.strptime(dateStr[:-3], "%Y-%m-%d %H:%M:%S")
timestamps = []
for i in range(len(birdData)):
    timestamps.append(datetime.datetime.strptime(birdData.date_time.iloc[i][:-3], "%Y-%m-%d %H:%M:%S"))

birdData['timestamp'] = pd.Series(timestamps, index=birdData.index)
# print(birdData.head(3))
"""
   Unnamed: 0  altitude  ... bird_name           timestamp <- NEW COLUMN
0           0        71  ...      Eric 2013-08-15 00:18:08
1           1        68  ...      Eric 2013-08-15 00:48:07
2           2        68  ...      Eric 2013-08-15 01:17:58
"""
times = birdData.timestamp[birdData.bird_name == "Eric"]
elapsedTime = [time - times[0] for time in times]
# print(elapsedTime[0])  # 0 days 00:00:00
# print(elapsedTime[1000])  # 12 days 02:02:00
# using specific units
# print(elapsedTime[1000] / datetime.timedelta(days=1))  # 12.084722222222222
# print(elapsedTime[1000] / datetime.timedelta(hours=1))  # 290.03333333333336

# PLOTTING
# plt.plot(np.array(elapsedTime) / datetime.timedelta(days=1))
# plt.xlabel("Observation")
# plt.ylabel("Elapsed Time (days)")
# plt.savefig("timePlot.pdf")

# -- 4.2.5 Calculating Daily Mean Speed
# data = birdData[birdData.bird_name == "Eric"]
# times = data.timestamp
# elapsedTime = [time - times[0] for time in times]
# elapsedDays = np.array(elapsedTime) / datetime.timedelta(days=1)
#
# nextDay = 1
# indices = []
# dailyMeanSpeeds = []
# for (i, t) in enumerate(elapsedDays):
#     if t < nextDay:
#         indices.append(i)
#     else:
#         # compute mean speed
#         dailyMeanSpeeds.append(np.mean(data.speed_2d[indices]))
#         nextDay += 1
#         indices = []

# plt.figure(figsize=(8, 6))
# plt.plot(dailyMeanSpeeds)
# plt.xlabel("Day")
# plt.ylabel("Mean Speeds (m/s)")
# plt.savefig("dailyMeanSpeed.pdf")

# data = birdData[birdData.bird_name == "Sanne"]
# print(data.timestamp)  # 2013-08-15 00:01:08

# -- 4.2.6 Using the Cartopy Library
proj = ccrs.Mercator()
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
ax = plt.axes(projection=proj)
ax.set_extent((-25.0, 20.0, 52.0, 10.0))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
names = ['Eric', 'Sanne', 'Nico']

for i in range(3):
    ix = birdData.bird_name == names[i]
    x, y = birdData.longitude[ix], birdData.latitude[ix]
    plt.subplot(1, 3, i + 1)
    ax.plot(x, y, '.', transform=ccrs.Geodetic(), label=names[i])
    plt.legend(loc='upper left')

plt.show()
