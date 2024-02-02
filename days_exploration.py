# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:20:54 2023

@author: Tuni
"""

import pandas as pd
#import missingno as msno
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# options for plotting
# ATTENTION: for figures included in thesis, i commented out the figure title, but for individual use I would recommend printing it for better orientation
plt.rcParams.update({"axes.labelsize": 22, # size of axis labels
                     "axes.titlesize": 24, # size of figure title
                     "xtick.labelsize": 20, # size of axes annotation of ticks
                     "ytick.labelsize": 20, # size of axis annotation of tickes
                     "figure.titlesize": 24, # plt.suptitle size
                     })
layout_color = "#004E8A"

#######################################################
# READ IN DATA
#######################################################

# read in CSV, take care to declare delimiter correctly (it's easy in our case, just a normal ",")
# parse dates directly for easier handling -> need to give column number (0-indexed)
# parse "closed" values as booleans instead of "f" / "t" -> easier to handle later
days_df = pd.read_csv("data/raw_data/days.csv",
                       sep=",", index_col=None, decimal=".", parse_dates=[2,4,5],
                       true_values=["t"], false_values=["f"])

#######################################################
# THE BASICS
#######################################################

# check if parsing worked correctly
print(days_df.head())
print(f"Shape of days.csv: {days_df.shape}")

# I would be interested in seeing whether closed is actually really True sometimes
# XX: updated_at and created_at have almost the same statistics, so I'm guessing they contain similar information and one can be discarded
# XX: no missing values in any column
# XX: there actually is entries with closed = True, but data type needs to be updated -> update in read_csv()
days_df.describe(include="all", datetime_is_numeric=True)

# especially check for correct parsing of data types
print(days_df.dtypes)

# manually convert ID for later analysis, they are categories not numbers
days_df["id"] = days_df["id"].astype("object")
days_df["canteen_id"] = days_df["canteen_id"].astype("object")

# convert "date" to datetime manually because parsing didn't work
# XX: there is a problem because many entries look similar to "4012-09-12" -> I'm guessing it is a typing error, but I will investigate it just in case
#days_df["date"] = pd.to_datetime(days_df["date"], format="%Y-%m-%d") # thorws error because of malformed data (see below)

#######################################################
# MALFORMED DATA: "date"
#######################################################

# to investigate typo, extract year, month and day as strings with named regular expression, then convert to numbers
days_df[["year", "month", "day"]] = days_df["date"].str.extract(pat=r"(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})", expand=True)
days_df[["year", "month", "day"]] = days_df[["year", "month", "day"]].astype(dtype="int32", copy=True)

# take a look at summary statistics to spot ranges of features
# XX: month and day range between 1-12 / 1-31, but year ranges from 1999 to 4012
print(days_df[["year", "month", "day"]].describe())

# visualize in boxplot
ax = days_df[["year", "month", "day"]].plot(kind="box", subplots=True, title="Distribution of year, month and day data of days.csv")

# look at year categories
# XX: there shouldn't be any data before 2012, because first canteen was only added in 2012
# XX: 4012 is the only typo -> I'll check creation date, but I'm assuming they were all meant to be 2012
# XX: some dates already for 2024
temp = days_df["year"].value_counts().sort_index().to_frame(name="count")
temp["percent"] = temp["count"] / days_df.shape[0] * 100
plt.figure()
ax = temp["percent"].plot(kind="bar", color=layout_color, ylabel="Percent [%]", xlabel="Years")
#ax.set_title("Year data contained in days.csv", fontsize=24)
plt.xlabel("Years", labelpad=15)
ax.bar_label(ax.containers[0], fmt="%.2f", fontsize=18)
ax.margins(y=0.1)
plt.tight_layout()

# isolate entries with year = 4012, then check if created in 2012
# XX: all of them created in 2012, I guess it's safe to manually correct them
wrong_year = days_df[days_df["year"] == 4012]
print(f"Entries with typo year = 4012 not created in year 2012: {(wrong_year['created_at'].dt.year == 2012).sum() - wrong_year.shape[0]}")

# manually correct year of data entries with typos
days_df["year_old"] = days_df["year"]
days_df["year"] = days_df["year_old"].mask(cond=days_df["year"] == 4012, other=2012)

# convert year, month and day to datetime feature
days_df["date_correct"] = pd.to_datetime(days_df[["year", "month", "day"]])

# now check if we have created duplicates (avoid day_id and metadata in analysis because they depend on moment of data creation)
# sort duplicates for easier analysis of duplicates
# XXX: canteens with malformed date have already been corrected in dataset -> we can delete malformed entries -> but not all of them, then we would have 714 duplicates (but we only have 644)
duplicates = days_df[days_df.duplicated(subset=["canteen_id", "date_correct"], keep=False)]
duplicates = duplicates.sort_values(by=["canteen_id", "date_correct", "closed"])

# extract malformed dates to check if really all of them have already been corrected in dataset
# XXX: right_only are manually-added corrected entries, left_only are not-yet-corrected entries, both are redundant entries that should be removed
not_replaced = pd.merge(wrong_year, duplicates, on="id", how="outer", indicator=True)
not_replaced["_merge"].value_counts()


########################################################
# DATA DISTRIBUTION: "closed"
########################################################

# check distribution of boolean values in "closed" with bar chart
# first calculate statistics
# XXX: we have WAY more days with open canteens -> that makes sense because otherwise there wouldn't be any recorded menus
closed_distribution = days_df["closed"].value_counts(dropna=False).to_frame(name="count")
closed_distribution["percent"] = closed_distribution["count"] / days_df.shape[0] * 100
print(closed_distribution)

# now plot statistics from above
# we'll combine it with plot from below as subplots (see below)
fig, ax = plt.subplots(1,2)
ax1 = closed_distribution["percent"].plot(kind="bar", ax=ax[0], color=layout_color, ylabel="Percent [%]", xlabel='Value of "closed"')
#ax.set_title('Distribution of feature "closed" in days.csv', fontsize=24)
ax1.bar_label(ax1.containers[0], fmt="%.2f", fontsize=18)
ax1.margins(y=0.1)
plt.tight_layout()

# check for each canteen id, if they actually have True and False entries for "closed"
# XXX: we don't have days for all canteens -> makes sense because some canteens have never been fetched once
# XXX: usually more open than closed days, some canteens don't have any closed days
opening_dates_per_canteen = days_df.groupby(by="canteen_id")["closed"].value_counts().to_frame(name="count").reset_index()
opening_dates_per_canteen = opening_dates_per_canteen.pivot(index="canteen_id", columns="closed", values="count")
opening_dates_per_canteen.columns = [["Open", "Closed"]]
opening_dates_per_canteen = opening_dates_per_canteen.fillna(0)

# plot distributions per feature of "closed"
opening_dates_per_canteen.plot(kind="box", subplots=True, sharey=True)
print(opening_dates_per_canteen.describe())

# count canteens with no closed dates and canteens with at least one closed date
count_canteens_closed = (opening_dates_per_canteen["Closed"] > 0).sum()[0]
count_canteens_never_closed = (opening_dates_per_canteen["Closed"] == 0).sum()[0]

# calculate percentages and save in df
ccc_percent = count_canteens_closed / opening_dates_per_canteen.shape[0] * 100
ccnc_percent = count_canteens_never_closed / opening_dates_per_canteen.shape[0] * 100
df = pd.DataFrame(data=[ccc_percent, ccnc_percent], index=["with_closing_dates", "without_closing_dates"], columns=["percentage"])

#fig, ax = plt.subplots()
# combined plotting of both plots mentioned
# first prepare canvas
fig, ax = plt.subplots(1,2)

# first plot
ax1 = closed_distribution["percent"].plot(kind="bar", ax=ax[0], color=layout_color, ylabel="Percent [%] of days", xlabel="", rot=0)
ax1.set_title('(a) Value of feature "closed"', fontsize=22)
ax1.bar_label(ax1.containers[0], fmt="%.2f", fontsize=18)
ax1.margins(y=0.1)
#plt.tight_layout()

# second plot
ax2 = df["percentage"].plot(kind="bar", ax=ax[1], color=layout_color, ylabel="Percent [%] of canteens", xlabel="", rot=0)
ax2.set_title("(b) Canteens ...", fontsize=22)
ax2.set_xticklabels(["with closing days", "without closing days"])
ax2.bar_label(ax2.containers[0], fmt="%.2f", fontsize=18)
ax2.margins(y=0.1)

#########################################################
# SAVE CLEANED DATA (FOR USAGE IN MEALS EXPLORATION)
#########################################################

# rename id feature to avoid duplicates when joining other CSVs
days_df.rename(columns={"id": "days_id"}, inplace=True)

# we can drop auxiliary columns (year, month, day), also created_at and updated_at
# I'm keeping the original date just in case there are further anomalies or problems with our conversion
to_be_saved = days_df[["days_id", "canteen_id", "date", "date_correct", "closed"]]

# save cleaned up data for joining with other dataset
to_be_saved.to_csv("data/raw_data/days_edited.csv")
