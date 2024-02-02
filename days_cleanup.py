# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 16:33:50 2023

@author: Tuni
"""

# this file contains a pipeline for cleaning up days.csv

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

#######################################################
# READ IN DATA
#######################################################

# read in CSV, read in attribute "closed" as booleans (instead of "t"/"f") -> easier to work with later on
# also try to parse dates directly so that we can avoid manual parsing later on
days_df = pd.read_csv("data/raw_data/days.csv",
                       sep=",", index_col=None, decimal=".", parse_dates=[2,4,5],
                       true_values=["t"], false_values=["f"])

# check if parsing worked correctly
# XXX: we can see that "date" wasn't parsed correctly as date because of some range issues ("4012" as year instead of "2012) -> we will need to fix that later
print(days_df.head())
print(f"Shape of days.csv: {days_df.shape}")

#######################################################
# THE BASICS: COLUMN NAMES, COLUMNS NEEDED, DATA TYPES, INDEX
#######################################################

# check columns to see if they need to be renamed
print(f"Column headers: {days_df.columns}")

# rename ID and metadata columns (for differentiation once merged with other CSVs) -> easiest to just append "day" to everything
# except canteen_id, to avoid confusion and work more easily with attribute just leave it at "canteen_id"
days_df = days_df.add_prefix(prefix="days_")
days_df = days_df.rename(columns={"days_canteen_id": "canteen_id", "days_date": "date"})
print(f"New column headers: {days_df.columns}")

# check datatypes to allow easy handling in future
# XXX: convert IDs to objects and then we need to take care of the range error in date separately
print("Old datatypes of days_df are:")
print(days_df.dtypes)
days_df["days_id"] = days_df["days_id"].astype("object")
days_df["canteen_id"] = days_df["canteen_id"].astype("object")

# check if everything worked correctly
print("New datatypes of canteen_df are:")
print(days_df.dtypes)

# check if ID is unique, if it is, assign as index
days_stats = days_df.describe(include="all", datetime_is_numeric=True)
days_df = days_df.set_index(keys="days_id")

#######################################################
# CLEAN UP MALFORMED DATE
#######################################################

# correct the date in df, then extract malformed dates and delete them in original df because they mainly already have corrected entries and are thus duplicates
days_df["date_correct"] = days_df["date"]
days_df["date_correct"] = days_df["date_correct"].str.replace("4012", repl="2012")

# look for entries which have already been replaced (these are now duplicates in "date_correct")
duplicates = days_df[days_df.duplicated(subset=["canteen_id", "date_correct"], keep=False)]

# delete duplicated entries which already had been replaced
to_be_deleted = duplicates[duplicates["date"].str.contains("4012")].index
days_df = days_df.drop(index=to_be_deleted)

# finally we can transform "date_correct" to datetime format
days_df["date_correct"] = pd.to_datetime(days_df["date_correct"], format="%Y-%m-%d")
print("New datatypes of canteen_df are:")
print(days_df.dtypes)

#######################################################
# FILTER ANALYSIS TIMEFRAME
#######################################################

# define start and stop date as investigated in exploration
om_start_date = pd.Timestamp(year=2012, month=8, day=1)
om_stop_date = pd.Timestamp(year=2023, month=9, day=1)

# filter days that lie outside of analysis timeframe
days_df = days_df[days_df["date_correct"] > om_start_date]
days_df = days_df[days_df["date_correct"] < om_stop_date]

#######################################################
# FILTER "closed"
#######################################################

# we don't need days on which canteens are closed because they won't have any meal information anyways
# so we will only select the open days
days_df = days_df.loc[~days_df["days_closed"]]

#######################################################
# FILTER ANALYSIS CANTEENS
#######################################################

# load cleaned canteens to be used in nutritional analysis
analysis_canteens = pd.read_pickle("data/processed_data/canteens_cleaned.pkl")
analysis_canteens_set = analysis_canteens.index

# filter days
days_df = days_df[days_df["canteen_id"].isin(analysis_canteens_set)]

#######################################################
# SELECT NEEDED FEATURES AND SAVE
#######################################################

# since closed now only contains False value, we can drop it
# metadata about data creation is probably not relevant either, but we will keep it for now
# original date is probably also not relevant, but we will keep it just in case for further analysis and error checking
days_df = days_df.drop(columns=["days_closed"])

# now save
days_df.to_pickle("data/processed_data/days_cleaned.pkl")
