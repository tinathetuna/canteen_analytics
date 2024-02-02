# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:41:56 2023

@author: Tuni
"""

# this file is for cleaning up notes-csv and exporting the clean data set for later use

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

#######################################################
# READ IN DATA
#######################################################

# read in notes.csv
notes_df = pd.read_csv("data/raw_data/notes.csv", sep=",", parse_dates=[2,3])

# to match notes and meals, we need to use auxiliary df meals_notes.csv
mapper_df = pd.read_csv("data/raw_data/meals_notes.csv", sep=",")


# check if everything worked correctly using head() -> too much data to use interactive explorer
print("Let's take a first glance at our data to check for reading errors and what we are dealing with")
print(notes_df.head())

print("Let's take a first glance at our data to check for reading errors and what we are dealing with")
print(mapper_df.tail())

#######################################################
# THE BASICS: COLUMNS, ROWS, DATA TYPES
#######################################################

#  add "notes_" to all headers to avoid confusion
print(f"Dataset notes.csv has shape: {notes_df.shape}")
print(f"Column headers of notes.csv: {notes_df.columns}")
notes_df = notes_df.add_prefix(prefix="notes_")

# same for mapper_df (but we just have one column to rename, "id")
print(f"Dataset meals_notes.csv has shape: {mapper_df.shape}")
print(f"Column headers of meals_notes.csv: {mapper_df.columns}")
mapper_df = mapper_df.rename(columns={"id": "mapper_id"})

# adjust mapper_df and notes_df data types to allow for easier data manipulation later on
print("datatypes of notes_df are:")
print(notes_df.dtypes)
notes_df["notes_id"] = notes_df["notes_id"].astype("object")
print("datatypes of notes_df (NEW) are:")
print(notes_df.dtypes)

print("datatypes of meals_notes.csv are:")
print(mapper_df.dtypes)
mapper_df = mapper_df.astype("object")
print("datatypes of meals_notes.csv (NEW) are:")
print(mapper_df.dtypes)


# set indices to work with more convenience with data
mapper_df = mapper_df.set_index(keys="mapper_id")
notes_df = notes_df.set_index(keys="notes_id")

#######################################################
# DUPLICATES
#######################################################

# only mapper_df contains duplicates, we will remove them except first occurrence
# XXX: ATTENTION: index should not be included in duplicates check
duplicates = mapper_df[mapper_df.duplicated(keep=False)]
print(f"Number of duplicates contained in mapper_df: {duplicates.shape[0]}")

# now delete -> ATTENTION: use setting keep="first" to retain one record of each duplicate group
mapper_df = mapper_df[~mapper_df.duplicated(keep="first")]

#######################################################
# SELECT ONLY NEEDED CANTEENS
#######################################################

# to filter out notes that belong to German university canteens we need the pre-cleaned meals_df
# this way we can also drastically reduce the amount of data we are dealing with
meals_df = pd.read_pickle("data/processed_data/meals_cleaned.pkl")


# join data together using the different IDs, use inner join to discard all data not contained in meals_df
mapper_subset = pd.merge(left=mapper_df, right=meals_df, how="inner", left_on="meal_id", right_index=True)
mapper_subset = pd.merge(left=mapper_subset, right=notes_df, how="inner", left_on="note_id", right_index=True)

# select unique set of notes used in German university canteens
notes_subset = notes_df[notes_df.index.isin(mapper_subset["note_id"])]

#######################################################
# DATA DISTRIBUTION: mapper_df
#######################################################

# during exploratory data analysis frequency analysis of used notes was especially interesting
# we will probably use 90% of the used tags so that we can still handle them manually
# but we will only take a look at notes belongign to subset of German university meals
notes_categories_subset = mapper_subset["note_id"].value_counts(dropna=False).to_frame(name="count")
notes_categories_subset["percent"] = notes_categories_subset["count"] / mapper_subset.shape[0] * 100
notes_categories_subset["cum_sum"] = notes_categories_subset["count"].cumsum()
notes_categories_subset["cum_percent"] = notes_categories_subset["cum_sum"] / mapper_subset.shape[0] * 100
notes_categories_subset = pd.merge(left=notes_categories_subset, right=notes_df["notes_name"], left_index=True, right_index=True, how="left").reset_index(names="note_id")

# to use for later processing of data, save note categories into csv
notes_categories_subset.to_csv("data/helper_data/analysis_subset_notes_categories_with_counts.csv")

#######################################################
# MERGING NOTES TOGETHER
#######################################################

# as the main subjects of our analysis are meals, I think it would be ideal to merge all notes of a meal together into one feature
# this way, meals will be the unique entities in DF instead of ID of mapper_df
# also, our DF won't be blown up as much this way

# convert notes into one nested list feature, we'll try 2 approaches: (a) use all notes, (b) use only the 90% most common notes
# we'll use the workflow established in db_copy_notes_exploration.py (see for more details)

# APPROACH (a) ----------------------------------------

# merge meals with mapper_subset in a left join to retain meals without notes
# reset index because otherwise meal_id would be lost during merge process (not unique anymore after joining)
# ATTENTION: reset index as individual operation to avoid memory constraints, also drop some columns to use less memory for merge operation (except meal_id and data on notes we won't need anything anyways)
meals_df_to_merge = meals_df.reset_index()
meals_df_to_merge = meals_df_to_merge[["meal_id", "meal_name"]]
meals_with_notes = pd.merge(left=meals_df_to_merge, right=mapper_df, how="left", left_on="meal_id", right_on="meal_id")
meals_with_notes = pd.merge(left=meals_with_notes, right=notes_df, how="left", left_on="note_id", right_index=True)

# before we can proceed, fill na so that string concatenation works (TypeError otherwise)
meals_with_notes["notes_nan_filled"] = meals_with_notes["notes_name"].fillna("N/A")

# now we can group by meal id and combine all notes corresponding to that ID into one list, which will be a new feature
# we will also keep track of how many notes this meal has
temp = meals_with_notes.groupby("meal_id").agg(notes_list=("notes_nan_filled", lambda col: ";".join(col)), notes_count=("notes_name", "count"))

# these two new feature we can now merge back into meals_df
meals_df = pd.merge(left=meals_df, right=temp, left_index=True, right_index=True, how="left")

# APPROACH (b) ------------------------------------------

# the same approach, but we will only use the 90% most common tags
common_notes = notes_categories_subset[notes_categories_subset["cum_percent"] <=90]
common_mapper = mapper_subset.loc[mapper_df["note_id"].isin(common_notes["note_id"]), ["meal_id", "note_id"]]
meals_with_notes = pd.merge(left=meals_df_to_merge, right=common_mapper, how="left", left_on="meal_id", right_on="meal_id")
meals_with_notes = pd.merge(left=meals_with_notes, right=notes_df, how="left", left_on="note_id", right_index=True)

# once again, fill na before proceeding
meals_with_notes["notes_nan_filled"] = meals_with_notes["notes_name"].fillna("N/A")

# now once again group by meal_id and extratct features, but rename
temp = meals_with_notes.groupby("meal_id").agg(notes_list_90=("notes_nan_filled", lambda col: ";".join(col)), notes_count_90=("notes_name", "count"))

# add back into meals_df
meals_df = pd.merge(left=meals_df, right=temp, left_index=True, right_index=True, how="left")

########################################################
# SAVE DATA
########################################################

# save cleaned data
# this time we will use pickles as saving format, because otherwise we will run into issues with reading in our notes lists later on
meals_df.to_pickle("data/processed_data/meals_cleaned_with_notes.pkl")
