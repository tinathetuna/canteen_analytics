# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 22:24:02 2023

@author: Tuni
"""

# this file is for exploring the content and structure of notes.csv

import pandas as pd
#import missingno as msno
#import geopandas as gpd
import matplotlib.pyplot as plt

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

# read in notes.csv
notes_df = pd.read_csv("data/raw_data/notes.csv", sep=",", parse_dates=[2,3])

# to match notes and meals, we need to use auxiliary df meals_notes.csv
# XXX: right away we can see that these DFs / database schemes work in a different way, we have just ~240,000 unique notes, but 38,000,000 (!!) mappings
# XXX: that also means that one meal can have multiple notes
mapper_df = pd.read_csv("data/raw_data/meals_notes.csv", sep=",")


# check if everything worked correctly using head() -> too much data to use interactive explorer
# XXX: I was expecting different data -> more allergens information instead of an additional description of meals
# XXX: like with other CSVs we have the actual data (id, name) and data curation metadata
# XXX: earlier notes seem malformed, repeat themselves twice in notes_name
print("Let's take a first glance at our data to check for reading errors and what we are dealing with")
print(notes_df.tail())
print(notes_df.head())


# XXX: a surprising amount of IDs haha
# XXX: as suspected, it's not a 1:1 mapping, but one meal can have different notes -> can one note also belong to different meals?
print("Let's take a first glance at our data to check for reading errors and what we are dealing with")
print(mapper_df.tail())


# take a first look at summary statistics for some basic knowledge about uniqueness, completeness, distribution
# XXX: as expected, notes are unique
# XXX: we have more notes created towards the later years, but we also have in general more data created in that period, so makes sense
# XXX: need to convert mapper_df to strings first, otherwise are treated as numbers
print(notes_df.describe(include="all", datetime_is_numeric=True))
print(mapper_df.describe(include="all"))

#######################################################
# THE BASICS: COLUMNS, ROWS, DATA TYPES
#######################################################

# let's check datset size and header quality
# XXXX: we should probably add "notes_" to all headers to avoid confusion
print(f"Dataset notes.csv has shape: {notes_df.shape}")
print(f"Column headers of notes.csv: {notes_df.columns}")
notes_df = notes_df.add_prefix(prefix="notes_")

# same analysis for mapper_df
# XXX: nothing much here, we can transform "id" to "mapper_id"
print(f"Dataset meals_notes.csv has shape: {mapper_df.shape}")
print(f"Column headers of meals_notes.csv: {mapper_df.columns}")
mapper_df = mapper_df.rename(columns={"id": "mapper_id"})

# let's check and djust data types (we have already seen above that they are mismatched)
# XXX: "id" should be object, the rest is fine
print("datatypes of notes_df are:")
print(notes_df.dtypes)

# XXX: the same for mapper_df, all IDs should be object (you can't do math with them)
print("datatypes of meals_notes.csv are:")
print(mapper_df.dtypes)

# change data types -> we can change all columns of mapper_df at once for more convenience
notes_df["notes_id"] = notes_df["notes_id"].astype("object")
mapper_df = mapper_df.astype("object")

# once again, data summary with the correct data types
# XXX: as expected above, one meal can have multiple notes (even though 297 seems excessive)
# XXX: note are reused, otherwise one note would't be matched to ~6,000,000 meals
notes_stats = notes_df.describe(include="all", datetime_is_numeric=True)
mapper_stats = mapper_df.describe(include="all")

# set indices to work with more convenience with data
mapper_df = mapper_df.set_index(keys="mapper_id")
notes_df = notes_df.set_index(keys="notes_id")

#######################################################
# MISSING DATA
#######################################################

# let's take a look at missing data to see if data is somehow corrupted
# XXX: as expected, notes are complete -> nothing to handle here
notes_missing_stats = notes_df.isna().sum().to_frame(name="count")
notes_missing_stats["percentage"] = notes_missing_stats["count"] / notes_df.shape[0] * 100
print("Missing entries per feature for notes.csv")
print(notes_missing_stats)

# same analysis for mapper_df
# XXX: also complete (actually we've seen that in summary statistics already) -> nothing to worry about here either
mapper_missing_stats = mapper_df.isna().sum().to_frame(name="count")
mapper_missing_stats["percentage"] = mapper_missing_stats["count"] / mapper_df.shape[0] * 100
print("Missing entries per feature for meals_notes.csv")
print(mapper_missing_stats)

#######################################################
# DUPLICATES
#######################################################

# a duplicate for notes_df would be a duplicated name, as the other categories are metadata -> we've already seen in summary stats that this feature is unique
# we'll check for duplicates across whole row too, though, just in case there was a DB error
# XXX: no duplicates across whole row, no duplicates in name -> nothing to handle
print("Print duplicates for notes.csv (all features):")
print(notes_df[notes_df.duplicated(keep=False)])
print("Print duplicates for notes.csv (just name):")
print(notes_df[notes_df.duplicated(subset=["notes_name"], keep=False)])

# a duplicate for mapper_df would be identical meal_id and note_id (due to prior cleaning we know that meal_id will refer to unique meals at this point)
# we'll also check for DB errors and completely identical rows
# XXX: no completely identical rows, but about ~180,000 duplicates in meal_id and note_id -> we could check if they alway belong to the same parser (parsing error) -> but probably we can just drop them
print("Print duplicates for meals_notes.csv (all features):")
print(mapper_df[mapper_df.duplicated(keep=False)])
print("Print duplicates for meals_notes.csv (just meal_id and note_id):")
duplicates_mapper = mapper_df[mapper_df.duplicated(subset=["meal_id", "note_id"], keep=False)]
print(duplicates_mapper)

#######################################################
# MERGE DATA WITH MEALS
#######################################################

# before we move on to the analysis of data distribution, we should select our canteen sample
# that is because in the two samples the distributions could be quite different (e.g., we already know that we have much data from Luxembourg which could influence also our notes)
meals_df = pd.read_csv("data/raw_data/meals_edited.csv", sep=",")
meals_df.rename(columns={"id": "meal_id", "name": "meal_name", "category": "meal_category", "city": "canteen_city"}, inplace=True)
meals_df = meals_df.set_index(keys="meal_id")

# join data together using the different IDs, use inner join to discard all data not contained in meals_df
german_meals = meals_df[meals_df["country"] == "Germany"]
mapper_subset = pd.merge(left=mapper_df, right=german_meals[["meal_name", "date_correct", "meal_category", "canteen_name", "canteen_city"]], how="inner", left_on="meal_id", right_index=True)
mapper_subset = pd.merge(left=mapper_subset, right=notes_df, how="inner", left_on="note_id", right_index=True)

# select unique set of notes used in German university canteens
notes_subset = notes_df[notes_df.index.isin(mapper_subset["note_id"])]

#######################################################
# DATA DISTRIBUTION: notes_df
#######################################################

# main attention is on notes_name

# first of all, content of notes_name was not what I expected -> maybe text length can give us more info on variety of content
# XXX: extremes are very extreme, just one character and 250 characters
# XXX: even 50 characters seems extreme
notes_df["notes_length"] = notes_df["notes_name"].str.len()
print(notes_df["notes_length"].describe())
plt.figure()
ax = notes_df["notes_length"].plot(kind="box", ylabel="Character count", color="black", patch_artist=True, boxprops=dict(facecolor=layout_color))
#ax.set_title("Distribution of note content length in notes.csv")

# take a look at data edges
# XXX: short notes seems confusing, but they are probably abbreviations and we don't know the key
# XXX: with more characters we also have short words, like "Fisch", "Lamm", etc.
# XXX: sometimes we also have numbers, not sure of they are prices or also abbreviations
# XXX: long descriptions contain several allergens, prices, and the possible side dish selection -> I think these may be from company canteens?
temp = notes_df.sort_values(by="notes_length")
temp = notes_df.sort_values(by="notes_length", ascending=False)

# now only for German university canteens
# XXX: notes are in general shorter, but extremes persist
# XXX: median of 40 characters is also longer than what I expected for allergens
notes_subset["notes_length"] = notes_subset["notes_name"].str.len()
print(notes_subset["notes_length"].describe())
fig, ax = plt.subplots()
notes_subset["notes_length"].plot(kind="box", title="Note length in notes used for German university canteens", ylabel="no. characters")

# take a look at data edges
# XXX: same a above, further insights: we also have English allergens
# XXX: sometimes allergens are chained (e.g. "c,e,g")
# XXX: all in all: will be hard to seperate the different types of notes and re-translate allergens
temp = notes_subset[["notes_name", "notes_length"]].sort_values(by="notes_length")
temp = notes_subset[["notes_name", "notes_length"]].sort_values(by="notes_length", ascending=False)

# we'll quickly check time difference between note creation and note update
# XXX: as expected, not much delay, in fact, none of them never once updated
notes_df["time_difference_creation_update"] = notes_df["notes_updated_at"] - notes_df["notes_created_at"]
print(notes_df["time_difference_creation_update"].describe())

#######################################################
# DATA DISTRIBUTION: mapper_df
#######################################################

# for mapper_df I think especially the frequency distributions of the individual notes are most interesting
# also, how many notes a meal usually has
# maybe also some advanced patterns depending on the association with canteen / Studierendenwerk

# (1) frequency distribution of note categories -------

# let's get started with freuqency distribution of notes
# XXX: we have only ~220,000 categories, so not even all notes are in use -> going by numbers only, every note is used 172 times
# XXX: data is severely slanted, 148 categories of 223.335 make up 90% of the dataset
# XXX: we have many French entries -> makes sense due to LUX meals
# XXX: finally we find the epxected allergen information ranging among the most common entries
notes_categories = mapper_df["note_id"].value_counts(dropna=False).to_frame(name="count")
notes_categories["percent"] = notes_categories["count"] / mapper_df.shape[0] * 100
notes_categories["cum_sum"] = notes_categories["count"].cumsum()
notes_categories["cum_percent"] = notes_categories["cum_sum"] / mapper_df.shape[0] * 100
notes_categories["categories_percent"] = 1/notes_df.shape[0] * 100
notes_categories["categories_percent_cumsum"] = notes_categories["categories_percent"].cumsum()
notes_categories = pd.merge(left=notes_categories, right=notes_df["notes_name"], left_index=True, right_index=True, how="left").reset_index()

# same analysis for German university canteens
# XXX: statistically speaking, every note is used ~370 times
# XXX: data is also slanted, 190 categories make up 90% of the used notes
notes_categories_subset = mapper_subset["note_id"].value_counts(dropna=False).to_frame(name="count")
notes_categories_subset["percent"] = notes_categories_subset["count"] / mapper_subset.shape[0] * 100
notes_categories_subset["cum_sum"] = notes_categories_subset["count"].cumsum()
notes_categories_subset["cum_percent"] = notes_categories_subset["cum_sum"] / mapper_subset.shape[0] * 100
notes_categories_subset = pd.merge(left=notes_categories_subset, right=notes_df["notes_name"], left_index=True, right_index=True, how="left").reset_index()

# generate plot similar to Gini coefficient showing data slantedness visually
plt.figure()
ax = notes_categories.plot(x="categories_percent_cumsum", y="cum_percent", kind="line", color=layout_color, lw=5, xlabel="Note categories covered [%]", ylabel="Notes covered [%]", legend=False)
#ax.set_title("Cumulated percent coverage of note categories used in meal_notes.csv")
print(notes_categories["count"].describe())

# (2) notes per meal -----------------------------------

# let's move on to how many notes a meal has
# XXX: median number of notes per meal is 3, seems reasonable
# XXX: maximum number of notes (297 per meal) seems complete out of range
# XXX: we should check later if there is any meals in meals_df that don't have any notes at all
meal_frequencies = mapper_df["meal_id"].value_counts()
meal_frequencies_stats = meal_frequencies.describe()
print(meal_frequencies_stats)

# let's take a look at the upper percentiles and if there is a continuous increase in notes per meal
# XXX: no, there isn't, we can see different "steps" in the boxplot, one at 297, the other at 227 and then again smaller steps starting at 61
# XXX: in general, the data is slanted again
plt.figure()
ax = meal_frequencies.plot(kind="box", ylabel="Note count", color="black", patch_artist=True, boxprops=dict(facecolor=layout_color))
ax.set_title("Distribution of note count per meal")

# filter meals above IQR * 1.5 (the edge of the upper whisker)
# XXX: we obtained 1,086,695 outliers
iqr = meal_frequencies_stats["75%"] - meal_frequencies_stats["25%"]
outliers = meal_frequencies[(meal_frequencies > (meal_frequencies_stats["75%"] + iqr * 1.5))]

# what kind of content can we find in outliers?
# XXX: many general purpose dishes and customizable dishes, that's why tagged with so many things
outliers = pd.merge(left=outliers, right=meals_df, left_index=True, right_on="meal_id", how="left")

# let's see if we have the same trends in German meals before we dig more into which kind of meals have so many tags and why (probably a parser error...)
# XXX: distribution is roughly the same, we also still have the meals with 297 notes
meal_frequencies_subset = mapper_subset["meal_id"].value_counts().rename("note_counts_per_meal")
print(meal_frequencies_subset.describe())

# XXX: analysis of steps also seems roughly the same
plt.figure()
meal_frequencies_subset.plot(kind="box", ylabel="no. notes associated with each meal", title="Count of notes each meal is tagged with (only German university canteens)")

# let's identify the meals and canteens that have an unusual amount of notes
# XXX: the top two counts are some kind of general purpose main dish description -> I guess it's not really an error, but not super high in explanatory power either -> leave them for now
# XXX. other meals with many tags are customizable salad bowls
meal_frequencies_subset = pd.merge(left=meal_frequencies_subset, right=meals_df, left_index=True, right_index=True, how="left")
temp = meal_frequencies_subset.head(500)

# let's take a look at tags -> I sampled the first entry of each meal frequency count (we're only interested in the larger counts for now and those were similar anyways)
# we need to reset and set again the index a couple of times so that we won't lose meal_id and can join on it again later
# XXX: Hauptkomponente is tagged with about every possible meal option -> parsing error? or just an unconventional way the data is managed? I guess the notes do not give us anything in this case, but we can leave them
# XXX: salad dishes are tagged with many things (since you can assemble them yourself, you can potentially touch upon many allergens) -> but sometimes the tags are also duplicates and that's why there are so many -> issue should solve itself once duplicates get removed
# XXX: some meals were also rated with stars on a sustainability scale and each star was a separate tag -> interesting, but for now probably useless for our analysis
# XXX: all in all: meals with many tags are not really "flawed", but not really helpful either
first = meal_frequencies_subset.reset_index(drop=False, names="meal_id").groupby("note_counts_per_meal").first().reset_index(drop=False)#.set_index(keys="meal_id", drop=True)
first = pd.merge(left=first, right=mapper_df, left_on="meal_id", right_on="meal_id", how="left")
first = pd.merge(left=first, right=notes_df, left_on="note_id", right_index=True, how="left")
