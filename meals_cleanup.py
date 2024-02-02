# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:54:19 2023

@author: Tuni
"""

# this file contains a pipeline for cleaning up meals.csv

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

#######################################################
# READ IN DATA
#######################################################

# read in CSV
meals_df = pd.read_csv("data/raw_data/meals.csv", sep=",")

# check if parsing worked correctly
print(meals_df.head())
print(f"Shape of meals.csv: {meals_df.shape}")

#######################################################
# THE BASICS: COLUMN NAMES, COLUMNS NEEDED, DATA TYPES, INDEX
#######################################################

# check columns to see if they need to be renamed
print(f"Column headers: {meals_df.columns}")

# rename ID, name, description etc (for differentiation once merged with other CSVs)
# easiest to just append "meal" to everything, except day_id to avoid confusion
meals_df = meals_df.add_prefix(prefix="meal_")
meals_df = meals_df.rename(columns={"meal_day_id": "day_id"})
print(f"New column headers: {meals_df.columns}")

# delete empty, identical or unneeded columns (they don't contain any valuable information)
meals_df = meals_df.drop(labels=["meal_description", "meal_pos"], axis="columns")
print(f"Column headers after deleting unneeded features: {meals_df.columns}")

# adjust data types where necessary for more convenience when working with the data
print("Old datatypes of meals_df are:")
print(meals_df.dtypes)

meals_df["meal_id"] = meals_df["meal_id"].astype("object")
meals_df["meal_created_at"] = pd.to_datetime(meals_df["meal_created_at"], format="%Y-%m-%d %H:%M:%S.%f")
meals_df["meal_updated_at"] = pd.to_datetime(meals_df["meal_updated_at"], format="%Y-%m-%d %H:%M:%S.%f")
meals_df["day_id"] = meals_df["day_id"].astype("object")

print("New datatypes of meals_df are:")
print(meals_df.dtypes)

# check if ID is unique, if it is, assign as index
meal_stats = meals_df.describe(include="all", datetime_is_numeric=True)
meals_df = meals_df.set_index(keys="meal_id")

#######################################################
# FILTER ACCORDING TO NEEDED CANTEENS AND DAYS
#######################################################

# to reduce the amount of data we are working with (execution time, complexity, etc.), filter data based on cleaned canteens.csv and days.csv
# so first of all, load cleaned data
canteen_df = pd.read_pickle("data/processed_data/canteens_cleaned.pkl")
days_df = pd.read_pickle("data/processed_data/days_cleaned.pkl")

# now merge with meals_df -> this time we will use an inner join, because we only want to keep the meals that belong to canteens contained in our pre-filtered canteens_df
# XXX: we end up with about 4,400,000 data points
german_university_meals = pd.merge(left=meals_df, right=days_df, how="inner", left_on="day_id", right_index=True)
german_university_meals = pd.merge(left=german_university_meals, right=canteen_df, how="inner", left_on="canteen_id", right_index=True)

#######################################################
# REMOVE DUPLICATES
#######################################################

# we'll consider the set of meal_name, meal_category, canteen_id, date_correct necessary to uniquely identify a meal
# meal_category because sometimes the same meal is offered in different menu lines
duplicates = german_university_meals[german_university_meals.duplicated(subset=['meal_name', 'meal_category', 'canteen_id', 'date_correct'], keep=False)]
print(f"Number of identical meals: {duplicates.shape[0]}")

# for practical reasons I will only keep the first occurence of all duplicates
mask = german_university_meals.duplicated(subset=['meal_name', 'meal_category', 'canteen_id', 'date_correct'], keep="first")
german_university_meals = german_university_meals[~mask]

#######################################################
# REMOVE NO-DATA ENTRIES
#######################################################

# sometimes canteens are closed and publish something like "Heute geschlossen" instead of a meal -> but this fake meal information will still get scraped by scraper and needs to be removed
# manually checked until "Ingwer-Karottencremesuppe" with count=300 and cum_percent=43.33 -> should be enough to infer schematics of closed canteens
# found: ".", "geschlossen", "entfällt", "Ausgabe geschlossen", "--", "Mensa geschlossen", "keine Angabe", "geschlossen oder geschlossen", "entfällt in der vorlesungsfreien Zeit", "Kein Angebot", "Unsere Mensa ist bis auf Weiteres geschlossen", "keine Ausgabe"
name_categories = german_university_meals["meal_name"].value_counts(dropna=False).to_frame(name="count")
name_categories["cum_sum"] = name_categories["count"].cumsum()
name_categories["cum_percent"] = name_categories["cum_sum"] / german_university_meals.shape[0] * 100

# find the minimum set of phrases that will catch all no-data entries identified above
# XXX: we need to handle "." and "--" separately because some meals contain abbreviations or an accidental double dash ("--")  and would be incorrectly matched otherwise
# XXX: some "real" meals were caught by our filters for closed canteens -> we will manually un-delete them
# XXX: real meals:
    # "eingeschlossene Rezepturen"
    # Pizza mit "kein Käse"
    # "Schaschlikeintopf"
    # "keine Beilage"
    # "Kein Angebot MensaVital, dafür: Schweineschnitzel Toskana (fleischlos: Blumenkohl-Käsebrätling, A,C,G,I) mit Erbsengemüse und Pommes frites oder Kartoffeln"
    # "*** Typisch bayrisch***Hax'n, Weißwurst, 1/2 Hend'l oder Fleischkäse mit verschiedenen Beilagen ***********Heute leider kein Hend´l - Lieferant hat uns vergessen *************"
    # "Geschlossen!! Nur Cafeteria geÃ¶ffnet!!!! Salat Bowl mit Roastbeef,KÃ¼rbispesto und SchafskÃ¤se"
    # "Schweinegulasch mit Schwenk-Kartoffeln oder Reis und Salat - Pizza-Point geschlossen!!"
    # "Pasta-Strecke geschlossen-Ausgabe über Schnitzel&Co: Champignon-Käsesoße"
    # "Seelachsfilet mit Broccoli-Käseauflage, helle SauceQuickeinudeln ( A )"
    # "Ausgabe heute geschlossen, bitte besuchen Sie uns für dieses Angebot an unserer Ausgabe im 1.OG: Pasta Carbonara, dazu italienischer Hartkäse"
    # "Ausgabe geschlossen, bitte besuchen Sie für dieses Angebot unsere Ausgabe im 2.OG: LON XOT CA CHUA - Schweinefleisch mit buntem Wokgemüse, dazu Thaireis"
    # "Ausgabe geschlossen, bitte besuchen Sie für dieses Angebot unsere Ausgabe im 2. OG: GA T XAO CARI DO - Zartes Putenfleisch mit buntem Wokgemüse, dazu Reisnudeln"
    # "Ausgabe geschlossen, bitte besuchen Sie für dieses Angebot unsere Ausgabe im 2. OG: GA HUONG CAM - Zartes Hähnchenfleisch mit buntem Wokgemüse, dazu Basmati Reis"
closed = german_university_meals[german_university_meals["meal_name"].str.contains(pat="geschlossen|entfällt|kein", case=False, regex=True)]
closed_categories = closed["meal_name"].value_counts(dropna=False).to_frame(name="count").reset_index()

# un-mark the "real" meals accidentally caught in closed filter
# delete the remaining meals
closed = closed[~closed["meal_name"].str.contains("Rezeptur|kein Käse|Schaschlik|keine Beilage|mensaVital|Hend'l|Bowl|Pizza-Point|Pasta-Strecke|Seelachsfilet|Hartkäse|Wokgemüse", case=False, regex=True)]
german_university_meals = german_university_meals.drop(index=closed.index)

# filter meals with name "." or "--" -> we can't use str.contains because these characters also appear in a lot of real meals
# we'll filter for any entries that contain just a repetition of special characters (and nothing else) -> regex for convenient approach
closed = german_university_meals[german_university_meals["meal_name"].str.fullmatch(pat=r"\W+")]
closed_categories = closed["meal_name"].value_counts(dropna=False).to_frame(name="count").reset_index()
german_university_meals = german_university_meals.drop(index=closed.index)

# filter for other announcements, exclude salad dishes
announcements = german_university_meals[german_university_meals["meal_name"].str.contains(pat="aufgrund|wir |gäste", case=False, regex=True)]
announcements_categories = announcements["meal_name"].value_counts().to_frame().reset_index()
announcements = announcements[~announcements["meal_name"].str.contains(pat="salat", case=False, regex=True)]
announcements_categories = announcements["meal_name"].value_counts().to_frame().reset_index()

# delete annoucnements
german_university_meals = german_university_meals.drop(index=announcements.index)

#######################################################
# REMOVE WRONGLY-PARSED DATA
#######################################################

# some data points were parsed incorrectly, probably due to special characters being misread etc.
# we will remove them for now, because cleaning them would take too much  effort
sus_meal_categories = german_university_meals[(german_university_meals["meal_category"].str.len() > 50) & (~german_university_meals["meal_category"].str.contains(pat="theke|heute|menü|flex-gericht|mittagsgericht|restaurant|to-go|EG Süd|pro Portion|Ausgabe|Cafeteria|delicious|foodhopper", case=False, regex=True))]
sus_meal_categories_counts = sus_meal_categories["meal_category"].value_counts(dropna=False).to_frame(name="count").reset_index()
print(sus_meal_categories["canteen_name"].value_counts())

# delete entries which were parsed wrongly
german_university_meals = german_university_meals.drop(index=sus_meal_categories.index)

#######################################################
# CLEAN UP PRICE INFORMATION
#######################################################

# prices are a relevant part of our analysis, but information needs to be complete and accurate for us to work with
# XXX: price information for students and employees (biggest group of guests in university canteens) are 80-90% complete, that's quite okay
meals_missing_stats = german_university_meals[["meal_price_student", "meal_price_employee", "meal_price_pupil", "meal_price_other"]].isna().sum().to_frame(name="count")
meals_missing_stats["percentage"] = meals_missing_stats["count"] / german_university_meals.shape[0] * 100

# we do have range issues though: we have prices ranging from 0€ to more than 200€ -> probably typos -> need to be fixed (e.g., imagina calculating an average meal price as indicator and having a 290€ meal messing up ypur calculations)
print(german_university_meals[["meal_price_student", "meal_price_employee", "meal_price_pupil", "meal_price_other"]].describe())

# for prices which are ==0€, we can set them to NA -> they don't really cost 0€, price is just not known
# ATTENTION: be careful not to mix up pd.NA and np.nan -> pd.NA crashes the program
zero_prices = german_university_meals[(german_university_meals[["meal_price_student", "meal_price_employee", "meal_price_pupil", "meal_price_other"]] <= 0).any(axis=1)]
german_university_meals[["meal_price_student", "meal_price_employee", "meal_price_pupil", "meal_price_other"]] = german_university_meals[["meal_price_student", "meal_price_employee", "meal_price_pupil", "meal_price_other"]].replace(to_replace=0, value=np.nan, inplace=False)

# for prices which are >20€, we can also set them to NA -> probably typos, but not worth it to fix manually
high_prices = german_university_meals[(german_university_meals[["meal_price_student", "meal_price_employee", "meal_price_pupil", "meal_price_other"]] > 20).any(axis=1)]
temp = german_university_meals[["meal_price_student", "meal_price_employee", "meal_price_pupil", "meal_price_other"]]
german_university_meals[["meal_price_student", "meal_price_employee", "meal_price_pupil", "meal_price_other"]] = temp.mask(cond=temp > 20, other=np.nan, inplace=False)

########################################################
# SEPARATE DATA INTO SIDE DISH / MAIN DISH / SALAD / DESSERT / (SOUP) / BAKED GOODS / OTHER
#######################################################

# for our analysis it would be ideal if we map each data point to the categories
# mentioned above, as they will probably need to be treated differently
# my first idea and also the most time-effective effort is to use existing "meal_category" feature


# we found out in exploration that data categories are quite slanted, so if we sort the biggest 245 categories, we'll cover 90% of the data -> cost-efficient
# danger: we lose otherwise perfectly fine meals, probably main dishes which could give our analysis more depth, especially if group of side dishes and snacks is comparatively big
category_categories = german_university_meals["meal_category"].value_counts(dropna=False).to_frame(name="count")
category_categories["cum_sum"] = category_categories["count"].cumsum()
category_categories["cum_percent"] = category_categories["cum_sum"] / german_university_meals.shape[0] * 100
category_categories = category_categories.reset_index()

# let's try a rule-based approach for the 99% first -> didn't work out
#category_categories["meal_super_category"] = pd.NA
#category_categories.loc[category_categories["index"].str.contains("brötchen|urknacker|bagel|croissant|lauge|gersterling", regex=True, case=False), "meal_super_category"] = "baked_goods"
#category_categories.loc[category_categories["index"].str.contains("beilage", regex=True, case=False), "meal_super_category"] = "side_dish"

# since a rule-based approach using str.contains didn't work out that well, we will look up category association manually
# takes time, so only for 90% of the data for now -> maybe we can find a better approach using a LM later 
# for the manual lookup following rules were used:
    # check category name, if clear indication towards one class, classify (e.g. category is already named "Beilagen")
    # otherwise take a look at 10 most common entries, decide based on that (if entries are mixed, classify as mixed)
    # salads and soups always listed as own category, even though they can be main dishes and may be contained also in main_dish category
    # if a category contains for example sweets and soups, it will be listed as "mix" for now
    # cafeteria snacks will be sorted into category other for now
    # other also contains any other items not matching the remaining categories (e.g., smoothies)
category_categories.to_csv("data/helper_data/analysis_subset_meal_categories_with_counts.csv")
lookup = german_university_meals[german_university_meals["meal_category"] == 'Wok und Pfanne']
lookup = lookup["meal_name"].value_counts(dropna=False)

# now merge super category back into df -> first read in df, only select needed columns to avoid overhead data
# fill na with dummy value
super_categories = pd.read_csv("data/helper_data/analysis_subset_meal_categories_with_counts_sorted.csv", sep=",", usecols=[1, 5])
super_categories = super_categories.fillna(value="unmatched")

# now merge, set index first, because otherwise we would join feature on feature and lose index of german_university_meals
super_categories = super_categories.set_index("index")
german_university_meals = pd.merge(left=german_university_meals, right=super_categories, how="left", left_on="meal_category", right_index=True)

# distribution of derived feature
dish_type_distribution = german_university_meals["meal_super_category"].value_counts(dropna=False).to_frame(name="count")
dish_type_distribution["percent"] = dish_type_distribution["count"] / german_university_meals.shape[0] * 100

# drop meals marked as "baked_goods" or "other"
german_university_meals = german_university_meals[(german_university_meals["meal_super_category"] != "baked_goods") & (german_university_meals["meal_super_category"] != "other")]

# during manual category sorting, I found some categories that contained malformed data -> remove
german_university_meals = german_university_meals[(german_university_meals["meal_category"] != "GreenCorner") & (german_university_meals["meal_category"] != "MA(h)l was anderes")]

#######################################################
# SELECT NEEDED FEATURES AND SAVE
#######################################################

# canteen_replaced_by is completely empty for our data selection, so we can drop it
# metadata about data creation is probably not relevant either, but we will keep it for now
# our auxiliary feature to_be_deleted, time_difference and price_missing / price_range we can probably drop because we won't need them anymore
german_university_meals = german_university_meals.drop(columns=["canteen_replaced_by"])

# now save
german_university_meals.to_pickle("data/processed_data/meals_cleaned.pkl")
