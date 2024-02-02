# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:47:14 2023

@author: Tuni
"""

import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# this file is for extracting metrics to analyze open mensa data

plt.rcParams.update({"axes.labelsize": 22, # size of axis labels
                     "axes.titlesize": 24, # size of figure title
                     "xtick.labelsize": 20, # size of axes annotation of ticks
                     "ytick.labelsize": 20, # size of axis annotation of tickes
                     "figure.titlesize": 24, # plt.suptitle size
                     })
layout_color = "#004E8A"

####################################################################
# LOAD DATA CORRECTLY
####################################################################

# first we will load our cleaned data
# we can just use the last csv that contains meals and notes as all other features
# are joined into it -> remember to parse dates already to save some effort later

meals_df = pd.read_pickle("data/processed_data/meals_cleaned_with_notes.pkl")

# check if everything worked
print(meals_df.head())
print(meals_df.columns)

# replace N/A with np.nan since while reading pickle file na values are not automatically parsed
meals_df[["notes_list", "notes_list_90"]] = meals_df[["notes_list", "notes_list_90"]].replace(to_replace="N/A", value=np.nan)
print(meals_df.head())

####################################################################
# CLASSIFY DIETARY TYPE
####################################################################

# import note category for rule-based approach
note_categories = pd.read_csv("data/helper_data/analysis_subset_notes_categories_with_counts.csv")

# (1) test run -----------------------------------------------------

# based on 90% of covered meal-note associations, we will derive rules for vegan and vegetarian dishes
# ATTENTION: vegan meals are not considered vegetarian by rules
meals_df["is_vegetarian"] = meals_df["notes_list_90"].str.contains(pat="vegetarisch|ohne Fleisch|fleischlos|kein Fleisch|ovo-lacto-vegetabil|OLV", case=False, regex=True)
meals_df["is_vegan"] = meals_df["notes_list_90"].str.contains(pat="vegan", case=False, regex=True)

# the remaining dishes will be marked as omnivorous
meals_df["is_omnivorous"] = (~(meals_df["is_vegan"] | meals_df["is_vegetarian"]))

# check if tag "vegan" and vegetarian tags are used mutually exclusively
# XXX: they are not, so I think it's better to mark vegan dishes as vegetarian dishes as well, then with subtraction we get the count of "only" vegetarian dishes
temp = meals_df[meals_df["is_vegan"] & meals_df["is_vegetarian"]]

# (2) final classification -----------------------------------------

# based on 90% of covered meal-note associations, we will derive rules for vegan and vegetarian dishes
# ATTENTION: vegan meals are considered vegetarian by rules
# ATTENTION: declare na=False to assign meals without notes boolean value instead of np.nan
meals_df["is_vegetarian"] = meals_df["notes_list_90"].str.contains(pat="vegetarisch|ohne Fleisch|fleischlos|kein Fleisch|ovo-lacto-vegetabil|OLV|vegan", case=False, regex=True, na=False)
meals_df["is_vegan"] = meals_df["notes_list_90"].str.contains(pat="vegan", case=False, regex=True, na=False)

# non-vegetarian dishes will be marked as omnivorous
meals_df["is_omnivorous"] = ~meals_df["is_vegetarian"]

# (3) analysis -----------------------------------------------------

# sanity check: are notes used correctly? -> are there omnivorous desserts?
# build mutually exclusive dish-type feature and then calculate contingency table
# ATTENTION: take care of correct order of vegetarian and vegan conversion, otherwise vegan entries will get deleted
meals_df.loc[meals_df["is_vegetarian"], "dietary_type"] = "vegetarian"
meals_df.loc[meals_df["is_vegan"], "dietary_type"] = "vegan"
meals_df.loc[meals_df["is_omnivorous"], "dietary_type"] = "omnivorous"

# now the contingency table
# XXX: we do have omnivorous desserts ...
temp = pd.crosstab(meals_df["dietary_type"], meals_df["meal_super_category"])

# distribution of meals over the three dietary types
# ATTENTION: as vegan dishes are also vegetarian dishes, sum exceeds 100%
# XXX: a lot more omnivorous dishes, but we know from above, data head glimpses that labelling is not 100% correct
dietary_stats = meals_df[["is_vegan", "is_vegetarian", "is_omnivorous"]].sum().to_frame(name="count")
dietary_stats["percent"] = dietary_stats["count"] / meals_df.shape[0] * 100

# plot distribution
ax = dietary_stats["percent"].plot(kind="bar", color=layout_color, xlabel="Dietary type", ylabel="Percent [%]", rot=0)
#ax.set_title("Distribution of dietary types over meal records contained in analysis data subset")
ax.set_xlabel("Dietary type", labelpad=15)
ax.bar_label(ax.containers[0], fmt="%.2f", fontsize=18)
ax.margins(y=0.1)
plt.tight_layout()

####################################################################
# EXTRACT FEATURE: avg count (main) meals / day
####################################################################

# one feature that would be interesting to see is how much freedom of choice we have in each canteen
# first, we will in general calculate the number of meals offered (excluding baked goods as they are more cafeteria items)
# second, we will calculate the number of main dishes offered (probably the more relevant indicator, as number of desserts e.g. is more of an add-on)
# we will aggregate each metric to monthly level, because otherwise dashboard will probably be overloaded

# ------------------ avg number of meals per day -------------------

# first we need to extract baked goods as they are not part of our analysis
subset = meals_df[meals_df["meal_super_category"] != "baked_goods"]

# next I will average per day first and then per month because we may not have data for every day
avg_count_meals = subset.groupby(by=["canteen_id", "date_correct"])["meal_name"].count().reset_index()
print(avg_count_meals.head())

# now extract month and year, then group by those values again and average
avg_count_meals["month"] = avg_count_meals["date_correct"].dt.month
avg_count_meals["year"] = avg_count_meals["date_correct"].dt.year
print(avg_count_meals.head())
avg_count_meals = avg_count_meals.groupby(by=["canteen_id", "year", "month"])["meal_name"].mean().rename("avg_count_meals").reset_index()
print(avg_count_meals.head())

# ------------------ avg number of main dishes per day -------------------

# same process as above, but this time we will additionally groubpy meal super category
avg_count_main_dishes = meals_df.groupby(by=["canteen_id", "date_correct", "meal_super_category"])["meal_name"].count().reset_index()
print(avg_count_main_dishes.head())

# now extract month and year, then group by those values again and average
avg_count_main_dishes["month"] = avg_count_main_dishes["date_correct"].dt.month
avg_count_main_dishes["year"] = avg_count_main_dishes["date_correct"].dt.year
print(avg_count_main_dishes.head())
avg_count_main_dishes = avg_count_main_dishes.groupby(by=["canteen_id", "year", "month","meal_super_category"])["meal_name"].mean().rename("avg_count_main_dishes").reset_index()
print(avg_count_main_dishes.head())

# we are only interested in main dishes for now, so filter
# then drop column "meal_super_category" to obtain format suitable for merging into results_df (see below)
avg_count_main_dishes = avg_count_main_dishes[avg_count_main_dishes["meal_super_category"] == "main_dish"]
avg_count_main_dishes = avg_count_main_dishes.drop(columns="meal_super_category")

####################################################################
# EXTRACT FEATURE: percent / count of vegan and vegetarian meals / day
####################################################################

# another interesting set of indicators is the availability of vegetarian and vegan meals
# we will calculate two indicators for each, the percentage (good for comparisons over time and between canteens)
# and the total number of options that one can select from (relevant constraint if too low in everyday life)

subset = meals_df



# ------------------ avg number and percent of vegetarian dishes (excl. desserts + baked goods per day -------------------

# using note_categories (cum 90%), find rule that matches all vegetarian tags
subset["is_vegetarian"] = subset["notes_list_90"].str.contains(pat="vegetarisch|ohne Fleisch|fleischlos|kein Fleisch|ovo-lacto-vegetabil|OLV", case=False, regex=True)

# I guess this may not be completely accurate, as we should maybe focus only on main dihes? or at least include distinction into analysis (but we did already remove desserts at least)
# extract count of vegetarian dishses and total count of dishes that day
# then calculate percent vegetarian
avg_count_vegetarian_dishes = subset.groupby(by=["canteen_id", "date_correct"]).agg(count_vegetarian=("is_vegetarian", "sum"), total_count=("meal_name", "count")).reset_index()
avg_count_vegetarian_dishes["percent_vegetarian"] = avg_count_vegetarian_dishes["count_vegetarian"] / avg_count_vegetarian_dishes["total_count"] * 100

# extract month and year to calculate averages, then apply groupby and average
avg_count_vegetarian_dishes["month"] = avg_count_vegetarian_dishes["date_correct"].dt.month
avg_count_vegetarian_dishes["year"] = avg_count_vegetarian_dishes["date_correct"].dt.year
avg_count_vegetarian_dishes = avg_count_vegetarian_dishes.groupby(by=["canteen_id", "year", "month"]).agg(avg_count_vegetarian=("count_vegetarian", "mean"), avg_percent_vegetarian=("percent_vegetarian", "mean")).reset_index()

# ------------------ avg number and percent of vegan dishes (excl. desserts + baked goods per day -------------------

# same procedure as above, but different rule
subset["is_vegan"] = subset["notes_list_90"].str.contains(pat="vegan", case=False, regex=True)

# find total count and percent per tracked day
avg_count_vegan_dishes = subset.groupby(by=["canteen_id", "date_correct"]).agg(count_vegan=("is_vegan", "sum"), total_count=("meal_name", "count")).reset_index()
avg_count_vegan_dishes["percent_vegan"] = avg_count_vegan_dishes["count_vegan"] / avg_count_vegan_dishes["total_count"] * 100

# extract month and year to calculate averages, then apply groupby and average
avg_count_vegan_dishes["month"] = avg_count_vegan_dishes["date_correct"].dt.month
avg_count_vegan_dishes["year"] = avg_count_vegan_dishes["date_correct"].dt.year
avg_count_vegan_dishes = avg_count_vegan_dishes.groupby(by=["canteen_id", "year", "month"]).agg(avg_count_vegan=("count_vegan", "mean"), avg_percent_vegan=("percent_vegan", "mean")).reset_index()

# mark as omnivorous if neither vegetarian nor vegan
subset["is_omnivorous"] = (~(subset["is_vegan"] | subset["is_vegetarian"]))
print(subset.tail())

# check if tag "vegan" and vegetarian tags are used mutually exclusive
test = subset[subset["is_vegan"] & subset["is_vegetarian"]]

test2 = subset[["is_vegan", "is_vegetarian", "is_omnivorous"]].sum().to_frame(name="count")
test2["percent"] = test2["count"] / subset.shape[0] * 100

####################################################################
# EXTRACT FEATURE: average price per main dish / day
####################################################################

# an interesting component of our analysis are price trends
# even better would be price trends within groups (e.g., vegetarian / non-vegetarian)
# for now we'll focus on price developments within main dishes and only on student prices

# group by canteen, date and the meal cateory, then find average student price per group for each day
avg_price_dish_categories = meals_df.groupby(by=["canteen_id", "date_correct", "meal_super_category"])["meal_price_student"].mean().reset_index()

# now we'll take the monthly average of the averaged daily prices
avg_price_dish_categories["month"] = avg_price_dish_categories["date_correct"].dt.month
avg_price_dish_categories["year"] = avg_price_dish_categories["date_correct"].dt.year
avg_price_dish_categories = avg_price_dish_categories.groupby(by=["canteen_id", "year", "month", "meal_super_category"])["meal_price_student"].mean().reset_index()

# filter only main dish meal prices, then drop super category
avg_price_dish_categories = avg_price_dish_categories[avg_price_dish_categories["meal_super_category"] == "main_dish"]
avg_price_dish_categories = avg_price_dish_categories.drop(columns="meal_super_category")

####################################################################
# EXTRACT FEATURE: count / percent of whole grain meals / day
####################################################################

# the German nutrition guidelines specifically recommend whole grain products where possible
# so we'll try to quantify the supply of meals with whole grain products

# really important this time to exclude baked goods (I'm assuming whole grain percentage in that group is higher than in the rest of the population)
# we'll also exclude desserts because they are not necessary more healthier if they contain whole grains
# TODO: we should also exclude salads I guess, they don't contain any grains at all, so for calculating the percentage, they would skew the results
subset = meals_df[(meals_df["meal_super_category"] != "baked_goods") & (meals_df["meal_super_category"] != "dessert")]

# we will apply a simple rule: if meal_name contains "Vollkorn", mark as whole grain (without recipe there is no other way for us to know)
subset["contains_whole_grain"] = subset["meal_name"].str.contains(pat="Vollkorn", case=False, regex=True)

# extract count of whole grain dishses and total count of dishes that day
# then calculate percent whole grain
avg_count_whole_grain = subset.groupby(by=["canteen_id", "date_correct"]).agg(count_whole_grain=("contains_whole_grain", "sum"), total_count=("meal_name", "count")).reset_index()
avg_count_whole_grain["percent_whole_grain"] = avg_count_whole_grain["count_whole_grain"] / avg_count_whole_grain["total_count"] * 100

# extract month and year to calculate averages, then apply groupby and average
avg_count_whole_grain["month"] = avg_count_whole_grain["date_correct"].dt.month
avg_count_whole_grain["year"] = avg_count_whole_grain["date_correct"].dt.year
avg_count_whole_grain = avg_count_whole_grain.groupby(by=["canteen_id", "year", "month"]).agg(avg_count_whole_grain=("count_whole_grain", "mean"), avg_percent_whole_grain=("percent_whole_grain", "mean")).reset_index()

####################################################################
# MERGE METRICS TOGETHER
####################################################################

# we'll collect all created metrics in one df that has canteen_id, year and month (as index) and then the metric columns
# missing months will be filled up with na

# since year and month are just numbers, it's easiest to just create dates made of two int lists
months = [x for x in range(1, 13)]
years = [x for x in range(2012, 2024)]
canteens = meals_df["canteen_id"].unique()

# now take cartesian product to obtain list with all combinations of canteens and dates
# then turn list into df
results_list = list(itertools.product(canteens, years, months))
results_df = pd.DataFrame(results_list, columns=["canteen_id", "year", "month"])

# filter rows out of analysis timeframe (< 08/2012 or > 08/2023)
results_df = results_df[(results_df["year"] != 2012) | (results_df["month"] >= 8)]
results_df = results_df[(results_df["year"] != 2023) | (results_df["month"] <= 8)]


# create list of monthly timestamps that we are interested in
# ATTENTION: format for start and end date is MM/DD/YYYY
#dates = pd.date_range(start="08/01/2012", end="09/01/2023", freq="M", name="dates").to_frame()

# we can use a cross merge to obtain cartesian product between months and canteens
#canteens = pd.Series(meals_df["canteen_id"].unique(), name="canteens")
#test = pd.merge(left = canteens, right=dates, how="cross")

# now we can merge our indicator DFs into results_df
results_df = pd.merge(left=results_df, right=avg_count_meals, how="left", left_on=["canteen_id", "year", "month"], right_on=["canteen_id", "year", "month"])
results_df = pd.merge(left=results_df, right=avg_count_main_dishes, how="left", left_on=["canteen_id", "year", "month"], right_on=["canteen_id", "year", "month"])
results_df = pd.merge(left=results_df, right=avg_count_vegetarian_dishes, how="left", left_on=["canteen_id", "year", "month"], right_on=["canteen_id", "year", "month"])
results_df = pd.merge(left=results_df, right=avg_count_vegan_dishes, how="left", left_on=["canteen_id", "year", "month"], right_on=["canteen_id", "year", "month"])
results_df = pd.merge(left=results_df, right=avg_price_dish_categories, how="left", left_on=["canteen_id", "year", "month"], right_on=["canteen_id", "year", "month"])
results_df = pd.merge(left=results_df, right=avg_count_whole_grain, how="left", left_on=["canteen_id", "year", "month"], right_on=["canteen_id", "year", "month"])

####################################################################
# SAVE METRICS
####################################################################

results_df.to_csv("data/indicators/indicators.csv")

