# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:52:22 2023

@author: Tuni
"""

import pandas as pd
import missingno as msno
#import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.ticker as ticker
import seaborn as sns
import calendar

# options for shell
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
meals_df = pd.read_csv("data/raw_data/meals.csv", sep=",")

# check if everything worked correctly using head() -> too much data to use interactive explorer
# XX: seems okay at first glance, but some features need to be checked further -> what is description, is that even used?
# XX: what is pos? is that even used?
# XX: we have created_at and updated_at -> data could have been updated, does that impact analysis somehow?
# XX: we can't link back directly to the day or the canteen, we need to take the detour via joining with days.csv and then with canteens.csv
# XX: for the first five rows, no prices are given -> is this a decoding error or were prices not tracked during the first years? (can't imagine that, it's a vital piece of information for a menu app)
print("Let's take a first glance at our data to check for reading errors and what we are dealing with")
print(meals_df.head())

# to find out more about the prices (and other mysterious categories) use summary statistics
# XX: prices are not completely empty, but also only filled out roughly about 30% -> it could be that some canteens only have one price category, so total amount of entries is price_student + price_employee + price_pupil + price_other -> needs to be checked
# XX: the ranges for the prices seem off, from -4€ to more than 2,000€ -> maybe test dummy data?
# XX: description is completely empty, can be discarded
# XX: category should be interesting -> check distribution and kind of entries later
# XX: position is filled almost 100% (that's good), but again, the ranges seem off (that's bad) -> all in all not sure if this information will be useful for analysis anyways
# XX: interesting: the ratio of unique to total menu name entries -> only ~5% are unique -> will be interesting to check further with more accuracy of days and canteens 
# XX: as expected, some data types are still wrong
print(meals_df.describe(include="all"))


#######################################################
# THE BASICS: COLUMNS, ROWS, DATA TYPES
#######################################################

# let's check datset size and header quality
# XX: we have a lot of data, more than 11 million rows -> could be challenging to work with
# XX: "id" can probably be renamed to "meal_id" for unambiguous join operations later
print(f"Dataset has shape: {meals_df.shape}")
print(f"Column headers: {meals_df.columns}")

meals_df.rename(columns={"id": "meal_id"}, inplace=True)

# let's check and djust data types (we have already seen above that they are mismatched)
# XX: "id" should be object, 
# XX: description should probably not be a float, but was empty anyways
# XX: "created_at", "updated_at" should be datetime
# XX: category could be a category, but object is fine for now
# XX: day_id should be object
# XX: position should be an int, unless there is some data errors
print("datatypes of meals df are:")
print(meals_df.dtypes)

meals_df["meal_id"] = meals_df["meal_id"].astype("object")
meals_df["created_at"] = pd.to_datetime(meals_df["created_at"], format="%Y-%m-%d %H:%M:%S.%f")
meals_df["updated_at"] = pd.to_datetime(meals_df["updated_at"], format="%Y-%m-%d %H:%M:%S.%f")
meals_df["day_id"] = meals_df["day_id"].astype("object")
meals_df["pos"] = meals_df["pos"].astype("Int64")

# once again, data summary with the correct data types
# XX: "id" is unique and complete, so can be used as index
# XX: "name" distribution should be studied in more detail -> one menu more than 52,000 times? seems off
# XX: a lot more data in later years, we can see it in quantiles of created_at and updated_at
# XX: "day_id" seems off, one day (days are specific to canteens) is referenced more than 540 times? maybe some data was duplicated?
# XX: I already talked about the prices being out of range above
# XX: the same goes for position: max position is 957??
meal_stats = meals_df.describe(include="all", datetime_is_numeric=True)

#######################################################
# MERGING WITH DATE
#######################################################

# for further analysis I think it would be best to merge data with days and canteens to detect patterns
# attention: date is not correctly parsed because of typo, but we will work with date_correct anyways
days_df = pd.read_csv("data/raw_data/days_edited.csv", sep=",", decimal=".", parse_dates=[3,4])

# drop duplicated index column
days_df.drop(columns="Unnamed: 0", inplace=True)

# add corresponding day and canteen id to each menu
meals_df_merged = pd.merge(left=meals_df, right=days_df, how="left",left_on="day_id", right_on="days_id", sort=False)

# remove unneeded feature like duplicated date, closed indicator
meals_df_merged = meals_df_merged.drop(columns=["days_id", "closed"])

#######################################################
# MISSING DATA
#######################################################

# let's take a closer look at missing data, especially of price features
# XX: relevant information for identification like meal name, day id are 100%, also some additional descriptive information like category is 100% complete
# XX: position is almost 100% complete (98.7% to be precise)
# XX: description is 100% empty, prices between 70% and 98% empty -> not good, needs to be investigated further
meals_missing_stats = meals_df.isna().sum().to_frame(name="count")
meals_missing_stats["percentage"] = meals_missing_stats["count"] / meals_df.shape[0] * 100
print("Missing entries per feature")
print(meals_missing_stats)

# visualize missing data graphically
#plt.figure()
#ax = msno.matrix(meals_df, sparkline=False)
#ax.set_title("Overview of missing data of meals.csv", fontsize=24)
#plt.tight_layout()

# we want to find out more about distribution of missing price information
# we will check completeness of prices by applying a heuristic: as long as any price is given, mark as price_missing =  False (use merged df to include date information)
# XX: we have gained some information, but the majority of the prices are still lost (63% where no data is given at all)
meals_df_merged["price_missing"] = meals_df_merged[["price_student", "price_employee", "price_pupil", "price_other"]].isna().all(axis="columns")
meals_missing_stats.loc["price_missing"] = [meals_df_merged["price_missing"].sum(), meals_df_merged["price_missing"].sum() / meals_df_merged.shape[0] * 100]

# let's check if there is some temporal trend
# we will plot price_missing on a time line -> instead of one long line it will be a matrix
temp = meals_df_merged[["price_missing", "date_correct"]].set_index(keys="date_correct", drop=True)
temp = temp.sort_index()

# we need to turn data from one long series into a matrix for visualization to work
# we can do this easily with the numpy bib, we just need to find divisors without remainder
# XX: we can see a trend, but it's not super nice to look at and interprete
# XX: I think it will be nice to build a visualization that indicates the percentage of price information for each month
matrix_data = np.reshape(temp.values, (739, 16111))
plt.figure()
sns.heatmap(matrix_data)

# we will need to group our price_missing information by year and month
# so first, extract year and month from our data
temp = meals_df_merged[["price_missing", "date_correct"]]
temp["month"] = temp["date_correct"].dt.month
temp["year"] = temp["date_correct"].dt.year

# now aggregate in groups per month / year and find the percentage of missing prices in that time period
# also count the number of available menus for each year / month
price_missing_per_year_month = temp.groupby(["year", "month"])["price_missing"].agg(["mean", "count"])
price_missing_per_year_month["percent_number"] = price_missing_per_year_month["mean"] * 100

# pivot to obtain rectangular data set for heatmap, then plot
# XX: we can now see the temporal trends clearly: surprisingly, it is the latest years that don't have price information
# XX: we can also see some range irregularities, data from 1999 and 2024
heatmap_data = pd.pivot(data=price_missing_per_year_month.reset_index(), index="year", columns="month", values="percent_number")
months = [month[:3] for month in calendar.month_name[1:]]
plt.figure()
ax = sns.heatmap(heatmap_data, cmap="crest", linewidth=0.3, square=True, annot=False, xticklabels=months, yticklabels=True, cbar_kws={"label": "Percent [%] of meals served in a given month \n without price information"})
#ax.set_title("Percentage [%] of missing price information \n in meals.csv over collection period", fontsize=24)
ax.set_xlabel("Month")
ax.set_ylabel("Year")

#######################################################
# DUPLICATES
#######################################################

# check for duplicates in the complete data set -> there shouldn't be any, would be errors that skew the analysis
# XX: no duplicates exist, that is good
print(f"Number of completely identical meals: {meals_df.duplicated(keep=False).sum()}")

# check for duplicates in name and date_correct (minimum information necessary to describe a menu) -> shouldn't be the case either, would be an error and skew the analysis
# XX: there is actually 9 million entries that are duplicates under these conditions ... we need to find out why to assess the data quality
# XX: one idea is that meals have common names and happen to be offered on the same day in various canteens -> include canteen ID in duplicated subset
print(f"Number of meals that have identical names and dates: {meals_df_merged.duplicated(subset=['name', 'date_correct'], keep=False).sum()}")
duplicated_entries = meals_df_merged.loc[meals_df_merged.duplicated(subset=['name', 'date_correct'], keep=False)]

# check for duplicates in name, canteen_id and date_correct (minimum information necessary to describe a menu) -> shouldn't be the case either, would be an error and skew the analysis
# XX: we expected to find 0 entries, but we have ~350,000 -> the data is somehow messy and we need to find out why and correct it first, before we can conduct valid analyses with it
# XX: but still better than the 9 million we had before, so canteen ID makes big difference
# XX: if we look at the duplicated entries manually, we can see that all the entries with the malformed date (4012 instead of 2012 as year) are contained twice in the data set -> let's correct that first by removing those entries
print(f"Number of meals that have identical names, canteens and dates: {meals_df_merged.duplicated(subset=['name', 'canteen_id', 'date_correct'], keep=False).sum()}")
duplicated_entries = meals_df_merged.loc[meals_df_merged.duplicated(subset=['name', 'canteen_id', 'date_correct'], keep=False)].sort_values(by=["date_correct", "canteen_id", "name"])

# select date entries starting with 4012 and remove them, then screen remaining duplicates again
# XX: we still have about ~347,000 entries
# XX: meals with no (reconstructable) menu information: Tagesangebot, Angebot des Tages, None, (Heute) Geschlossen, Heute kein Angebot, Buffet, Der Mensaetrieb findet über die Burse statt -> need to be filtered after screening the whole data set
# XX: in general we have menu entries for desserts, salads, side dishes -> desserts can probably be discarded, salads and side dishes should be analyzed separately from main dishes?
# XX: possibly not duplicates: meals with the same name, canteen ID and date, but they belong to different menu categories (e.g., Tellermenü or Wahlgericht; Essen 2 or Essen 3)
duplicates_cleaned = duplicated_entries.loc[~duplicated_entries["date"].str.startswith("4")]
duplicates_cleaned = duplicates_cleaned.loc[duplicates_cleaned.duplicated(subset=['name', 'canteen_id', 'date_correct'], keep=False)]
print(f"Number of meals that have identical names, canteens and dates after removing the malformed year 4012 entries: {duplicates_cleaned.shape[0]}")

duplicated_entries["date"].str.startswith("4").tail()
duplicated_entries.tail()

# I think menus with same name, canteen and date but different categories can be treated as different menu option, but we'll check manually just to be sure
# we'll gather canteens and their menu lines in a list
# XX: we can see some of the already discovered problems (Tellergericht or Wahlgericht), but some canteens clearly just have one category -> must be real duplicates
# XX: we also see a surprising number of categories dealing with baked products -> should be checked for rest of data set and then removed
# XX: after introducing the category feature for the duplicates analysis: the remaining ~77,000 entries really are duplicates
# XX: sometimes we have as many as four repeated entries, some of them differ in the pos feature, but some don't
# XX: we should remove them before starting our analysis
same_menu_different_category = duplicates_cleaned.groupby(["canteen_id", "category"])["name"].nunique()
duplicates_cleaned = duplicates_cleaned.loc[duplicates_cleaned.duplicated(subset=['name', 'canteen_id', 'date_correct', 'category'], keep=False)]
print(f"Number of really duplicated meals (identical names, canteens, dates, categories: {duplicates_cleaned.shape[0]}")
print(duplicates_cleaned.describe(include="all", datetime_is_numeric=True))

##############################################################
# DATA DISTRIBUTION: "name"
##############################################################

# in theory names should be quite unique unless they are common meals (e.g., "Spaghetti Bolognese") or are offered repeatedly
# but we have already seen that that is not the case, so let's check unique values and their counts
# XX: the most common meals are all in French, probably offered every day in every canteen
# XX: the most common ones are also mainly bistro items, sometimes even drinks, so those should definitely be filtered
unique_meals = meals_df_merged["name"].value_counts(dropna=False)

# merge data with canteen information to filter away French items and work mor easily with remaining data
# import canteen csv and drop duplicated indices
canteen_df = pd.read_csv("data/raw_data/canteens_edited.csv")
canteen_df = canteen_df.drop(columns=["Unnamed: 0"], inplace=False)

# rename id and other common entries to avoid confusion while merging
canteen_df.rename(columns={"id": "canteen_id", "name": "canteen_name", "created_at": "canteen_created_at"}, inplace=True)

# now merge, use subset of columns for now to reduce computation time
meals_df_merged = pd.merge(left=meals_df_merged, right=canteen_df[["canteen_id", "canteen_name", "city", "state", "country"]], how="left",left_on="canteen_id", right_on="canteen_id", sort=False)

# let's take a look at meal distribution over countries
# XXX: as meal counts already indicated, we have many (more than half) meals originating in Luxembourg
# XXX: Germany on position number 2 with ~5 mio meals -> should still be enough for our analysis depth
# XXX: we should take a closer look at the NaN category to see if they can be discarded
meal_country_distribution = meals_df_merged["country"].value_counts(dropna=False).rename("count").to_frame()
meal_country_distribution["percentage"] = meal_country_distribution["count"] / meals_df_merged.shape[0] * 100

# add counts and percentages of canteens
canteen_country_distribution = canteen_df["country"].value_counts(dropna=False).rename("count_canteens").to_frame()
canteen_country_distribution["percentage_canteens"] = canteen_country_distribution["count_canteens"] / canteen_df.shape[0] * 100
temp = pd.merge(left=meal_country_distribution, right=canteen_country_distribution, how="outer", left_index=True, right_index=True).sort_values("count", ascending=False)

# plot both distributions in one plot
plt.figure()
ax = temp[["percentage", "percentage_canteens"]].plot(kind="bar", color=layout_color, subplots=True, legend=False, title=["Meals", "Canteens"], ylabel="Percent [%]", xlabel="Countries", sharey=True, figsize=(18,9))
#plt.suptitle("Distribution of meals vs. canteens contained in OpenMensa data over countries", fontsize=26)
for bar_chart in ax:
    bar_chart.bar_label(bar_chart.containers[0], fmt="%.2f", fontsize=18)
    bar_chart.margins(y=0.1)
plt.tight_layout()

# filter NaN meals to see if they need to be manually adapted or can be discarded
# XXX: most data is from obsolete (=deleted) canteens, but the meal information is actually still relevant for our analysis -> maybe we can map the canteens back to their replacement canteen (if there is any), otherwise data probably needs to be discarded because context can't be reconstructed
# XXX: other four canteens can probably be manually corrected in canteens_cleaned.csv
# XXX: other four canteens are also all German
no_countries = meals_df_merged[meals_df_merged["country"].isna()]
print(no_countries["canteen_name"].value_counts(dropna=False))

# back to original task: filter only German canteens and then see frequency of meal names
# XXX: side dishes and salads dominate frequency count -> makes sense -> we need to think about how to include them in our analysis
# XXX: funny enough we have "Currywurst" as meal quite often
# XXX: we have many entries to filter still, e.g. "geschlossen", ".", "entfällt", "--"
# XXX: probably we should also filter all kind of sandwich rolls and other bistro items, all kinds of desserts
# XXX: we also need to deal with spelling duplicates, e.g. "Pommes" and "Pommes frittes"
# XXX: "real" meals as expected ranked rather low (if served every week for ten year count still shoulnd't be over 520, and that quite unrealistic), but popularity differences can be seen (e.g. "Spaghetti "Bolognese"" has a count of 478)
# XXX: ratio of size germany_meals to unique_meals_germany is 567,042 : 5,069,495 = 0.1119 -> so statistically speaking every item is repeated 10 times over 10 years -> but of course only a rough approximation, salads and side dished skew the analysis completely
germany_meals = meals_df_merged[meals_df_merged["country"] == "Germany"]
unique_meals_germany = germany_meals["name"].value_counts(dropna=False).to_frame()

# let's create a boxplot with the number of times each menu appears in dataset
# XXX: we can see that most data is centered at low numbers, but we have many outliers that appear up to 30,000 times
# XXX: we already talked about this above: not all data can be treated equally (main dish, side dish, salads, dessert)
plt.figure()
unique_meals_germany.plot.box()

###############################################################
# DATA DISTRIBUTION: "category"
###############################################################

# let's take a closer look at categories -> it would be especially interesting to see if we can use feature to identify desserts etc
# XXX: we have about ~6,000 categories in complete data set and about ~4,000 for Germany
# XXX: in complete data we again have many French entries, so we can probably only focus on German canteens due to our analysis scope
# XXX: some categories are clearly desserts ("Desserts", "Kuchen und Gebäck"), some clearly bread rolls ("Brötchen" etc.), some side dishes ("Sättigungsbeilage") -> can be used for filtering, but I'm not sure if we will catch all desserts / bread rolls / side dishes with that
# XXX: sometimes we have info on diet ("Vegetarisch"), menu line or day specials -> we probably can't do anything with that -> it would be interesting to see how many meals are tagged with Mensa Vital or other menu lines
# XXX: maybe Tellergericht would be helpful for sorting out side dishes
categories_complete = meals_df["category"].value_counts(dropna=False).to_frame(name="count").reset_index()
categories_germany = germany_meals["category"].value_counts(dropna=False).to_frame(name="count").reset_index()

# quick look at summary statistics of associated meal counts
# XXX: range from 1 associated meal to 444,087 (or 392,947) -> we were expecting not to find any low numbers
print("Complete dataset:")
print(categories_complete["count"].describe())
print("Only German subset:")
print(categories_germany["count"].describe())

# look at cumulativ sum to get a feel for data distribution
# XXX: we can see that a small number of bigger categories cover a lot of meals, many small categories with almost no weight
categories_complete["cum_sum"] = categories_complete["count"].cumsum()
categories_germany["cum_sum"] = categories_germany["count"].cumsum()
plt.figure()
categories_complete["cum_sum"].plot(kind="line")

# even more obvious when looking at cumulative percent
categories_complete["cum_percent"] = categories_complete["cum_sum"] / meals_df_merged.shape[0] * 100
categories_germany["cum_percent"] = categories_germany["cum_sum"] / germany_meals.shape[0] * 100


# let's also take a look at the least frequent entries -> we would actually expect categories to occur rather frequently
# XXX: obvious in German subset: some are actually meal descriptions, some are special day menu lines
# XXX: if the meal descriptions are redundant, they can be discarded
temp = categories_complete.tail(10)
temp = categories_germany.tail(10)

# filter for menu categories with more than 50 characters -> they are probably meal descriptions
# XXX: we can clearly see some menu descriptions that are wrongly parsed and have the meal in the category instead of in name attribute
# XXX: sometimes we have two meals in one entry, one is in name and another one is in description
# XXX: all of these probably need to be fixed manually ...
# XXX: we should try to take out as many canteens with long desriptions (but they are no meal descriptions) -> " Heute kein Angebot Mensa vital, dafür Ausgabe Asia Theke" and "Bio-Menü Alle Menükomponenten auch einzeln erhältlich" -> we can see them in sus_meal_categories_counts
# XXX: after taking them away, we have about 1,671 entries to be manually corrected -> considering the final count of available data, we should clean them or discard them
sus_meal_categories = germany_meals[(germany_meals["category"].str.len() > 50)]
sus_meal_categories_counts = sus_meal_categories["category"].value_counts(dropna=False)
sus_meal_categories = germany_meals[(germany_meals["category"].str.len() > 50) & (~germany_meals["category"].str.contains(pat="theke|heute|menü|flex-gericht|mittagsgericht|restaurant|boissons|emballages|to-go|EG Süd|pro Portion|Ausgabe", case=False, regex=True))]
sus_meal_categories_counts = sus_meal_categories["category"].value_counts(dropna=False)

##################################################################
# DATA DISTRIBUTION: "created_at", "updated_at", "date_corrected"
##################################################################

# for our temporal data it would be interesting to see the distribution over time -> we've already done a suitable plot while looking at the missing prices
# also, it would be good to investigate created_at and updated_at and see whether we can just drop them
# also, we need to find out what is going on with data from before 2010 and why we already have meals for 2024

# (1) meal distribution over time --------------------------------

# we are using the same plot and method as for the missing prices
temp = meals_df_merged[["name", "canteen_id", "date_correct", "country"]]
temp["month"] = temp["date_correct"].dt.month
temp["year"] = temp["date_correct"].dt.year

# in contrast to before, we additionally group by country and also count the number of unique canteens per month
aggregation_per_year_month_country = temp.groupby(["country", "year", "month"], dropna=False)["canteen_id"].agg(["count", "nunique"])
aggregation_per_year_month_country = aggregation_per_year_month_country.reset_index()

# reshape data so that we have values for each country as columns
agg_meals = pd.pivot(data=aggregation_per_year_month_country, index=["year", "month"], columns="country", values="count")
agg_canteens = pd.pivot(data=aggregation_per_year_month_country, index=["year", "month"], columns="country", values="nunique")

# add total count for each time series
agg_meals["total"] = agg_meals.sum(axis=1)
agg_canteens["total"] = agg_canteens.sum(axis=1)

# quick summary statistics
print(agg_meals.describe())

# prepare labels and shared colorbar for plot
months = [month[:3] for month in calendar.month_name[1:]]
#vmin = agg_meals.min(axis=1).min() # don't use vmin/vmax, as comparison within countries won't be possible anymore (scale is too small)
#vmax = agg_meals.max(axis=1).max()

# prepare canvas for combined plot
fig, ax = plt.subplots(1,2)

# now plot meal availability, first for complete dataset
# XX: in general an increasing trend
# XX: no sharp decline due to covid -> could be worth it to plot growth rate compared to prior month to see trends more clearly
heatmap_data = pd.pivot(data=agg_meals.reset_index(), index="year", columns="month", values="total")
ax1 = sns.heatmap(heatmap_data, ax=ax[0], cmap="crest", linewidth=0.3, square=True, annot=False, xticklabels=months, yticklabels=True, cbar_kws={"label": "Meal count", "format": "{x:,.0f}"})
cbar = ax1.collections[0].colorbar
cbar.set_label("Meal count / month", labelpad=15)
ax1.set(title="(a) Complete dataset", xlabel="Month", ylabel="Year")
#plt.tight_layout()

# same plot, but only for German canteens
# XXX: we can see the hiatus due to Covid clearly
# XXX: up to 70,000 items per month,  that is quite a lot -> but we haven't removed side dishes etc yet
heatmap_data = pd.pivot(data=agg_meals.reset_index(), index="year", columns="month", values="Germany")
ax2 = sns.heatmap(heatmap_data, ax=ax[1], cmap="crest", linewidth=0.3, square=True, annot=False, xticklabels=months, yticklabels=True, cbar_kws={"label": "Meal count", "format": "{x:,.0f}"})
cbar = ax2.collections[0].colorbar
cbar.set_label("Meal count / month", labelpad=15)
ax2.set(title="(b) German data subset", xlabel="Month", ylabel="Year")
#ax.set(title="Available meals per month (only Germany)", xlabel="Month", ylabel="Year")
plt.subplots_adjust(wspace=0.5)

# same plot, but with the number of available canteens per month
# XXX: we can see a pretty smooth trend up until 2020, then sharp decrease due to Covid
# XXX: in 2022, we see a sharp jump in the number of tracked canteens
heatmap_data = pd.pivot(data=agg_canteens.reset_index(), index="year", columns="month", values="Germany")
plt.figure()
ax = sns.heatmap(heatmap_data, cmap="flare", linewidth=0.3, square=True, annot=False, xticklabels=months)
ax.set(title="Number of tracked unique canteens per month and year - only German canteens", xlabel="Month", ylabel="Year")

# (2) relation between created_at and updated_at --------------------------------

# let's turn to created_at and updated_at -> let's check out the time difference between these two columns
# XXX: for 75% of the data, creation and update happens on the same day -> we do have at least one outlier though with almost a year difference
germany_meals["time_difference_creation_update"] = germany_meals["updated_at"] - germany_meals["created_at"]
print(germany_meals["time_difference_creation_update"].describe())

# let's take a closer look at meals with a large time difference
# XXX: percentage of German meals with time difference bigger than one day is ~22%
# XXX: meals with largest time difference seem otherwise normal though, they were "just" created one year before their serving date
# XXX: sorted by time_difference we can see that universities repeat themselves -> maybe a special way the parser works? but seems not relevant
temp = germany_meals[germany_meals["time_difference_creation_update"].dt.days > 1]
susp_meals = temp["canteen_name"].value_counts(dropna=False)

# just to check, compare created_at and date_correct
# XXX: looks similar, but we have days that were served before they were created -> that actuallly shouldn't be the case ...
# XXX: I guess since canteens usually publish meals for every week, you probably could create the entry in database up to 7 days later
# XXX: we can see that only the lowest quartile is above a one day delay
# XXX: lowest quartile contaisn extreme cases like data from supposedly 1999, but then drops down to 360 days and then to 30 days
germany_meals["time_difference_creation_serving"] = germany_meals["date_correct"] - germany_meals["created_at"]
print(germany_meals["time_difference_creation_serving"].describe())
temp = germany_meals[germany_meals["time_difference_creation_serving"].dt.days < 0]
print(temp["time_difference_creation_serving"].describe())

# (3) data outside collection time period --------------------------------

# the large time spans between creation update and serving seem weird, but for now we can probably ignore them
# as long as meal was really served as it is contained in meals.csv, we don't really care when the entry was updated/created
# something that does seem weird though, is data from before when the first canteen was created and after the copy of the DB was given
# we'll try to filter that data using canteen with ID == 1 (the earliest canteen)
# convert "canteen_created_at" to pd.datetime first so that we can work with it more conveniently
canteen_df["canteen_created_at"] = pd.to_datetime(canteen_df["canteen_created_at"], format="%Y-%m-%d %H:%M:%S.%f")
om_start_date = canteen_df.loc[canteen_df["canteen_id"] == 1, "canteen_created_at"].iloc[0]
om_stop_date = pd.Timestamp(year=2023, month=8, day=15)

# filter meals that were served before creation of DB
# XXX: meals from august 2012 should be fine, it is plausible that data is still available up to two weeks later -> floor om_start_date
out_of_timeframe = meals_df_merged[meals_df_merged["date_correct"] < om_start_date]
om_start_date = om_start_date.to_period("M").to_timestamp()

# now analize again with floored start date
# XXX: we have 610 entries, all of them from German canteens
# XXX: mainly canteens from Studierendenwerk Aachen -> maybe a special way their API works
# XXX: all in all: not relevant due to low amount, can be deleted
out_of_timeframe = meals_df_merged[meals_df_merged["date_correct"] < om_start_date]
print(out_of_timeframe["country"].value_counts(dropna=False))
print(out_of_timeframe["canteen_name"].value_counts(dropna=False))

# the same but with stop date
# XXX: we have a surprising amount of 27,000 meals
# XXX: but right away I can see the Luxembourgian entries -> which of course can be planned into the future since they also contain drinks and other unchanging snacks
out_of_timeframe = meals_df_merged[meals_df_merged["date_correct"] > om_stop_date]
print(out_of_timeframe["country"].value_counts(dropna=False))

# only German canteens
# XXX: we drastically reduced the amount of entries, now only 1,800
# XXX: I have a feeling many entries are from Würzburg and Bamberg, created more than a year earlier, sometimes even with serving on Saturdays -> seems wrong
# XXX: I think data up to the end of the month should be unproblematic, it's plausible that meal plans extend that far ahead
# XXX: even with current end date we only lose 1,800 data entries, which is a very small amount compared to the time spent trying to figure out why they were created so far in advance
out_of_timeframe = meals_df_merged[(meals_df_merged["date_correct"] > om_stop_date) & (meals_df_merged["country"] == "Germany")]
print(out_of_timeframe["canteen_name"].value_counts(dropna=False))

# same filter again with slightly edited stop date (allow for a little bit of planning ahead)
# XXX: no more Luxembourgian canteens
om_stop_date = pd.Timestamp(year=2023, month=9, day=1)
out_of_timeframe = meals_df_merged[meals_df_merged["date_correct"] > om_stop_date]
print(out_of_timeframe["country"].value_counts(dropna=False))

# only German meals
# XXX: I think it's safe to just remove these entries from DF
# XXX: obvious, though, that most canteens belong to Studierendenwerk Würzburg
out_of_timeframe = meals_df_merged[(meals_df_merged["date_correct"] > om_stop_date) & (meals_df_merged["country"] == "Germany")]
print(out_of_timeframe["canteen_name"].value_counts(dropna=False))


# a thing that seems weird is the menus that were updated more than one year in advance
# we'll check to confirm that this only happened with the meals from after August 2023
# XXX: negative numbers are probably from meals served in 1999
# XXX: only a small portion of data was created that far in advance (not more than 25%) -> check upper quantiles in more detail
germany_meals["time_difference_update_serving"] = germany_meals["date_correct"] - germany_meals["updated_at"]
print(germany_meals["time_difference_update_serving"].describe())

# look at upper quantiles in 5% steps
# XXX: as I said before, anything up to 1-2 months ahead is probably normal operative planning (since ingredients need to be bought, etc.)
quantiles = [x/100 for x in range(70, 100, 5)]
print(germany_meals["time_difference_update_serving"].describe(percentiles=quantiles))

# once again a closer look at the upper quantiles
# XXX: it seems like it is really just an edge phenomenon
quantiles = [x/100 for x in range(90, 100)]
print(germany_meals["time_difference_update_serving"].describe(percentiles=quantiles))

# take a look at just the upper 1% to see if all the data is from > Aug 2023
# XXX: in the boxplot we can see two groups, one of rather realistic seeming data updated about ~3 months prior to serving
# XXX: and the other group of increasingly unrealistic data updated more than one year prior
susp_data = germany_meals[germany_meals["time_difference_update_serving"].dt.days > 45]
plt.figure()
susp_data["time_difference_update_serving"].dt.days.to_frame().plot(kind="box")

# XXX: only about 1,700 entries, we can delete those without losing too much data and without spending too much time onf figuring out if entries are valid or not
# XXX: not only recent data, but consistent over years, especially towards last five years
# XXX: it is notworthy that all canteens containig to that data are operated by Studierendenwerk Würzburg -> so they probably all use the same parser -> parser either really is broken or canteens really do plan that long ahead
# XXX: I can't find anything on the site being parsed, so maybe parser is broken?  -> checked the parser source code, there is an error in manual setting of year
susp_data = germany_meals[germany_meals["time_difference_update_serving"].dt.days > 200]
print(susp_data["date_correct"].describe(datetime_is_numeric=True))
print(susp_data["canteen_name"].value_counts(dropna=False))

###############################################################
# DATA DISTRIBUTION: prices
###############################################################

# (1) overview of anomalies -----------------------------------

# some of the prices show weird range anomalities -> meals cost -4€ or more than 100€
# we'll take a closer look
prices_df = meals_df_merged[["meal_id", "name", "price_student", "price_employee", "price_pupil", "price_other", "date_correct", "canteen_name", "city", "country"]]

# first we will exclude NaN, Swedish and Swiss canteens because of different currencies
# we only lose about 5,000,000 entries, so should be fine
prices_df = prices_df[~(prices_df["country"].isna() | (prices_df["country"] == "Sweden") | (prices_df["country"] == "Switzerland"))]

# look at summary stats and visualize as boxplot
# XXX: price_student is definitely the most extreme with regards to outlier ranges
# XXX: for price_pupil and price_employee you can still see the boxes, even though they also exhibit strong outliers (from pandas doc: "By default, they [annot: the whiskers] extend no more than 1.5 * IQR (IQR = Q3 - Q1) from the edges of the box, ending at the farthest data point within that interval")
print(prices_df[["price_student", "price_employee", "price_pupil", "price_other"]].describe())
plt.figure()
ax = prices_df[["price_student", "price_employee", "price_pupil", "price_other"]].plot(kind="box", subplots=True, sharey=False, ylabel="Price [€]", color="black", patch_artist=True, boxprops=dict(facecolor=layout_color), notch=True)
# title="Distribution of price features contained in meals.csv"
plt.subplots_adjust(wspace=0.5)

# let's try the same but only for German canteens -> I'm wondering if it is maybe due to currency issues or maybe the company canteens in Switzerland? even though outliers shouldn't be that extreme
# XXX: nothing much changes
ax = prices_df.loc[prices_df["country"] == "Germany", ["price_student", "price_employee", "price_pupil", "price_other"]].plot(kind="box", subplots=True, sharey=False, title="Distribution of prices features contained in meals.csv (Germany only)", ylabel="Price [€]", color="black", patch_artist=True, boxprops=dict(facecolor="gray"))

# (2) data in upper price range --------------------------------

# let's take a more detailed look at the upper quantiles using qcut - let's start with 10 bins first
# XXX: everything still "normal" (<10€) for now, so let's decrease the step size
quantiles = [x/10 for x in range(10)]
print(prices_df[["price_student", "price_employee", "price_pupil", "price_other"]].describe(percentiles=quantiles))

# XXX: it is the upper 1% that is weird for all price features, other prices are below <16€ (still expensive, but could be the case for a restaurant like canteen)
quantiles = [x/100 for x in range(90, 100)]
print(prices_df[["price_student", "price_employee", "price_pupil", "price_other"]].describe(percentiles=quantiles))

# sort data by descending price_student
# XXX: we can cleary see that the top 13 meals contain errors regarding prices, maybe typos or else
# XXX: often just price_student is out of range and the rest of the prices are "normal" -> so clearly errors -> but no way for us to reconstruct, i would just delete teh price info since it is just 15 entries
# XXX: below top 15 we can see many menus priced between 10-17€/CHF, but they are mainly high-quality meat dishes (burgers, steaks, etc - sometimes even organic), and since there is so many I believe the prices are maybe accurate
temp = prices_df.sort_values(by="price_student", ascending=False, inplace=False)
prices_df["max_price"] = prices_df[["price_student", "price_employee", "price_pupil", "price_other"]].max(axis=1, skipna=True)
temp = prices_df.sort_values(by="max_price", ascending=False, inplace=False)
print(temp.head(50))

# same analysis, but only German canteens
# XXX: similar results as above, only a handful of obviously incorrect entries -> we should set the prices to NA
# XXX: many high-priced menues from "restaurants" (Jena, Zur Rosen; Hannover, Restaurant c.t.) and/or high-quality meat dishes
# XXX: we also have the canteen of Hochschule Ulm a surprising amount of times -> I wonder if it is maybe some kind of all-you-can-eat menu? some menus even contain "buffet" in the name -> in that case price is accurate
# XXX: an idea would be to filter menus where the price between the different categories varies more than ~5€ -> those are probably comma errors
temp_germany = prices_df[prices_df["country"] == "Germany"].sort_values(by="price_student", ascending=False, inplace=False)
temp_germany = prices_df[prices_df["country"] == "Germany"].sort_values(by="max_price", ascending=False, inplace=False)
print(temp_germany.head(50))

# filter menus where the price between different categories is more than 5€ using min and max values
# XXX: we can see different reasons for extreme price ranges:
    # XXX: one reason is price typos, for example 26€ instead of 2.60€ -> I guess delete price information completely because reconstruction is only manually possible (too much work for the benefit)
    # XXX: another reason is external prices which can be more than double the student prices -> these meals are fine, maybe we could raise the threshold to 6€?
    # XXX: sometimes price_other is way lower -> for university canteens that shouldn't be the case, the external prce is always the highest -> probably delete price information?
    # XXX: sometimes instead of a missing price being NA it is 0.00 instead, causing extreme ranges -> won't be a problem anymore once we set 0 prices to NA
prices_df["price_range"] = prices_df[["price_student", "price_employee", "price_pupil", "price_other"]].max(axis=1, skipna=True).subtract(prices_df[["price_student", "price_employee", "price_pupil", "price_other"]].min(axis=1, skipna=True))
susp = prices_df[prices_df["price_range"] > 5]

# take a look only at German canteens with meals with extreme ranges
# XXX: we suddenly have a lot less meals, only 1385 -> same issues as above
susp_germany = prices_df[(prices_df["price_range"] > 5) & (prices_df["country"] == "Germany")]

# (3) data in lower price range --------------------------------

# filter prices <=0 -> seems a little bit unrealistic
# XXX: prices that are == 0 are meals that have no price information -> we should probably set them to NA to avoid biases in statistics calculated
zero_prices = prices_df.loc[(prices_df[["price_student", "price_employee", "price_pupil", "price_other"]] <= 0).any(axis=1)]
print(zero_prices["price_range"].describe())

# filter prices <0 
# XXX: mostly denote closed canteens in Switzerland -> we can completely ignore them for our analysis
# XXX: because we excluded Switzerland above, we can't see this anymore
below_zero_prices = prices_df.loc[(prices_df[["price_student", "price_employee", "price_pupil", "price_other"]] < 0).any(axis=1)]

# check for German entries in both categories
# XXX: still a lot of 0 prices (see conclusion above), no below 0 prices -> probably really just a way Swiss canteen information was managed
zero_prices_germany = zero_prices[zero_prices["country"] == "Germany"]
below_zero_prices_germany = below_zero_prices[below_zero_prices["country"] == "Germany"]

#########################################################################
# SAVE
#########################################################################

# save meals (especially with country feature added) for analysis of notes.csv
meals_df_merged.to_csv("data/raw_data/meals_edited.csv")
