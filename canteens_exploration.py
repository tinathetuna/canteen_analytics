# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:31:26 2023

@author: Tuni
"""

import pandas as pd
import missingno as msno
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import contextily as cx

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# options for plotting
# ATTENTION: for figures included in thesis, I commented out the figure title, but for individual use I would recommend printing it for better orientation
plt.rcParams.update({"axes.labelsize": 22, # size of axis labels
                     "axes.titlesize": 24, # size of figure title
                     "xtick.labelsize": 20, # size of axes annotation of ticks
                     "ytick.labelsize": 20, # size of axis annotation of tickes
                     "figure.titlesize": 24, # plt.suptitle size
                     })
layout_color = "#004E8A" # color of thesis layout for color theme

#######################################################
# READ IN DATA
#######################################################

# read in CSV, take care to declare delimiter correctly (it's easy in our case, just a normal ",")
canteen_df = pd.read_csv("data/raw_data/canteens.csv", sep=",", index_col=None, decimal=".")

#######################################################
# THE BASICS: COLUMNS, ROWS, DATA TYPES, GIBBERISH
#######################################################

# let's print the first five rows to see if reading was successful and what our data looks like
# XX: everything worked fine, we have quite some NaNs
print("Let's take a first glance at our data to check for readinng errors and what we are dealing with")
print(canteen_df.head())

# let's check datset size and header quality
# XX: compared to version from API we have more columns and also more rows
# XX: new columns are: "created_at", "updated_at", "last_fetched_at", "state",
# XX: "phone", "email", "availability", "openingTimes", "replaced_by"
# XX: "latitude" and "longitude" are separate, that's more convenient to work with
# XX: names seem okay to work with, only "openingTimes" -> "state" and "availability" should be further investigated, not sure what they indicate
print(f"Dataset has shape: {canteen_df.shape}")
print(f"Column headers: {canteen_df.columns}")

# data summary for a feel of distribution
# XXX: we have some wrong data types that hinder interpretation
canteen_stats = canteen_df.describe(include="all")

# let's check our data types for later data processing
# XX: "id" should be object, 
# XX: "created_at", "updated_at" and "last_fetched_at" should be datetime
# XX: "replaced_by" should be object to match "id"
print("datatypes of canteen df are:")
print(canteen_df.dtypes)

canteen_df["id"] = canteen_df["id"].astype("object")
canteen_df["replaced_by"] = canteen_df["replaced_by"].astype("object")
canteen_df["created_at"] = pd.to_datetime(canteen_df["created_at"], format="%Y-%m-%d %H:%M:%S.%f")
canteen_df["updated_at"] = pd.to_datetime(canteen_df["updated_at"], format="%Y-%m-%d %H:%M:%S.%f")
canteen_df["last_fetched_at"] = pd.to_datetime(canteen_df["last_fetched_at"], format="%Y-%m-%d %H:%M:%S.%f")

# once again, data summary for a feel of distribution
# XX: "id" is unique and complete, so can be used as index
# XX: "name" is surprisingly not unique -> special attention on "obsolte" maybe to indicate inactivity?
# XX: "adress" not unique, but maybe because multiple canteens are on one campus
# XX: creation, update and last fetch date range back more than 10 years!
# XX: "state" == "active" is the biggest category and the count (1159) is the count of canteens I received through the API
# XX: "phone" and "email" are not consequently used -> can probably be excluded anyways
# XX: "availability" only has one category and that is "t" -> maybe TRUE for selection o records to be exported? -> can be neglected
# XX: "openingTimes" is completely empty -> modelled through days.csv -> can be neglected
# XX: "replaced_by" maybe fits with "state" == "inactive" -> should be further investigated
canteen_stats = canteen_df.describe(include="all")

#######################################################
# MISSING DATA
#######################################################

# let's take a look at missing data and see if any feature is unworkeable
# XX: address and coordinates are missing a couple of times -> if they still are active canteens, could be updated manually or just excluded
# XX: "last_fetched_at" is missing a couple of times as well -> could be correlated to status
# XX: "phone", "email" and "openingTimes" are unusuable, but probably not that important anyways
# XX: "replaced_by" is also quite empty, but could be that many canteens are still active -> or just didn't get updated
canteen_missing_stats = canteen_df.isna().sum().to_frame(name="count")
canteen_missing_stats["percentage"] = canteen_missing_stats["count"] / canteen_df.shape[0] * 100
print("Missing entries per feature")
print(canteen_missing_stats)

# graphical overview over NA distribution
# XXX: we can see that features are either pretty complete or pretty incomplete
#ax = msno.matrix(canteen_df, color=(0, 0.3, 0.5))
#ax.set_title("Overview over missing data of canteens.csv", fontsize=24)
plt.tight_layout()

#######################################################
# DUPLICATES
#######################################################

# check for duplicates
# XX: as expected, no duplicates (we've seen that already in the stats)
print("Print duplicates:")
print(canteen_df[canteen_df.duplicated(keep=False)])

# let's check for duplicates in name and address, because this is the minimum necessary data to identify a canteen -> sort by name for comparison
# XX: we have 68 duplicated entries, many are obsolete
# XX: the other ones usually vary in their state -> no problem if one is active and one is archived
# XX: we could have missed duplicates if there are slight variations in name (e.g., additional space)
duplicated = canteen_df[canteen_df.duplicated(subset=["name", "address"], keep=False)].sort_values(by="name")
print(f"Print duplicates in name and address (n={duplicated.shape[0]}):")
print(duplicated)

# let's check how many entries are duplicates and both active
# XX: only one canteen ("Ettelbrück, LTA Cafétéria (Lycée Technique Agricole)") is contained twice in dataset and both times active
duplicates_groups = duplicated.groupby(["name", "address", "state"]).size()

# let's try the even more extreme approach and look for duplicates only in "name" -> do we have generic, identical names in different cities?
# XX: the names are unique, so no generic "Hauptmensa" for city A, B and C
# XX: it's interesting to see that compared to name+address duplicates, we have many more name-only duplicates -> due to spelling mistakes abbreviations, different formats, etc. -> e.g., Garystr. vs. Garystraße (Cafeteria FU Wirtschaftswissenschaften, Berlin)
print("Canteen names appearing multiple times:")
duplicated_names = canteen_df[canteen_df.duplicated(subset="name", keep=False)].sort_values(by="name")
print(duplicated_names)

# same principle as above, but checking only address duplicates -> this way we might spot semantic duplicates
# XX: actually a duplicated address usually also means a duplicated canteen
# XX: in some cases a canteen and a cafeteria / café exist at the same location
# XX: we have a lot more entries in duplicated_addresses, which is partially due to canteen /cafeteria at the same address, but also because the names vary slightly sometimes (but easy to spot as duplicates for humans)
print("Addresses appearing multiple times:")
duplicated_addresses = canteen_df[canteen_df.duplicated(subset="address", keep=False)].sort_values(by="address")

# filter only for active canteens, as some duplicates are already marked archived
# XX: remaining duplicates seem to be updates parsers, as the fetch periods do not overlap
# XX: e.g., Mensa Erzbergerstr (Karlsruhe) or Mensa Finkenau (Hamburg)
# XX: I think for these canteens it would be good to update the replaced_by feature manually since they are not that many, but multiple years worth of data could be salvaged within the analysis time scope
# XX: some are also different canteens at the same location (e.g., a canteen and a cafeteria)
active_duplicated_addresses = duplicated_addresses[duplicated_addresses["state"] == "active"]
active_duplicated_addresses = active_duplicated_addresses[active_duplicated_addresses.duplicated(subset="address", keep=False)]

##############################################################
# DATA DISTRIBUTION: countries
##############################################################

# some canteens are located outside of Germany, let's try to find out the ratio
# for that, we will use a geopandas df and then spatial joining with a shape of Germany

# transfer to geopanda df, important: set crs correctly -> assuming epsg:4326 out of popularity and OSM compliance
canteen_gdf = gpd.GeoDataFrame(canteen_df, geometry=gpd.points_from_xy(canteen_df["longitude"], canteen_df["latitude"]), crs="EPSG:4326")

# get shapes of European countries to match with canteen locations
# we actually just need the name of the country and the geometry for now, so rest will be dropped
countries = gpd.read_file("data/helper_data/ne_50m_admin_0_sovereignty.shp")
countries = countries[countries["CONTINENT"] == "Europe"]
countries = countries[["NAME", "geometry"]]

# map each canteen to its country based on geo position
# attention: crs need to match to align data
print(f"CRS of the countries data: {countries.crs}")
print(f"CRS of the canteen data: {canteen_gdf.crs}")
print(f"Do the CRS match? ---- {countries.crs == canteen_gdf.crs}")
canteen_gdf = canteen_gdf.sjoin(df=countries, how="left")

# let's check the distribution over Europe
# XXX: 68% of the data is from German canteens, about 10% from Luxembourg, Austria, Switzerland, 1% from Italy
country_distribution = canteen_gdf["NAME"].value_counts(dropna=False).to_frame(name="count")
country_distribution["percent"] = country_distribution["count"] / canteen_gdf.shape[0] * 100
print(country_distribution)

# check nan entries
# XXX: a mix of test canteens, deleted canteens and valid canteens with missing coordinates
# XXX: invalid canteens should be removed, valid canteens should be updated with correct coordinates
nan_entries = canteen_gdf[canteen_gdf["NAME"].isna()]

# check 1-record entries
# XXX: data from Poland and France are German towns on the border -> maybe some inaccuracy in the maps
one_record_only = canteen_gdf[canteen_gdf["NAME"].isin(["France", "Poland", "Sweden"])]

# visualize distributions and format graph: (1) set color, labels and title, (2) add percentages to bars, (3) make figure fit window
ax = country_distribution["percent"].plot(kind="bar", color=layout_color, xlabel="Countries", ylabel="Percent [%]")
#ax.set_title("Distribution of tracked canteens in canteens.csv over European countries", fontsize=24)
ax.bar_label(ax.containers[0], fmt="%.2f", fontsize=18) # add percentages to bars
ax.margins(y=0.1) # format percentage location
plt.tight_layout()

# we want to plot canteens on a map, but before plotting it is really important to exclude data with missing geometry, otherwise map will not plot
# six entries did not correctly get transformed to point geometry -> check why -> missing coordinates -> we need to remove them
empty_geom= canteen_gdf.loc[canteen_gdf["latitude"].isna()]
canteen_subset = canteen_gdf.drop(canteen_gdf[canteen_gdf["latitude"].isna()].index, axis="index")

# there is two kinds of maps: (1) static maps saved as png files, (2) interactive maps saved as html files
# both map types can be enriched using a background map to plot on, a so called tile
# first, let's generate a static map -> IMPORTANT: we need to align crs of data and background map, otherwise data won't be plotted at the correct location on the map
# I selected CartoDB Voyager as background because it's subtle but still easy to understand locations
# XXX: two canteens are outside of Europe and take up a lot of attention -> check in interactive map which canteens they are
ax = canteen_subset.plot(kind="geo", color="gray")
cx.add_basemap(ax, crs=canteen_subset.crs.to_string(), source=cx.providers.CartoDB.Voyager)
ax.tick_params(axis="both", bottom=False, left=False, labelleft=False, labelbottom=False)
ax.set_title("Geolocation of canteens contained in canteens.csv", fontsize=24)

# for generating an interactive map we use a slightly different method, then save html file to display it in browser
# again we're using CartoDB Voyager as subtle but easy to interpret background
# ATTENTION: tooltip and popup cannot process datetime, so we'll just use the name, adress and geometry of canteens for now
# XXX: two outlier canteens are a test canteen and a deleted canteen
canteen_map = canteen_subset[["name", "address", "geometry"]].explore(tiles="CartoDB Voyager", color="gray")
canteen_map.save("maps/canteen_map_simple.html")

# let's create another static map that we can include in the thesis PDF, but without the outlier canteens
# XXX: after filtering outliers, we can clearly see the distribution with the agglomeration of canteens in Luxembourg, many canteens distributed all over Germany
ax = canteen_subset[~canteen_subset["NAME"].isna()].plot(kind="geo", color="gray")
cx.add_basemap(ax, crs=canteen_subset.crs.to_string(), source=cx.providers.CartoDB.Voyager)
ax.tick_params(axis="both", bottom=False, left=False, labelleft=False, labelbottom=False)
ax.set_title("Geolocation of canteens contained in canteens.csv (only Europe)", fontsize=24)

#######################################################
# DATA DISTRIBUTION: "created_at", "updated_at", "last_fetched_at"
#######################################################

# first of all, check the time relation between the three features
# created_at should always be the oldest date; definitely older than last_fetched_at
# XX: doesn't apply to all canteens
print(f"Canteens with creation before last fetch date: {(canteen_df['created_at'] < canteen_df['last_fetched_at']).sum()} of {canteen_df.shape[0]}")

# what about the canteens where creation date is not before last fetch date?
# XX: all of these canteens do no have a time given as last fetch date
# XX: they are all archived or new though, doesn't apply to active canteens
# XX: so it shouldn't really be a problem for our further analysis
temp = canteen_df[~(canteen_df["created_at"] < canteen_df["last_fetched_at"])]
temp["state"].value_counts(dropna=False)

# it would be interesting to see for how many days menus were collected for each canteen
# we will use a simplification for now and just get the distance between the creation date and the last fetch date
# fill na with 0s
canteen_df["days_active"] = (canteen_df["last_fetched_at"] - canteen_df["created_at"]).dt.days
canteen_df["days_active"] = canteen_df["days_active"].fillna(0)
print("Summary statistics of operating days per canteen:")
print(canteen_df["days_active"].describe())

# visually present the summary statistics using boxplots
# add second axis on the right where days are displayed in years for easier orientation
ax = canteen_df.boxplot(column="days_active", color="black", patch_artist=True, boxprops=dict(facecolor=layout_color), ylabel="Period tracked in days")
#ax.set_title(f"Distribution of amount of tracked days per canteen (n={canteen_df.shape[0]})", fontsize=24)
secax = ax.secondary_yaxis(location="right", functions=(lambda days: days/365, lambda years: years * 365), ylabel="Period tracked in years")
secax.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0)) # tick for every year

# I will also use a conditional boxplot, as I have a feeling the amount of active days varies according to canteen state
# plot and prettify for easier interpretation
plt.figure()
ax = canteen_df.boxplot(column="days_active", by="state", color="black", patch_artist=True, notch=True, boxprops=dict(facecolor=layout_color), ylabel="Period tracked in days", xlabel="State")
#ax.set_title(f"Distribution of amount of tracked days per canteen and state (n={canteen_df.shape[0]})", fontsize=24)
ax.set_title("")
plt.suptitle("")
plt.xlabel("State", labelpad=15) # x label is too close to tick descriptions otherwise

# include group counts of non-NA data points in description
# first prepare data to be used as labels for easy plotting, then adjust ticklabels
label_data = canteen_df[~canteen_df["days_active"].isna()]["state"].value_counts(dropna=False)
x_ticks = ax.get_xmajorticklabels()
x_ticks = [f"{category.get_text()} (n={label_data[category.get_text()]})" for category in x_ticks]
ax.set_xticklabels(x_ticks)
#ax.tick_params(axis='both', labelsize=14)

# add second axis on the right where days are displayed in years for easier orientation
secax = ax.secondary_yaxis(location="right", functions=(lambda days: days/365, lambda years: years * 365), ylabel="Period tracked in years")
secax.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0)) # tick for every year
#secax.set_ylabel("Period tracked in years", size=18)
#secax.tick_params(axis="y", labelsize=14)

# summary statistics of boxplot just created
state_stats = canteen_df.groupby("state")["days_active"].describe()
print(state_stats)

# convert periods in boxplot from days to years for better interpretability
new_cols = [col + "_years" for col in state_stats.columns]
state_stats[new_cols] = state_stats / 365
state_stats = state_stats[sorted(state_stats.columns)]

# it would also be interesting to see how many canteens have been fetched for every day of the last five years (or maybe even longer back)
# again, we will use a simplification and assume that the canteen has data for every day in between creation date and last fetch date

# exclude canteens which have never been fetched once
# for every canteen, we will extract a list of dates on which it was open (the period between creation and last fetch)
tracked_days = canteen_df.loc[~canteen_df["last_fetched_at"].isna(), ["id", "name", "address", "created_at", "last_fetched_at"]]
tracked_days["tracked_days"] = tracked_days.apply(lambda row: pd.date_range(start=row["created_at"], end=row["last_fetched_at"], freq="B").date, axis=1)

# now we can explode list of dates and then pivot into wide format
# we can sum up all canteens for each listed day, thus obtaining the number of tracked canteens per day
tracked_days_long = tracked_days.explode(column="tracked_days", ignore_index=True)
tracked_days_long["dummy"] = 1
tracked_days_matrix = tracked_days_long.pivot(index="id", columns="tracked_days", values="dummy")
canteens_per_day = tracked_days_matrix.sum(axis=0)


###########################################################
# DATA DISTRIBUTION: type of organization
###########################################################

# add organization feature based on exploration in interactive IDE
canteen_df["org"] = canteen_df["name"]
canteen_df["org"] = canteen_df["org"].mask(canteen_df["org"].str.contains(pat="(?<!hoch)schul|gymnasium|lyc[ée]e|[ée]cole|school", case=False, regex=True), other="school", inplace=False)
canteen_df["org"] = canteen_df["org"].mask(canteen_df["org"].str.contains(pat="kinder|kita|infanzia", case=False, regex=True), other="kindergarten", inplace=False)
canteen_df["org"] = canteen_df["org"].mask(canteen_df["org"].str.contains(pat="mensa|caf[eé]|bistro|uni|hochschule", case=False, regex=True), other="university", inplace=False)
canteen_df["org"] = canteen_df["org"].mask(canteen_df["org"].str.contains(pat="restaurant|gastronomie|betrieb", case=False, regex=True), other="company", inplace=False)

# put some dummy attribute for further analysis for unmatched entries
canteen_df["org"] = canteen_df["org"].where(canteen_df["org"].str.contains(pat="school|kindergarten|university|company", regex=True), other="other", inplace=False) 

# XXX: summarized statistics about group distribution
org_distribution = canteen_df["org"].value_counts(dropna=False).to_frame(name="count")
org_distribution["percent"] = org_distribution["count"] / canteen_df.shape[0] * 100
print(org_distribution)

# visualize distributions and format graph
plt.figure()
ax = org_distribution["percent"].plot(kind="bar", color=layout_color, ylabel="Percent [%]", xlabel="Organizational types")
#ax.set_title("Distribution of tracked canteens in canteens.csv over organizational types", fontsize=24)
ax.bar_label(ax.containers[0], fmt="%.2f", fontsize=18)
ax.margins(y=0.1)
plt.tight_layout()

# plot organizational types on map, static and dynamic
# first update gdf with new feature "org"
canteen_subset = pd.merge(left=canteen_subset, right=canteen_df[["id", "org"]], left_on="id", right_on="id")

# now static map
plt.figure()
ax = canteen_subset[~canteen_subset["NAME"].isna()].plot(kind="geo", column="org", cmap="Paired", legend=True)
cx.add_basemap(ax, crs=canteen_subset.crs.to_string(), source=cx.providers.CartoDB.Voyager)
ax.tick_params(axis="both", bottom=False, left=False, labelleft=False, labelbottom=False)
ax.set_title("Geolocation of canteens contained in canteens.csv by assigned organizational type (only Europe)", fontsize=24)

# now, dynamic: print with subtle background map for better visualization, then save
org_map = canteen_subset[["name", "address", "geometry", "org"]].explore(column="org", tiles="CartoDB Voyager", cmap="Paired")
org_map.save("maps/canteen_map_organizations.html")

#######################################################
# DATA DISTRIBUTION: "state"
#######################################################

# I'm especially interested in the categories of "name", "address", "city", "state"
# XX: "state" only has three categories: "active", "archived", "new"
# XX: the "active" count matches the number of canteens availabble through API download
# XXX: data is extremely slanted towards active canteens, new canteens completely irrelevant
canteen_state_categories = canteen_df["state"].value_counts(dropna=False).to_frame(name="count")
canteen_state_categories["percent"] = canteen_state_categories["count"] / canteen_df.shape[0] * 100
print(canteen_state_categories)

# visualize distributions and format graph
plt.figure()
ax = canteen_state_categories["percent"].plot(kind="bar", color=layout_color, ylabel="Percent [%]", xlabel='Categories of feature "state"')
#ax.set_title('Distribution of tracked canteens in canteens.csv for feature "state"', fontsize=24)
ax.bar_label(ax.containers[0], fmt="%.2f", fontsize=18)
ax.margins(y=0.1)
plt.xlabel('Categories of feature "state"', labelpad=15) # x label is too close to tick descriptions otherwise
plt.tight_layout()

# seperate archived and active canteens for better handling in further analysis
canteens_archived = canteen_df[canteen_df["state"] == "archived"]
canteens_active = canteen_df[canteen_df["state"] == "active"]

# let's take a look at some of the archived canteens and check online what their status is and if they are on openmensa.org
# XX: "state" is not necessarily unambiguous -> e.g., canteen "Rote Bete Ruhr-Universität-Bochum" is contained twice in the dataset, once active and once archived
# XX: it is similar for "Kiel, Mensa I am Westring" (but the name changed) and "Tübingen, Mensa Morgenstelle"
# XX: however, none of them have an entry for "replaced_by"
# XX: other canteens are still running (e.g., Darmstadt) according to website,  but are not displayed in OpenMensa.org -> correctly archived without successor
# XX: canteen "Lüneburg, Mensa Rotes Feld" e.g. doesn't exist anymore
print("Some sample canteens marked as archived")
canteens_archived[canteens_archived["name"].str.contains("rote|morgenstelle|westring", case=False)]

print("Checking whether these canteens really cannot be found in the active canteens anymore")
canteens_active[canteens_active["name"].str.contains("rote|morgenstelle|westring", case=False)]

# check how many archived canteens have replacement canteen filled in
count_replaced_by = canteens_archived[~canteens_archived["replaced_by"].isna()].shape[0]
print(f"Out of {canteens_archived.shape[0]} archived canteens, {count_replaced_by} have the feature 'replaced_by' filled in.")
print(f"That is {count_replaced_by / canteens_archived.shape[0] * 100}%.")
print(f"In the complete dataset, we have {canteen_df[~canteen_df['replaced_by'].isna()].shape[0]} 'replaced_by' entries.")

# let's check out the new canteens
# XX: I'm not sure what is new about them, other than that they apparently never have been fetched (scrolling through explorer)
# XX: these canteens do not appear on the openMensa.org website -> some have a previous crawler though
canteens_new = canteen_df[canteen_df["state"] == "new"]
print("Some canteens in Düsseldorf marked as new")
print(canteens_new[canteens_new["city"] == "Düsseldorf"])

# check if new canteens are also contained in active canteens
# XXX: they are, but only in form of semantic duplicates
print("Surprisingly, we find the same canteens (but different crawler) also in the active canteens")
print(canteens_active[canteens_active["city"] == "Düsseldorf"])

#######################################################
# DATA DISTRIBUTION: "city"
#######################################################

# just to get a feel for the distribution
# XX: as expected, university cities take the lead
# XX: we have a loooong list of cities with just one canteen, probably due to schools
# XX: some cities are not consistent, for example "Dortmund / NRW" and "Dortmund" or "44801 Bochum" and "Bochum"
s = canteen_df["city"].value_counts(dropna=False)
print(f"Cities contained in canteen data (n={s.size}):")
print(s)
#######################################################
# DATA DISTRIBUTION: "name"+ "address"
#######################################################

# since we already checked for duplicates, only thing missing is checking for canteen agglomerations
# I'm assuming canteen names and addresses to be unique, so I'm checking for repeated entries instead
# XX: the names are unique, so no generic "Hauptmensa" for city A, B and C
# XX: it's interesting to see that compared to name+address duplicates, we have many more name-only duplicates -> due to spelling mistakes abbreviations, different formats, etc. -> e.g., Garystr. vs. Garystraße (Cafeteria FU Wirtschaftswissenschaften, Berlin)
print("Canteen names appearing multiple times:")
duplicated_names = canteen_df[canteen_df.duplicated(subset="name", keep=False)].sort_values(by="name")
print(duplicated_names)

# same principle as above, but checking only address duplicates -> this way we might spot semantic duplicates
# XX: actually a duplicated address usually also means a duplicated canteen
# XX: in some cases a canteen and a cafeteria / café exist at the same location
# XX: we have a lot more entries in duplicated_addresses, which is partially due to canteen /cafeteria at the same address, but also because the names vary slightly sometimes (but easy to spot as duplicates for humans)
print("Addresses appearing multiple times:")
duplicated_addresses = canteen_df[canteen_df.duplicated(subset="address", keep=False)].sort_values(by="address")

# filter only for active canteens, as some duplicates are already marked archived
# XX: (I'm leaving apart all foreign canteens for now)
# XX: a couple of duplicates remain, but they seem to be updates parsers, as the fetch periods do not overlap
# XX: e.g., Mensa Erzbergerstr (Karlsruhe) or Mensa Finkenau (Hamburg)
# XX: I think for these canteens it would be good to update the replaced_by feature manually since they are not that many, but multiple years worth of data could be salvaged within the analysis time scope
active_duplicated_addresses = duplicated_addresses[duplicated_addresses["state"] == "active"]
active_duplicated_addresses = active_duplicated_addresses[active_duplicated_addresses.duplicated(subset="address", keep=False)]

########################################################
# SAVE
########################################################

# save version of CSV with new features to use in exploration of other CSVs
canteen_gdf = canteen_gdf.rename(columns={"NAME": "country"})
canteen_df = pd.merge(left=canteen_df, right=canteen_gdf[["id", "country"]], left_on="id", right_on="id", how="left")
canteen_df.to_csv("data/raw_data/canteens_edited.csv")
