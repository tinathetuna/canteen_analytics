# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:39:54 2023

@author: Tuni
"""

# this file contains a pipeline for cleaning up canteens.csv

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

#######################################################
# READ IN DATA
#######################################################

# read in CSV, no extra settings needed
canteen_df = pd.read_csv("data/raw_data/canteens.csv")

#######################################################
# THE BASICS: COLUMN NAMES, COLUMNS NEEDED, DATA TYPES, INDEX
#######################################################

# check if we read everything sucessfully
print("Check to see if everything was read correctly")
print(canteen_df.head())

# check columns to see if they need to be renamed
print(f"Column headers: {canteen_df.columns}")

# rename ID, name (for differentiation once merged with other CSVs)
# easiest to just append "canteen" to everything
canteen_df = canteen_df.add_prefix(prefix="canteen_")
print(f"New column headers: {canteen_df.columns}")

# delete empty, identical or unneeded columns
canteen_df = canteen_df.drop(labels=["canteen_phone", "canteen_email", "canteen_availibility", "canteen_openingTimes"], axis="columns")
print(f"Column headers after deleting unneeded features: {canteen_df.columns}")

# adjust data types, especially format dates to be able to process them later
print("Old datatypes of canteen_df are:")
print(canteen_df.dtypes)

canteen_df["canteen_id"] = canteen_df["canteen_id"].astype("object")
canteen_df["canteen_replaced_by"] = canteen_df["canteen_replaced_by"].astype("object")
canteen_df["canteen_created_at"] = pd.to_datetime(canteen_df["canteen_created_at"], format="%Y-%m-%d %H:%M:%S.%f")
canteen_df["canteen_updated_at"] = pd.to_datetime(canteen_df["canteen_updated_at"], format="%Y-%m-%d %H:%M:%S.%f")
canteen_df["canteen_last_fetched_at"] = pd.to_datetime(canteen_df["canteen_last_fetched_at"], format="%Y-%m-%d %H:%M:%S.%f")

print("New datatypes of canteen_df are:")
print(canteen_df.dtypes)

# check if ID is unique, if it is, assign as index
canteen_stats = canteen_df.describe(include="all")
canteen_df = canteen_df.set_index(keys="canteen_id")

#######################################################
# HANDLING MISSING DATA
#######################################################

# let's take a look at missing data and see what needs to be fixed
# XX: replaced_by -> can be disregarded for now, need to look for duplicates manually anyway
# XX: address and coordinates are urgently necessary to work with data -> check these canteens manually
# XX: last_fetched_at -> irrelevant for analysis, jsut ignore
canteen_missing_stats = canteen_df.isna().sum().to_frame(name="count")
canteen_missing_stats["percentage"] = canteen_missing_stats["count"] / canteen_df.shape[0] * 100
print("Missing entries per feature")
print(canteen_missing_stats)

# let's start with missing coordinates
# XX: we have test canteen and canteen in Turkey -> mark to be deleted because not necessary for our analysis
# XX: include the other coordinates manually
missing_geo = canteen_df[canteen_df["canteen_latitude"].isna() | canteen_df["canteen_longitude"].isna()]

# proceed as described above, take care that order of IDs and order of coordinates match
canteen_df.loc[[1769, 780, 779, 191], "canteen_latitude"] = [48.48373, 49.25419, 49.34325, 51.053]
canteen_df.loc[[1769, 780, 779, 191], "canteen_longitude"] = [9.18817, 7.03849, 7.03391, 13.74192]
canteen_df = canteen_df.drop(index=[192, 1648])

# now quick look at missing addresses
# XX: we just have Mensa HTW Göttelborn and Bistro Oeconomicum to fix, the rest is marked as "obsolete"
# XX: we probably need to delete the obsolete canteen because there is no way for us to re-engineer the spatial dimension
missing_address = canteen_df[canteen_df["canteen_address"].isna()]

# replace addresses
canteen_df.loc[[779, 1214], "canteen_address"] = ["Am Campus 4, 66287 Quierschied", "Universitätsstraße 14-16, 48143 Münster"]

# delete "obsolte" canteens
canteen_df = canteen_df[~(canteen_df["canteen_name"] == "obsolte")]

#######################################################
# ADDING AND FILTERING WITH "country" FEATURE
#######################################################

# we are only interested in German canteens for our analysis, so the other ones we can discard to avoid data overhead
# first of all, let's convert our pandas df into a geopandas df using the provided coordinates -> we can then access more and powerful spatial methods
# we already fixed missing coordinates, so nothing to take special care of, except CRS -> assuming epsg:4326 out of popularity and OSM compliance
canteen_gdf = gpd.GeoDataFrame(canteen_df, geometry=gpd.points_from_xy(canteen_df["canteen_longitude"], canteen_df["canteen_latitude"]), crs="EPSG:4326")

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

# clean up canteen_gdf for easier handling -> we don't need the feature index_right
# we should also rename the feature containing the country
canteen_gdf = canteen_gdf.drop(columns=["index_right"])
canteen_gdf = canteen_gdf.rename(columns={"NAME": "country"})

# let's check the distribution over Europe
# XXX: 68% of the data is from German canteens, about 10% from Luxembourg, Austria, Switzerland, 1% from Italy
# XXX: data from Poland and France are German towns on the border -> maybe some inaccuracy in the maps
# XXX: same with the NaN, it's a town on the Germany island Föhr
canteen_country_stats = canteen_gdf["country"].value_counts(dropna=False).to_frame(name="count")
canteen_country_stats["percentage"] = canteen_country_stats["count"] / canteen_gdf.shape[0] * 100
print(canteen_country_stats)

# closer look at NaN canteens
# XXX: just one real NaN canteen, the other is a test canteen
# XXX: valid NA canteen is a town on the Germany island Föhr -> located outside of Germany shape boundary due to inaccuracy -> can be updated manually
missing_country = canteen_gdf[canteen_gdf["country"].isna()]
canteen_gdf.loc[1360, "country"] = "Germany"
canteen_gdf = canteen_gdf.drop(index=217)

# countries with just one canteen
# XXX: one canteen is deleted, the other ones are actually in Germany, but assigned differently due to inaccuracy of shape
one_canteen_country = canteen_gdf[canteen_gdf["country"].isin(["Poland", "Sweden", "France"])]
canteen_gdf.loc[[880, 1509], "country"] = "Germany"
canteen_gdf = canteen_gdf.drop(index=1161)

# check other borders visually for correct assignment, maybe some Luxembourgian canteens are also actually German or vice versa
# take care to eliminate any entries with mising geometry and datetime columns, as they will prevent the map from plotting
# XXX: two canteens in Echternach (LUX) (IDs: 1034, 957) are accidentally mislabelled as German
# XXX: canteen in Konstanz (ID: 193) is accidentally labelled Swiss
# XXX: two Swiss canteens accidentally labelled Austrian (IDs: 634, 675)
my_map = canteen_gdf[["canteen_name", "canteen_address", "geometry", "country"]].explore(column="country", tiles="Carto DB Voyager", cmap="Paired", marker_kwds={"radius":5})
my_map.save("maps/canteen_map_cleanup.html")

# update inaccurate assignments
canteen_gdf.loc[[1034, 957], "country"] = "Luxembourg"
canteen_gdf.loc[193, "country"] = "Germany"
canteen_gdf.loc[[634, 675], "country"] = "Switzerland"

# now use country feature to filter only Germany canteens
canteen_gdf = canteen_gdf[canteen_gdf["country"] == "Germany"]

#######################################################
# ADDING AND FILTERING WITH "org" FEATURE
#######################################################

# use organization heuristic introduced during exploration
canteen_gdf["canteen_org"] = canteen_gdf["canteen_name"]
canteen_gdf["canteen_org"] = canteen_gdf["canteen_org"].mask(canteen_gdf["canteen_org"].str.contains(pat="(?<!hoch)schul|gymnasium", case=False, regex=True), other="school", inplace=False)
canteen_gdf["canteen_org"] = canteen_gdf["canteen_org"].mask(canteen_gdf["canteen_org"].str.contains(pat="kinder|kita", case=False, regex=True), other="kindergarden", inplace=False)
canteen_gdf["canteen_org"] = canteen_gdf["canteen_org"].mask(canteen_gdf["canteen_org"].str.contains(pat="mensa|caf[eé]|bistro|uni|hochschule", case=False, regex=True), other="university", inplace=False)
canteen_gdf["canteen_org"] = canteen_gdf["canteen_org"].mask(canteen_gdf["canteen_org"].str.contains(pat="restaurant|gastronomie|betrieb", case=False, regex=True), other="company", inplace=False)

# put some dummy attribute for further analysis for unmatched entries
canteen_gdf["canteen_org"] = canteen_gdf["canteen_org"].where(canteen_gdf["canteen_org"].str.contains(pat="school|kindergarden|university|company", regex=True), other="other", inplace=False) 

# check heuristic assignments of the categories and correct if necessary
# XXX: schools were fine, the only thing worth to mention are Berufsschulen, but they are still schools (although tertiary)
# XXX: kindergardens were fine too
schools = canteen_gdf[canteen_gdf["canteen_org"] == "school"]
kindergardens = canteen_gdf[canteen_gdf["canteen_org"] == "kindergarden"]

# XXX: for company canteens: there are several restaurants which are actually run by the Studierendenwerke, so I will group them with universities even though they have a more a la carte style and other price ranges
# XXX: for company canteens: canteen in Eschweiler actually provides food for schools
# XXX: for company canteens: scientific research institutes like Max-Planck-Institute will be considered companies
# XXX: for company canteens: Wilhelm Gastronomie is run by a private organization, but is located in university -> assign to "university"
companies = canteen_gdf[canteen_gdf["canteen_org"] == "company"]
canteen_gdf.loc[[337, 852, 347, 17, 773], "canteen_org"] = "university"
canteen_gdf.loc[[1283], "canteen_org"] = "school"

# "other" will be matched manually by looking up canteens or infering from name (TU, HTW, FU, etc.)
# some schools were listed with caterer instead of own name
# some canteens will remain in the category "other" because they are neither affiliated only with one school, business or kindergarden
other = canteen_gdf[canteen_gdf["canteen_org"] == "other"]
canteen_gdf.loc[[251, 252, 850, 288, 99, 349, 2, 81, 149, 54, 348, 830, 178, 193, 156, 839, 851, 20, 831, 7,
                 579, 578, 580, 583, 869, 573, 576, 571, 568, 572, 575, 574, 853, 808, 246, 923, 78, 116, 1718,
                 1740, 1755, 1787, 1791, 564, 1790], "canteen_org"] = "university"
canteen_gdf.loc[[199, 188, 1603, 1628, 1660, 1717], "canteen_org"] = "company"
canteen_gdf.loc[[1257, 1280, 1335, 1290, 1376, 1387, 1611], "canteen_org"] = "school"
canteen_gdf.loc[[1259], "canteen_org"] = "kindergarden"

# now take a closer look at canteens assigned to "university"
# XXX: it is kind of hard to look through all ~560 canteens manually -> because not all of them have the university included in their name
universities = canteen_gdf[canteen_gdf["canteen_org"] == "university"]

# that's why I will merge canteen list with my curated list of canteens -> everything that is on this list is confirmed associated with universities
# we will use the address as lookup -> not ideal, but an okay proxy
universities_curated = pd.read_excel("data/helper_data/Mensen_Kantinen_final.xlsx", sheet_name=0)
universities_left = pd.merge(left=universities, right=universities_curated, how="left", left_on="canteen_address", right_on="Adresse")

# XXX: 260 canteens were matched -> I was expecting more 
print((~universities_left["Kantine"].isna()).sum())

# one of the reasons why address lookup doesn't match is because address is given with an additional "Beispielstraße 1, 12345 Beispielstadt, Deutschland" or ", Germany" respectively -> remove to facilitate join
# we should also replace leading and trailing whitespaces that could get in the way
# we should also replace "Str." with "Straße" -> ATTENTION: str.replace uses regex even when turned off -> escape "." character
universities["address_formatted"] = universities["canteen_address"].str.replace(pat=", Deutschland", repl="", case=False)
universities["address_formatted"] = universities["address_formatted"].str.replace(pat=", Germany", repl="", case=False)
universities["address_formatted"] = universities["address_formatted"].str.strip()
universities["address_formatted"] = universities["address_formatted"].str.replace(pat="str\.", repl="straße", case=True, regex="False")
universities["address_formatted"] = universities["address_formatted"].str.replace(pat="Str\.", repl="Straße", case=True, regex="False")

# same for my curated list, it also contains both Str. and Straße -> unify
universities_curated["address_new"] = universities_curated["Adresse"].str.replace(pat="str\.", repl="straße", case=True, regex="False")
universities_curated["address_new"] = universities_curated["address_new"].str.replace(pat="Str\.", repl="Straße", case=True, regex="False")

# the remaining canteens we need to check manually, we are now performing left join to spot unmatched canteens easily
# XXX: ATTENTION: index is lost while merging canteens, reset_index before joining to retain canteen_id
universities_left = pd.merge(left=universities.reset_index(), right=universities_curated, how="left", left_on="address_formatted", right_on="Adresse")
print((~universities_left["Kantine"].isna()).sum())

# the remaining canteens need to be checked manually
manual_check = universities_left[universities_left["Studierendenwerk"].isna()]

# before starting the manual check, we can try to match canteens based on name
# XXX: manual check was reduced from ~270 to ~210 canteens
universities_left = pd.merge(left=universities_left, right=universities_curated, how="left", left_on="canteen_name", right_on="Kantine")
manual_check = universities_left[universities_left["Studierendenwerk_x"].isna() & universities_left["Studierendenwerk_y"].isna()]

# if it contains a university in name -> assume university
# if it is contained in my list -> assume university
# if address lookup in google maps is on a university campus -> assume university
# else: assume non-university and google details -> usually canteens are not in use anymore and thus not on my list -> or they are not canteens
canteen_gdf.loc[[104, 946, 1262], "canteen_org"] = "other"
canteen_gdf.loc[[91], "canteen_org"] = "school"
canteen_gdf.loc[[112], "canteen_org"] = "company"

# prepare to extract only university canteens
universities = canteen_gdf[canteen_gdf["canteen_org"] == "university"]

#######################################################
# SAVE CLEANED DATA
#######################################################

# before saving, drop columns that now contain all the same information
clean_data = universities.drop(columns=["country", "canteen_org"])
clean_data.to_pickle("data/processed_data/canteens_cleaned.pkl")
