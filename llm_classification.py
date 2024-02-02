# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:47:09 2023

@author: Tuni
"""

# this file is for experimenting with meal classification through an LLM
# it contains the preparation of the test set used for evaluation

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# for plotting purposes
plt.rcParams.update({"axes.labelsize": 22, # size of axis labels
                     "axes.titlesize": 24, # size of figure title
                     "xtick.labelsize": 20, # size of axes annotation of ticks
                     "ytick.labelsize": 20, # size of axis annotation of tickes
                     "figure.titlesize": 24, # plt.suptitle size
                     })
layout_color = "#004E8A"

####################################################################
# READ IN DATA
####################################################################

# read meal data that we want to classify
meals_df = pd.read_pickle("data/processed_data/meals_cleaned_with_notes.pkl")

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

############################################
# CLASSIFICATION 1: meal category
############################################

# we want to use the LLM to help us divide the meals into main dishes, side dishes, salads, soups, desserts and other
# this will be relevant for our further analysis, as for example main dishes and desserts need to be measured and interpreted differently (desserts are always vegetarian, e.g.)

# we have already removed meals identified as baked goods and other snacks so we can draw from complete meals_df
meals_population = meals_df

# draw 100 samples randomly to use as test set for LLM classification
# ATTENTION: set the seed, otherwise results won't be reproducible
random.seed(3)
sample_size = 100
random_indices = random.sample(range(meals_population.shape[0]+1), sample_size)

# as data changed after extracting samples, these are the correct sample IDs
sample = pd.read_csv("data/helper_data/meals_testset_v2.csv")
sample_IDs = sample["meal_id"]

# remove meals that have been removed from meals_df while data was changed
isin = sample_IDs[sample_IDs.isin(meals_df.index)]
random_meals = meals_df.loc[isin]

# select corresponding entries from df and export as csv -> now these examples can be labelled manually as reference
# only save meal id, meal name, meal category, notes list and notes list  (90%) for easier handling of data
random_meals = random_meals[["meal_name", "meal_category", "notes_list", "notes_list_90", "meal_super_category"]]

# remove any "|" from the data because otherwise our table format used in LLM will get destroyed
random_meals["meal_name"] = random_meals["meal_name"].str.replace(pat="|", repl=",", regex=False)
random_meals.to_csv("data/llm_testsets/dish_type_testset.csv")


############################################
# CLASSIFICATION 2: vegetarian vs. non-vegetarian
############################################

# we want to use the LLM to help us divide dishes into vegetarian and non-vegetarian options
# we will focus on main dishes because these are the most likely to vary in classification results (easier to evaluate of algorithm works if test set is not strongly unbalanced)

# exclude desserts, other, baked goods
meals_population = meals_df[meals_df["meal_super_category"] == "main_dish"]

# draw 100 samples randomly to use as test set for LLM classification
# ATTENTION: set the seed, otherwise results won't be reproducible
random.seed(9)
sample_size = 100
random_indices = random.sample(range(meals_population.shape[0]+1), sample_size)

# as data changed after extracting samples, these are the correct sample IDs
sample_veg = pd.read_csv("data/helper_data/meals_testset_vegetarian.csv")
sample_IDs_veg = sample_veg["meal_id"]

# select corresponding entries from df and export as csv -> now these examples can be labelled manually as reference
#random_meals = meals_population.iloc[random_indices]
random_meals = meals_df.loc[sample_IDs_veg]

# just for evaluation purposes remove unnecessary features, then save to file
# some meals can be configured to be either vegetarian or non-vegetarian -> mark as "other"
random_meals = random_meals[["meal_name", "meal_category", "notes_list", "notes_list_90", "is_vegetarian", "is_vegan", "is_omnivorous", "dietary_type"]]
random_meals.to_csv("data/llm_testsets/dietary_type_testset.csv")


############################################
# CLASSIFICATION 3: dishes with compulsory vegetable component added
############################################

# we want to use the LLM to identify dishes which have a compulsory vegetable component (without choosing side dishes, because in side dishes you could choose in a way to avoid vegetables)
# again we will focus on main dishes, so we can use the same sample as above

# save as different file though
# for labelling: following DGE recommendations (nr. 2 and nr. 8)
# vegetable componentn can be:
        # raw (salads, fingerfood)
        # gently-processed with little water and fat (steamed, stewed)
        # heavily-processed (soups, sauces)
        # None
        # Unknown: not clear from description
        # legumes?? -> maybe as extra feature?
        # could combine raw, gently and heavily into one feature and just have binary classification
        # do potatoes count as vegetables? -> no? -> according to DGE food groups, they don't
        # if multiple components, use the most gently-prepared one
        # tomato sauce?
random_meals.to_csv("data/llm_testsets/veg_comp_testset.csv")

############################################
# CLASSIFICATION 4: source of protein
############################################

# we want to use the LLM to identify (main) protein sources of dishes
# we will use all meals except desserts (and baked goods obviously) for test set, especially soups should be included (because of legumes)

# exclude desserts, other, baked goods
# TODO: might be needed to translate into German
meals_population = meals_df[meals_df["meal_super_category"].isin(["main_dish", "soup", "side_dish" "salad"])]

# draw 100 samples randomly to use as test set for LLM classification
# ATTENTION: set the seed, otherwise results won't be reproducible
random.seed(21)
sample_size = 100
random_indices = random.sample(range(meals_population.shape[0]+1), sample_size)

# select corresponding entries from df and export as csv -> now these examples can be labelled manually as reference
random_meals = meals_population.iloc[random_indices]

# just for evaluation purposes remove unnecessary features, then save to file
# based on EAT Lancet planetary health diet, we have protein sources "red_meat", "white_meat", "eggs", "fish", "legumes", "nuts"
random_meals = random_meals[["meal_name", "notes_list", "meal_category", "date_correct", "canteen_name", "canteen_address", "meal_super_category", "notes_count", "notes_list_90", "notes_count_90"]]
random_meals.to_csv("data/llm_testsets/protein_testset.csv")


############################################
# CLASSIFICATION 5: sweet dish
############################################

# we want to use the LLM to identify which main dishes are actually sweet dishes (like pancakes or Germknödel or similar)
# TODO: we will use only main dishes, but the ratio will probably be very skewed -> not sure of to prepare balanced test set?

# TODO: might be needed to translate into German
meals_population = meals_df[meals_df["meal_super_category"] == "main_dish"]

# draw 100 samples randomly to use as test set for LLM classification
# ATTENTION: set the seed, otherwise results won't be reproducible
random.seed(20)
random.seed(105)
random.seed(3022)
sample_size = 100
random_indices = random.sample(range(meals_population.shape[0]+1), sample_size)

# select corresponding entries from df and export as csv -> now these examples can be labelled manually as reference
random_meals = meals_population.iloc[random_indices]

# just for evaluation purposes remove unnecessary features, then save to file
random_meals = random_meals[["meal_name", "notes_list", "meal_category", "date_correct", "canteen_name", "canteen_address", "meal_super_category", "notes_count", "notes_list_90", "notes_count_90"]]
random_meals.to_csv("data/llm_testsets/sweet_dish_testset.csv")
