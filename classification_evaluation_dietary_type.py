# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 09:16:39 2023

@author: Tuni
"""

# this file is for evaluating llm classification, rule-based classification and manual classification for dietary type feature
# we will use confusion matrices, classification metrics

import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

# for plotting purposes
plt.rcParams.update({"axes.labelsize": 22, # size of axis labels
                     "axes.titlesize": 24, # size of figure title
                     "xtick.labelsize": 20, # size of axes annotation of ticks
                     "ytick.labelsize": 20, # size of axis annotation of tickes
                     "figure.titlesize": 24, # plt.suptitle size
                     })
layout_color = "#004E8A"

################################################
# LOAD DATA: manual labels + rule-based labels
################################################

# load manually-labelled data
manual_labels = pd.read_csv("data/classification_labels/dietary_type_testset_v2_classified.csv")

# we only need the meal ID features and the labels of the manual data
manual_labels = manual_labels[["meal_id", "meal_name", "true_dietary_type", "is_vegetarian", "is_vegan", "is_omnivorous", "dietary_type"]]

# set index to meal index to address unique meals quickly
manual_labels = manual_labels.set_index(keys="meal_id")

# rename rule-based label for more clarity
manual_labels = manual_labels.rename(columns={"dietary_type": "rule_based_eng"})

# translate rule-based labels from English to German to fit manual classification
# ATTENTION: don't mix up str.replace and "normal" replace
repl_dict = {"vegetarian": "vegetarisch", "omnivorous": "omnivor", "vegan": "vegetarisch"}
manual_labels["rule_based"] = manual_labels["rule_based_eng"].replace(repl_dict)

################################################
# LOAD DATA: LLM labels
################################################

# load LLM results -> this time an excel file instead of CSV, slightly more complicated to load because data is distributed over different sheets
# we need to specify that we want all sheets, but we want to keep the data separate (sheet_name=None)
llm_excel = pd.read_excel("data/classification_labels/dietary_type_testset_v2_llm.xlsx", sheet_name=None)

# we need to transfer our data from being in six separate DFs into one big DF while keeping the association with its trial
# we can do this easily using pandas concat which will retain dict keys for us as index
llm_labels = pd.concat(llm_excel)

# we should clean up the new index, the trial association is much more convenient as a normal feature for us and the numbered index per sheet we don't need anymore
llm_labels = llm_labels.reset_index()
llm_labels = llm_labels.rename(columns={"level_0": "trial", "level_1": "candidate_no"})

# some trials used the column "Name" and some "Gericht" to save the meals name -> fix by transfering data from one column to another
# afterwards we can drop "Name" because it contains redundant information
llm_labels["Gericht"] = llm_labels["Gericht"].fillna(llm_labels["Name"])
llm_labels["Gericht"] = llm_labels["Gericht"].fillna(llm_labels["Gerichtsname"])
llm_labels = llm_labels.drop(columns=["Name", "candidate_no", "Gerichtsname"])

# there is some duplicates in the data, probably due to some llm confusion -> we need to delete the duplicates so that we can pivot into wide format
temp = llm_labels[llm_labels.duplicated(subset=["trial", "ID", "Gericht"], keep=False)]
llm_labels = llm_labels[~llm_labels.duplicated(subset=["trial", "ID", "Gericht"], keep="first")]

# there is a format error due to "|" being contained in a meal name, the LLM classification didn't get saved
# set to "Unclassified" manually
llm_labels["Klassifizierung"] = llm_labels["Klassifizierung"].str.replace(pat="Joghurtdip", repl="Unclassified", regex=False)

# we have case upper and lower case in classification label
# use replacement to match labels assigned for manual review
llm_labels["Klassifizierung"] = llm_labels["Klassifizierung"].str.lower()
llm_labels["Klassifizierung"] = llm_labels["Klassifizierung"].str.replace(pat="nicht vegetarisch", repl="omnivor", regex=False)

# we can pivot from long format into wide format which will be easier to compare to manual labels side by side
# XXX: some meals were written slightly different (e.g., "Dazu Kartoffel-Sahnepürree" vs "dazu Kartoffel-Sahnepürree"), but the IDs seem fine so far
# XXX: sometimes meals are missing in the individual trials
# XXX: some meals are classified relatively consistently across trials, some alternate between two categories -> supports "2 out of 3" logic
llm_labels_id_meals = llm_labels.pivot(index=["ID", "Gericht"], columns="trial", values="Klassifizierung")

# for merging purposes, create the same pivot table again but with only ID
# XXX: size of dataset is 103 ... 3 entries hallucinated?
llm_labels_id = llm_labels.pivot(index="ID", columns="trial", values="Klassifizierung")

# for plotting and further analysis, we need to fill the missing values with some placeholder
llm_labels_id = llm_labels_id.fillna(value="unclassified")

################################################
# COMBINE DATA
################################################

# combine labels into one dataset for easier comparison and plotting, also same ordering etc
labels_df = pd.merge(left=manual_labels, right=llm_labels_id, how="left", left_index=True, right_index=True)

# we'll also extract a majority vote among the first three trials (low temperature) and last three trials (high temperature)
# in some cases there may be a draw, so we will only take the first value that appears as mode
labels_df["mode_low_temp"] = labels_df[["v1", "v2", "v3"]].mode(axis=1)[0]
labels_df["mode_default_temp"] = labels_df[["v4", "v5", "v6"]].mode(axis=1)[0]

# swap column order for easier plotting
cols = list(labels_df.columns)
cols = cols[:6] + cols[7:] + cols[6:7]
labels_df = labels_df[cols]

# translate data into English for inclusion in thesis
translation_dict = {"omnivor": "omnivorous", "vegetarisch": "vegetarian"}
labels_df = labels_df.replace(translation_dict)

################################################
# CALCULATE CONFUSION MATRIX
################################################

# confusion matrix panel for v1-v3 (low temperature)
# ATTENTION: positional arguments!!!
fig, axes = plt.subplots(1, 3)
for count, ax in enumerate(axes):
    count = count+6
    ConfusionMatrixDisplay.from_predictions(labels_df["true_dietary_type"], labels_df.iloc[:, count], cmap="Blues", ax=ax, xticks_rotation="vertical", colorbar=False, text_kw={"fontsize":16})
    ax.set_title(f"Trial {count-5}", fontsize=22)
    ax.set_xlabel("Predicted label", fontsize=22, labelpad=15)
    ax.set_ylabel("True label", fontsize=22, labelpad=10)
    ax.tick_params(axis="both", labelsize=18)
#fig.suptitle("Confusion matrices for LLM classification of dietary type (low temperature)", fontsize=30, y=0.85)   
fig.subplots_adjust(wspace=0.9)

# confusion matrix panel for v4-v6 (high temperature)
# ATTENTION: positional arguments!!!
fig, axes = plt.subplots(1, 3)
for count, ax in enumerate(axes):
    count = count+9
    ConfusionMatrixDisplay.from_predictions(labels_df["true_dietary_type"], labels_df.iloc[:, count], cmap="Blues", ax=ax, xticks_rotation="vertical", colorbar=False, text_kw={"fontsize":16})
    ax.set_title(f"Trial {count-5}", fontsize=22)
    ax.set_xlabel("Predicted label", fontsize=22, labelpad=15)
    ax.set_ylabel("True label", fontsize=22, labelpad=10)
    ax.tick_params(axis="both", labelsize=18)
#fig.suptitle("Confusion matrices for LLM classification of dietary type (default temperature)", fontsize=30, y=0.85)   
fig.subplots_adjust(wspace=0.9)

# confusion matrix panel for mode of low and high trials as well as rule-based classification
# ATTENTION: positional arguments!!!
fig, axes = plt.subplots(1, 3)
for count, ax in enumerate(axes):
    count = count+12
    ConfusionMatrixDisplay.from_predictions(labels_df["true_dietary_type"], labels_df.iloc[:, count], cmap="Blues", ax=ax, xticks_rotation="vertical", colorbar=False, text_kw={"fontsize":16})
    ax.set_title(labels_df.columns[count], fontsize=22)
    ax.set_xlabel("Predicted label", fontsize=22, labelpad=15)
    ax.set_ylabel("True label", fontsize=22, labelpad=10)
    ax.tick_params(axis="both", labelsize=18)
#fig.suptitle("Confusion matrices for dietary type classification: LLM vs. rule-based approach", fontsize=30, y=0.85)   
fig.subplots_adjust(wspace=0.9)

################################################
# CALCULATE CLASSIFICATION EVALUATION METRICS
################################################

# based on confusion matrices we can calculate a number of metrics to interpret differences between trials
results = []
y_true = labels_df["true_dietary_type"]

# loop over trials, calculate metrics for each trial and append to results list
# ATTENTION: we get some warning beause we have classes in true and assigned labels that don't exist in other dataset
for col in labels_df.columns[6:]:
    y_pred = labels_df[col]
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro', labels=["omnivor", "vegetarisch", "unclassified"])
    accuracy = accuracy_score(y_true, y_pred)
    bal_acc_weighted = balanced_accuracy_score(y_true, y_pred)
    results.append([col, precision, recall, fscore, accuracy, bal_acc_weighted])

results_df = pd.DataFrame(results, columns=["trial", "precision", "recall", "f_score", "accuracy", "balanced_accuracy_weighted"])