# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 08:28:35 2023

@author: Tuni
"""

# this file is for setting up the dashboard of our canteen analysis

import pandas as pd
from dash import Dash, dcc, html, Output, Input, dash_table, ctx
import plotly.express as px

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# load data that we want to use for plotting -> canteen_df for map
canteen_df = pd.read_pickle("../data/processed_data/canteens_cleaned.pkl") # change to absolute path for code to run smootly
canteen_df = canteen_df.set_index(keys="canteen_id")

# load meal data that we want to display in table below graphs
# important: use .pkl file because it loads a lot quicker -> then reset index to obtain int index for displaying data in data table with ease
meals_df = pd.read_pickle("../data/processed_data/meals_cleaned_with_notes.pkl",
                       sep=",", parse_dates=[2, 3, 12, 13, 14, 17, 18, 19], nrows=100) # change to absolute path for code to run smootly

cols_to_include = ["meal_id", "meal_name", "meal_category", "meal_price_student", "date_correct", "canteen_name", "canteen_address", "canteen_city", "meal_super_category", "notes_list_90"]

# global attribute: current meal selection (needed for data table)
current_meal_selection = meals_df

# load data that we want to use for plotting -> indicators_df for timeline graphs
indicators_df = pd.read_csv("../data/indicators/indicators.csv") # change to absolute path for code to run smootly

# extract data needed for dropdown and radio buttons
indicators_set = indicators_df.columns[4:]

# for plotting purposes we should combine year and month again
# it will just be a string for now, also easier to control formatting -> use ISO format "YYYY-MM" so that plotly can infer date
indicators_df["date"] = indicators_df["year"].astype("str") + "-" + indicators_df["month"].astype("str")

indicators_df = pd.merge(left=indicators_df, right=canteen_df, how="left", left_on="canteen_id", right_index=True)

# create a reference to an external stylesheet for formatting the look of the dashboard
# especially declare where font family to use is located
external_stylesheets = [{"href": ("https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap"),
                         "rel": "stylesheet"}]

# initialize dashboard object, tell it to use external stylesheet that we have created apart
app = Dash(__name__, external_stylesheets=external_stylesheets)

# this is the text displayed in the browser tab and as header in (Google) search results
app.title = "Canteen Analytics: Nutrition Transition in German university Canteens"

# prepare map to be displayed in dashboard
fig = px.scatter_mapbox(data_frame=canteen_df, # df that contains data we want to plot on map
                        lat="canteen_latitude", # coordinates we want to plot
                        lon="canteen_longitude", # coordinates we want to plot
                        color_discrete_sequence=["#004E8A"], # color for the canteen markers
                        hover_name="canteen_name", # feature displayed upon hovering over marker with mouse (in bold)
                        hover_data={"canteen_address": True, "canteen_city": True, "canteen_name": True, "canteen_latitude": False, "canteen_longitude": False}, # additional features to be displayed -> careful to get formatting right
                        mapbox_style="open-street-map", # background map that we want to plot on -> I chose a standard map where it's easy to orient oneself
                        title="Analyzed German university canteens", 
                        height=600, # height of the map, width will be calculated automatically based on rules in CSS stylesheet
                        zoom=4,) # map is centered upon loading, but not auto-zoomed -> important to provide a zoom factor so that we can see all data at once

# configure information to be displayed upon hovering
fig.update_traces(hovertemplate="<b>%{customdata[2]}</b> <br><br>Address: %{customdata[0]} <br>City: %{customdata[1]}")



# now define the layout of the dashboard -> using hierarchical code similar to HTML
app.layout = html.Div(
    
    # all elements of the dashboard
    children=[
    
        # all text elements in the header
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H1(
                            children="Canteen Analytics",
                            className="header-title"),
                        html.P(
                            children="Analyze the nutrition transition from 2012-2023 in German university canteens using open source data",
                            className="header-description")],
                    className="header-plaque")],
            className="header"),

        # interactive map displaying the canteens
        html.Div(
            children=[
                dcc.Graph(
                    # global attributes of graph
                    id="canteen-map",
                    #config={"displayModeBar": False},
    
                    # attributes and settings dealing with actual canvas will be set dynamically
                    figure=fig)],
            className="map-container"),
        
        # multiple dropdowns to select the metrics and views wanted
        html.Div(
            children=[
                html.Div(
                    children=[
                        # this component is for filtering the indicator, it has an additional title above
                        html.Div(
                            children="Choose your indicator", 
                            className="menu-title"),
                        dcc.Dropdown(
                            id="indicator-filter",
                            options=[{"label": indicator, "value": indicator} for indicator in indicators_set], # list of dicts created with list comprehension
                            value="avg_count_meals",
                            clearable=False,
                            #multi=True,
                            placeholder="Choose your indicators",
                            className="dropdown")]),
                
                html.Div(
                    children=[
                    # this component is for filtering the data groups, it has an additional title above
                        html.Div(
                            children="Choose your focus", 
                            className="menu-title"),
                        dcc.Dropdown(
                            id="focus-filter",
                            options=[{"label": "Canteens", "value": "canteens"},
                                     {"label": "Meals", "value": "meals"},
                                     {"label": "Clusters", "value": "cluster_analysis"}],
                            value="canteens",
                            clearable=False,
                            #multi=True,
                            #placeholder="Canteens",
                            className="dropdown")]),
            
                html.Div(
                    children=[
                    # this component is for filtering the data groups, it has an additional title above
                        html.Div(
                            children="Choose your grouping feature", 
                            className="menu-title"),
                        dcc.Dropdown(
                            # this dropdown will be populated once the focus filter is set, so for now it only has a placeholder text and ID
                            id="grouping-filter",
                            options = [],
                            clearable=False,
                            #placeholder="Click to select",
                            className="dropdown")])],
            
            className="dropdown-container"),
        
        # graph elements for the selected indicator
        html.Div(
            children=[
                # this section displays graph in a box
                html.Div(
                    children=[
                        # now we move on to the actual graph
                        dcc.Graph(
                            # global attributes of graph
                            id="avg-count-main-dishes-chart",
                            config={"displayModeBar": False},
            
                            # attributes and settings dealing with actual canvas are set in update method
                            )],
                    className="card")],
                          
                
                className="graph-container"),
        
        # table that contains data values displayed for transparency reasons
        # the table will be dynamic so that dashboard won't be overloaded
        html.Div(
            children=[
                dash_table.DataTable(
                    # global attributes of table
                    id="meals-table",
                    columns=[{"name": col, "id": col} for col in cols_to_include],
                    
                    # set attributes necessary for paging
                    page_current=0,
                    page_size=10,
                    page_action="custom",
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", 
                                "paddingLeft": "10px",
                                "paddingRight": "10px"
                                #'width': '80%',
                                #"whiteSpace": "normal"
                                },
                    style_cell_conditional=[{"if": {"column_id": "meal_price_student"}, "textAlign": "right"},
                                            {"if": {"column_id": "meal_name"}, "width": "300px"},
                                            {"if": {"column_id": "canteen_address"}, "width": "300px"},
                                            {"if": {"column_id": "notes_list_90"}, "width": "300px"}],
                    style_header={"backgroundColor": "#636363",
                                  "color": "white",
                                  "fontWeight": "bold"},
                    style_data={"backgroundColor": "white",
                                "color": "black"},
                    style_as_list_view=True)],
            className="map-container"
            ),
        
        
        ])


@app.callback(
    Output("grouping-filter", "options"),
    Output("grouping-filter", "value"),
    Input("focus-filter", "value")
)
def set_grouping_options(focus):
    
    my_options = []
    my_value = ""
    
    # depending on the requested focus, we want to display attributes we can group by
    if focus == "canteens":
        my_options = [{"label": "Canteen ID", "value": "canteen_id"},
                      {"label": "Studierendenwerk", "value": "canteen_studierendenwerk"}]
        my_value = "canteen_id"
        
    elif focus == "meals":
        my_options = [{"label": "Diet type", "value": "diet_type"},
                      {"label": "Meal type", "value": "meal_type"}]
        my_value = "diet_type"
        
    elif focus == "cluster_analysis":
        my_options = []
        
    elif focus == "Click to select":
        my_options = []
        
    return my_options, my_value


@app.callback(
    Output("avg-count-main-dishes-chart", "figure"),
    Input("indicator-filter", "value"),
    Input("focus-filter", "value"),
    Input("grouping-filter", "value"))

# ATTENTION: while defining parameters, use same order as inputs defined above
def update_charts(indicator, focus, group_attribute):

    print(indicator)    

    # filter the data indicated by parameters
    filtered_data = indicators_df[(indicators_df["canteen_id"] == 24) | (indicators_df["canteen_id"] == 1)]
    
    # assign attribute to group data by (use canteen_id as default)
    # by using default values in dropdown assignment we have assured that group_attribute will never be empty
    group_by = "canteen_name" # group_attribute
    needed_for_table = group_attribute

    # create a line chart using plotly express
    fig = px.line(data_frame=filtered_data,
                  x="date",
                  y=indicator,
                  #line_group="canteen_id",
                  color=group_by,
                  custom_data=[needed_for_table, group_by],
                  labels={"avg_count_meals": "Average count of meals/day", "canteen_id": "Canteen ID", "date": "Date"},
                  color_discrete_sequence=px.colors.colorbrewer.Paired,
                  markers=True,
                  title=f"{indicator} per {group_by}",
                  template="plotly_white")
    
    return fig

# this method updates the table at the end of the dashboard
# it contains two different events: 
    # (1) clicking on the arrows to maneuver through data -> this will load the next page of table
    # (2) clicking on a month in the graph and loading the data used for building the aggregated indicator
# ATTENTION: table cannot be updated in two different methods, so we need to distinguish which event triggered the update
@app.callback(
    Output('meals-table', 'data'),
    Input('meals-table', "page_current"),
    Input('meals-table', "page_size"),
    Input("avg-count-main-dishes-chart", "clickData"),
    Input("grouping-filter", "value"))
def update_table(page_current, page_size, click_data, grouping_attribute):
    global current_meal_selection
    
    # first of all check which event triggered the update
    trigger = ctx.triggered_id
    
    # if event was triggered by table load the next page
    if trigger == "meals-table":
        # we need to select the next page, but only columns we want to display
        # transfer them to dict format that dash app needs
        return current_meal_selection.loc[page_current*page_size:(page_current+ 1)*page_size, cols_to_include].to_dict('records')

    # if event was triggered by graph, load corresponding data
    if trigger == "avg-count-main-dishes-chart":
        # we have set up graph in a way that upon clicking, it will return the group_by feature in the customdata feature
        # we also have the clicked-upon date given by the x position
        print(click_data)
        date = click_data["points"][0]["x"]
        current_group = click_data["points"][0]["customdata"][0]
        print(f"Date: {date} Type: {type(date)}")
        print(f"Current group of grouping feature: {current_group}")
        
        # now we can convert the date to a timestamp so that we can ue it for filtering our data more easily
        # filter all meals that belong to the clicked-upon month and year
        # then filter by grouping feature also
        date = pd.Timestamp(date)
        filtered_meals = meals_df[(meals_df["date_correct"].dt.month == date.month) & (meals_df["date_correct"].dt.year == date.year)]
        filtered_meals = filtered_meals[filtered_meals[grouping_attribute] == current_group]
        
        print(filtered_meals.shape)
        
        # set the new current meal selection for being able to browse the table correctly
        current_meal_selection = filtered_meals        
        
        # before returning data, remember to convert entries to dictionary for compliance with dash framework
        return filtered_meals.loc[0:page_size, cols_to_include].to_dict("records")

# run the app
if __name__ == "__main__":
    app.run_server(debug=True)
