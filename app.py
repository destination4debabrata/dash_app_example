import os

import dash
import dash_core_components as dcc
import dash_html_components as html
'''
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.H2('Hello World'),
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
        value='LA'
    ),
    html.Div(id='display-value')
])

@app.callback(dash.dependencies.Output('display-value', 'children'),
              [dash.dependencies.Input('dropdown', 'value')])
def display_value(value):
    return 'You have selected "{}"'.format(value)

if __name__ == '__main__':
    app.run_server(debug=True)
'''
###########################################################################################################################################################################################################################################
# https://dash-app-data608-module4.herokuapp.com/
###########################################################################################################################################################################################################################################

import pandas as pd
import numpy as np

'''
This module we'll be looking at the New York City tree census: https://data.cityofnewyork.us/Environment/2015-Street-Tree-Census-Tree-Data/uvpi-gqnh

This data is collected by volunteers across the city, and is meant to catalog informationabout every single tree in the city.

This data was provided by a volunteer driven census in 2015, and we'll be accessing it via the socrata API. The main site for the data is here, and on the upper right hand side you'll be able to see the link to the API.

The data is conveniently available in json format, so we should be able to just read it directly in to Pandas:
'''

url = 'https://data.cityofnewyork.us/resource/nwxe-4ae8.json'
trees = pd.read_json(url)
trees.head(10)

#Looks good, but lets take a look at the shape of this data:
trees.shape

'''
1000 seems like too few trees for a city like New York, and a suspiciously round number. What's going on?

Socrata places a 1000 row limit on their API. Raw data is meant to be "paged" through for applications, with the expectation that a UX wouldn't be able to handle a full dataset.

As a simple example, if we had a mobile app with limited space that only displayed trees 5 at a time, we could view the first 5 trees in the dataset with the url below:
'''

firstfive_url = 'https://data.cityofnewyork.us/resource/nwxe-4ae8.json?$limit=5&$offset=0'
firstfive_trees = pd.read_json(firstfive_url)
firstfive_trees

#If we wanted the next 5, we would use this url:
nextfive_url = 'https://data.cityofnewyork.us/resource/nwxe-4ae8.json?$limit=5&$offset=5'
nextfive_trees = pd.read_json(nextfive_url)
nextfive_trees

'''
You can read more about paging using the Socrata API here

In these docs, you'll also see more advanced functions (called SoQL) under the "filtering and query" section. These functions should be reminding you of SQL.

Think about the shape you want your data to be in before querying it. Using SoQL is a good way to avoid the limits of the API. For example, using the below query I can easily obtain the count of each species of tree in the Bronx:
'''

boro = 'Bronx'
soql_url = ('https://data.cityofnewyork.us/resource/nwxe-4ae8.json?' +\
        '$select=spc_common,count(tree_id)' +\
        '&$where=boroname=\'Bronx\'' +\
        '&$group=spc_common').replace(' ', '%20')
soql_trees = pd.read_json(soql_url)

soql_trees

'''
This behavior is very common with web APIs, and I think this is useful when thinking about building interactive data products. When in a Jupyter Notebook or RStudio, there's an expectation that (unless you're dealing with truly large datasets) the data you want can be brought in memory and manipulated.

Dash and Shiny abstract away the need to distinguish between client side and server side to make web development more accessible to data scientists. This can lead to some unintentional design mistakes if you don't think about how costly your callback functions are (for example: nothing will stop you in dash from running a costly model triggered whenever a dropdown is called.)

The goal of using the Socrata is to force you to think about where your data operations are happening, and not resort to pulling in the data and performing all operations in local memory.
'''

# Module 4
'''
Build a dash app for a arborist studying the health of various tree species (as defined by the variable 'spc_common') across each borough (defined by the variable 'borough'). This arborist would like to answer the following two questions for each species and in each borough:

What proportion of trees are in good, fair, or poor health according to the 'health' variable?
Are stewards (steward activity measured by the 'steward' variable) having an impact on the health of trees?
NOTE: One tip in dealing with URLs: you may need to replace spaces with '%20'. I personally just write out the url and then follow the string with a replace:

Please see the accompanying notebook for an introduction and some notes on the Socrata API. Deployment: Dash deployment is more complicated than deploying shiny apps, so deployment in this case is optional (and will result in extra credit). You can read instructions on deploying a dash app to heroku here: https://dash.plot.ly/deployment
'''

#https://api-url.com/?query with spaces'.replace(' ', '%20')

'''
Now, let's make a query to get the count of trees per each unique combination of stewardship, health, species, and borough present in the data.

There are 5 levels of steward including empty (None, 1or2, 3or4, and 4orMore, and empty string), 4 levels of health (Poor, Fair, Good, and empty string), 133 levels of spc_common including empty string, and 5 boroughs. So we could have up to 5 4 133 * 5 = 13,300 results from this query. However, I checked in a browser and we get less than 1,000 results starting on offset=4000, meaning that there are less than 5,000 results. So run for no offset, then offset 1000, 2000, 3000, and 4000.
'''

soql_url = ('https://data.cityofnewyork.us/resource/nwxe-4ae8.json?' +\
        '$select=spc_common,boroname,health,steward,count(tree_id)' +\
        '&$group=spc_common,boroname,health,steward').replace(' ', '%20')

pg1 = pd.read_json(soql_url)
pg2 = pd.read_json(soql_url + '&$offset=1000')
pg3 = pd.read_json(soql_url + '&$offset=2000')
pg4 = pd.read_json(soql_url + '&$offset=3000')
pg5 = pd.read_json(soql_url + '&$offset=4000')

counts = pd.concat([pg1,pg2,pg3,pg4,pg5])

#python -m pip install --upgrade pip
# pip install dash
# pip install dash
# pip install plotly
# pip install cufflinks
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.offline as py
import plotly.graph_objs as go
from plotly import tools

#Question #1
#What proportion of trees are in good, fair, or poor health according to the 'health' variable?

#gather data for question 1
trees_q1 = trees[['spc_common','health','boroname']]
#convert nans to 'Uknown'
trees_q1['spc_common'].fillna('Unknown',inplace = True)
#drop remaining nans
trees_q1.dropna(inplace = True)

#identify different health conditions
statuses = list(set(trees_q1['health']))
#create colors for different health conditions
colors = ['rgb(49,130,189)','rgb(204,204,204)','rgba(222,45,38,0.8)']

#create columns that specify tree health conditions
for status in set(trees_q1['health']):
    trees_q1[status] = np.where(trees_q1['health']==status,1,0)
    
trees_q1 = pd.DataFrame(trees_q1.groupby(['boroname','spc_common']).sum())

#find out boroughs
boroughs = list(set(trees['boroname']))

#calculate proportion of trees in different conditions
trees_q1['total'] = trees_q1.sum(axis=1)
for column in list(trees_q1.columns):
    trees_q1[column] = (trees_q1[column]/trees_q1['total'])*100
trees_q1.head()

#create list to store data for each borough
trace_list = []

#create plot titles
borough_list = list(map(lambda x: str(x), boroughs))
row = 1
col = len(boroughs)

fig_q1 = tools.make_subplots(rows=row, cols=col, subplot_titles=tuple(borough_list))

#iterate through boroughs
for borough in boroughs:
        for i in range(0,len(statuses)):
            trace = go.Bar(
            x = list(trees_q1.loc[borough].index),
            y = list(trees_q1.loc[borough][statuses[i]]),
            name = statuses[i],
            marker=dict(color=colors[i])
            )
            trace_list += [trace]

row_i = []
col_j = []
for i in range(1,row+1):
    for j in range (1,col+1):
        for n in range (1,4):
            row_i.append(i)
            col_j.append(j)

for i in range(0,len(trace_list)):        
     fig_q1.append_trace(trace_list[i], row_i[i],col_j[i]) 
 
        
fig_q1['layout'].update(showlegend=False,height=400, width=1400, title='Proportion of Trees in Good, Fair and Poor Conditions', barmode='stack')

app = dash.Dash()

colors = {
    'background': '#ffffff',
    'text': '#111111'
}

'''
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Question #1',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(children='Proportion of trees in Good, Fair and Poor conditions', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    
    html.Div([
        dcc.Graph(figure=fig_q1, id='my-figure-q1')
    ])
])
'''

#Question #2
#Are stewards (steward activity measured by the 'steward' variable) having an impact on the health of trees? In order to answer this question correlation between 'stewards' amnd 'health' should be correlated.

trees_q1 = trees[['spc_common','status','boroname']]
trees_q1['spc_common'].fillna('Unknown',inplace = True)

#create columns that specify tree status
for status in set(trees_q1['status']):
    trees_q1[status] = np.where(trees_q1['status']==status,1,0)
    
trees_q1 = pd.DataFrame(trees_q1.groupby(['boroname','spc_common']).sum())
trees_q1.head()

#find out boroughs
boroughs = list(set(trees['boroname']))

trace_list_q2 =[]

#create plot titles
borough_list = list(map(lambda x: str(x), boroughs))

trees_q2 = trees[['spc_common','health','boroname','steward']]

trees_q2['spc_common'].fillna('Unknown',inplace = True)
trees_q2.dropna(inplace = True)
trees_q2[['steward','health']] = trees_q2[['steward','health']].apply(lambda x : pd.factorize(x)[0])
trees_q2_cor = pd.DataFrame(trees_q2.groupby(['boroname','spc_common']).corr())
fig_q2 = tools.make_subplots(rows=1, cols=len(boroughs), subplot_titles=tuple(borough_list))

boroughs = list(set(trees_q2['boroname']))
plants = list(set(trees_q2['spc_common']))

for borough in boroughs:
    trace = go.Bar(
            x = list(trees_q1.loc[borough].index),
            y = list(trees_q2_cor.loc[borough]['steward'][::2])
            )
    trace_list_q2 += [trace]

for i in range(len(boroughs)):
    fig_q2.append_trace(trace_list_q2[i], 1, i+1) 
        
fig_q2['layout'].update(showlegend=False,height=500, width=1400, title='Proportion of Trees in Good, Fair and Poor Conditions')

app = dash.Dash()

colors = {
    'background': '#ffffff',
    'text': '#111111'
}

'''
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Question #2',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(children='Correlation between stewardsand health of trees', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    
    html.Div([
        dcc.Graph(figure=fig_q2, id='my-figure-q2')
	])
])

if __name__ == '__main__':
    app.run_server()
'''
    
# http://127.0.0.1:8050/

#Interactive Responses to Questions 1 & 2
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

def get_tree_list():
    soql_url = (
        "https://data.cityofnewyork.us/resource/nwxe-4ae8.json?"
        + "$select=spc_common,count(tree_id)"
        + "&$group=spc_common"
    ).replace(" ", "%20")
    temp = pd.read_json(soql_url).dropna()
    return temp.spc_common.tolist()


def get_human_intervention(x):
    if x == "None":
        return "Nature Only"
    else:
        return "Steward Intervention"


def get_steward_graph_data(boroname="Bronx", tree="American beech"):
    soql_url = (
        "https://data.cityofnewyork.us/resource/nwxe-4ae8.json?"
        + "$select=steward,health,count(tree_id)"
        + "&$where=spc_common='"
        + tree
        + "' AND boroname='"
        + boroname
        + "'"
        + "&$group=steward,health"
    ).replace(" ", "%20")
    df = pd.read_json(soql_url).dropna().rename(columns={"count_tree_id": "n"})
    df["type"] = df.steward.apply(get_human_intervention)

    df = df.groupby(["type", "health"])["n"].sum().reset_index()

    temp = df.groupby("type")["n"].sum().reset_index().rename(columns={"n": "total"})
    df = pd.merge(df, temp)
    df["share"] = df.n / df.total * 100

    shares = {
        "Steward Intervention": {"Poor": 0.0, "Fair": 0.0, "Good": 0.0},
        "Nature Only": {"Poor": 0.0, "Fair": 0.0, "Good": 0.0},
    }

    for index, row in df.iterrows():
        shares[row["type"]][row["health"]] = row["share"]

    poor = {
        "name": "Poor",
        "type": "bar",
        "x": ["Steward Intervention", "Nature Only"],
        "y": [
            round(shares["Steward Intervention"]["Poor"], 0),
            round(shares["Nature Only"]["Poor"], 0),
        ],
        "marker": {"color": "rgb(215, 48, 39)"},
    }
    fair = {
        "name": "Fair",
        "type": "bar",
        "x": ["Steward Intervention", "Nature Only"],
        "y": [
            round(shares["Steward Intervention"]["Fair"], 0),
            round(shares["Nature Only"]["Fair"], 0),
        ],
        "marker": {"color": "rgb(254, 224, 144)"},
    }
    good = {
        "name": "Good",
        "type": "bar",
        "x": ["Steward Intervention", "Nature Only"],
        "y": [
            round(shares["Steward Intervention"]["Good"], 0),
            round(shares["Nature Only"]["Good"], 0),
        ],
        "marker": {"color": "rgb(69, 117, 180)"},
    }
    return [poor, fair, good]


def get_tree_health_graph_data(tree="American beech"):
    soql_url = (
        "https://data.cityofnewyork.us/resource/nwxe-4ae8.json?"
        + "$select=boroname,health,count(tree_id)"
        + "&$where=spc_common='"
        + tree
        + "'"
        + "&$group=boroname,health"
    ).replace(" ", "%20")
    df = pd.read_json(soql_url).dropna().rename(columns={"count_tree_id": "n"})
    temp = (
        df.groupby("boroname")["n"].sum().reset_index().rename(columns={"n": "total"})
    )
    df = pd.merge(df, temp)
    df["share"] = df.n / df.total * 100

    shares = {
        "Bronx": {"Poor": 0.0, "Fair": 0.0, "Good": 0.0},
        "Brooklyn": {"Poor": 0.0, "Fair": 0.0, "Good": 0.0},
        "Manhattan": {"Poor": 0.0, "Fair": 0.0, "Good": 0.0},
        "Queens": {"Poor": 0.0, "Fair": 0.0, "Good": 0.0},
        "Staten Island": {"Poor": 0.0, "Fair": 0.0, "Good": 0.0},
    }

    for index, row in df.iterrows():
        shares[row["boroname"]][row["health"]] = row["share"]

    poor = {
        "name": "Poor",
        "type": "bar",
        "x": ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"],
        "y": [
            round(shares["Bronx"]["Poor"], 0),
            round(shares["Brooklyn"]["Poor"], 0),
            round(shares["Manhattan"]["Poor"], 0),
            round(shares["Queens"]["Poor"], 0),
            round(shares["Staten Island"]["Poor"], 0),
        ],
        "marker": {"color": "rgb(215, 48, 39)"},
    }
    fair = {
        "name": "Fair",
        "type": "bar",
        "x": ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"],
        "y": [
            round(shares["Bronx"]["Fair"], 0),
            round(shares["Brooklyn"]["Fair"], 0),
            round(shares["Manhattan"]["Fair"], 0),
            round(shares["Queens"]["Fair"], 0),
            round(shares["Staten Island"]["Fair"], 0),
        ],
        "marker": {"color": "rgb(254, 224, 144)"},
    }
    good = {
        "name": "Good",
        "type": "bar",
        "x": ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"],
        "y": [
            round(shares["Bronx"]["Good"], 0),
            round(shares["Brooklyn"]["Good"], 0),
            round(shares["Manhattan"]["Good"], 0),
            round(shares["Queens"]["Good"], 0),
            round(shares["Staten Island"]["Good"], 0),
        ],
        "marker": {"color": "rgb(69, 117, 180)"},
    }
    return [poor, fair, good]


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
'''
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
	html.H2("NYC Trees"),
    html.P(
            "How healthy are NYC's trees?  Are the activities of stewards helping the trees?  The following visualizations allow you to search a tree and view the results of the steward actions and answer these questions yourself.  Please keep in mind that correlation is not causation when looking at the steward intervention vs nature alone graph."
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="tree-dropdown",
                    options=[{"label": i.title(), "value": i} for i in get_tree_list()],
                    value="American beech",
                ),
                dcc.Graph(
                    config={"displayModeBar": False},
                    figure=go.Figure(
                        data=get_tree_health_graph_data(),
                        layout=go.Layout(
                            title="Q1. ARE AMERICAN BEECH TREES HEALTHY?",
                            yaxis=go.layout.YAxis(
                                title=go.layout.yaxis.Title(text="Percent"),
                                range=[0, 100],
                            ),
                        ),
                    ),
                    id="tree-health-graph",
                ),
            ],
            style={"width": "48%", "float": "left"},
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="borough-dropdown",
                    options=[
                        {"label": i, "value": i}
                        for i in [
                            "Bronx",
                            "Brooklyn",
                            "Manhattan",
                            "Queens",
                            "Staten Island",
                        ]
                    ],
                    value="Bronx",
                    searchable=False,
                ),
                dcc.Graph(
                    config={"displayModeBar": False},
                    figure=go.Figure(
                        data=get_steward_graph_data(),
                        layout=go.Layout(
                            title="Q2. ARE BRONX STEWARDS MAKING A DIFFERENCE?",
                            yaxis=go.layout.YAxis(
                                title=go.layout.yaxis.Title(text="Percent"),
                                range=[0, 100],
                            ),
                        ),
                    ),
                    id="steward-graph",
                ),
            ],
            style={"width": "48%", "float": "right"},
        )
])

if __name__ == "__main__":
    app.run_server()
'''
# http://127.0.0.1:8050/

counts = pd.concat([pg1,pg2,pg3,pg4,pg5])

counts = counts.dropna() #Remove rows missing information.

counts = counts.replace('None','0_stewards') #Replace steward="None" with "0_stewards" so that this level will be first in the plot. Realize this is a bit messy, but was running into issues when I made this just '0'.

#From Dash documentation.

def generate_table(dataframe,max_rows=counts.shape[0] + 1):
	return html.Table(
		#Header
		[html.Tr([html.Th(col) for col in dataframe.columns])] +
		
		# Body
		[html.Tr([
			html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
		]) for i in range(min(len(dataframe), max_rows))]
	)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

'''
if __name__ == '__main__':
	app.run_server()
'''   
# http://127.0.0.1:8050/

offset = 1000
max_row = 5000

'''
Retrieve data set of NYC trees through API call using Socrata query.
'''

for x in range(0, max_row, offset):
    #print('x is ' + str(x))
    soql_url = ('https://data.cityofnewyork.us/resource/nwxe-4ae8.json?$limit=1000&$offset=' + str(x) +\
        '&$select=borocode,spc_common,health,steward,count(tree_id)' +\
        '&$group=borocode,spc_common,health,steward').replace(' ', '%20')
    soql_trees = pd.read_json(soql_url)
    if(x==0):
        df = pd.DataFrame(columns=list(soql_trees.columns.values))
    df = df.append(soql_trees)
    #print(df)

df = df.reset_index(drop=True)

'''
Size of retrieved data set:
'''
print('Size of retrieved data set:' + str(len(df)))

'''
There are 132 unique tree species name
'''
print ('No. of unique species: ' + str(len(list(df.spc_common.unique()))))

'''	
Remove rows that do not have complete data. 11 rows removed that do not have complete data.
'''
df = df.dropna(axis=0, how='any')

'''
Preview data set retrieved: 
'''
print(df.head(5))

'''
Prepare data to be able to provide functionality that arborist needs
Build a dash app for a arborist studying the health of various tree species (as defined by the variable 'spc_common') across each borough (defined by the variable 'borough'). This arborist would like to answer the following two questions for each species and in each borough:
What proportion of trees are in good, fair, or poor health according to the 'health' variable?
Are stewards (steward activity measured by the 'steward' variable) having an impact on the health of trees?
'''

'''
Plan to meet requirements for question 1
For every specice and in each borough, what proportion of trees are in good, fair, or poor health?
The application will allow arborist to select one specie, and the application will display proportion of trees that are in good, fair, or poor health 
across all boroughs. Arborist will be able to compare health of particular specie across all five boroughs.
Bar graphs will be used to present the proportions. The orientation of the bar graphs will be vertical. 
The bar graphs will be first grouped by boroughs for each health status.
The goal of the code below is to create a dataframe that has the columns: borocode, spc_common, health, ratio.
Ratio is the proportion of spc_common in the given borough that has the given heath level. 
For example, a ratio for red maple in Queens with a health of good is the proportion of good red maple trees in Queens.
'''

#Reshape data for question 1:
df_totals = df.groupby(['borocode', 'spc_common'])['count_tree_id'].sum()
df_total_by_borocode_specie_health = df.groupby(['borocode', 'spc_common', 'health'])['count_tree_id'].sum()
df_totals = df_totals.reset_index(drop=False)
df_total_by_borocode_specie_health = df_total_by_borocode_specie_health.reset_index(drop=False)
df_totals.columns = ['borocode', 'spc_common', 'total_for_specie_in_borough']
df_total_by_borocode_specie_health.columns = ['borocode', 'spc_common', 'health', 'total']
tree_proportions = pd.merge(df_total_by_borocode_specie_health, df_totals, on=['borocode', 'spc_common'])
tree_proportions['ratio'] = tree_proportions['total']/ tree_proportions['total_for_specie_in_borough']
tree_proportions['spc_common'] = tree_proportions['spc_common'].apply(lambda x: x.title())
#print(tree_proportions.head(10))

#For species dropdown:
species = np.sort(tree_proportions.spc_common.unique())

'''
Preparing data to meet requirements for question 2
I would like to use a scatter plot to represent the overall health status of the selected specie across all the boroughs. 
An overall health index is determined by assigning a numeric value to each health level (Poor=1, Fair=2, Good=3) and 
then calculating a weighted average for the selected specie for each borough. 
The overall health index score has a minimum score of 1 and a maximum score of 3.
'''
#Reshape data for question 2:
df_total_by_steward = df.groupby(['borocode', 'spc_common', 'steward'])['count_tree_id'].sum()
df_total_by_steward = df_total_by_steward.reset_index(drop=False)
df_total_by_steward.columns = ['borocode', 'spc_common', 'steward', 'steward_total']
df['borocode'] = pd.to_numeric(df['borocode'])
df_steward = pd.merge(df, df_total_by_steward, on=['borocode', 'spc_common', 'steward'])
di = {'Poor':1, 'Fair':2, 'Good':3}
df_steward['health_level'] = df_steward['health'].map(di)
#df_steward.sort_values(by=['borocode', 'spc_common', 'steward']).head(10)
df_steward['health_index'] = (df_steward['count_tree_id']/df_steward['steward_total']) * df_steward['health_level']
#df_steward.sort_values(by=['borocode', 'spc_common', 'steward']).head(10)
df_overall_health_index = df_steward.groupby(['borocode', 'spc_common', 'steward'])['health_index'].sum()
df_overall_health_index = df_overall_health_index.reset_index(drop=False)
df_overall_health_index.columns = ['borocode', 'spc_common', 'steward', 'overall_health_index']
di2 = {'3or4':3, '4orMore':4, 'None':1, '1or2':2}
df_overall_health_index['steward_level'] = df_overall_health_index['steward'].map(di2)
di3 = { 1:'Manhattan', 2:'Bronx', 3:'Brooklyn', 4:'Queens', 5:'Staten Island'}
df_overall_health_index['borough'] = df_overall_health_index['borocode'].map(di3)
df_overall_health_index['spc_common'] = df_overall_health_index['spc_common'].apply(lambda x: x.title())
#print(df_overall_health_index.head(10))


'''
Code below if for DASH application
'''
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    ######################################################################################################################################################################
    html.H1(
        children='Question #1',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(children='Proportion of trees in Good, Fair and Poor conditions', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    
    html.Div([
        dcc.Graph(figure=fig_q1, id='my-figure-q1')
    ]),
	######################################################################################################################################################################
    html.H1(
        children='Question #2',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.Div(children='Correlation between stewardsand health of trees', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    
    html.Div([
        dcc.Graph(figure=fig_q2, id='my-figure-q2')
	]),
    ######################################################################################################################################################################
	html.H2("NYC Trees"),
    html.P(
            "How healthy are NYC's trees?  Are the activities of stewards helping the trees?  The following visualizations allow you to search a tree and view the results of the steward actions and answer these questions yourself.  Please keep in mind that correlation is not causation when looking at the steward intervention vs nature alone graph."
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="tree-dropdown",
                    options=[{"label": i.title(), "value": i} for i in get_tree_list()],
                    value="American beech",
                ),
                dcc.Graph(
                    config={"displayModeBar": False},
                    figure=go.Figure(
                        data=get_tree_health_graph_data(),
                        layout=go.Layout(
                            title="Q1. ARE AMERICAN BEECH TREES HEALTHY?",
                            yaxis=go.layout.YAxis(
                                title=go.layout.yaxis.Title(text="Percent"),
                                range=[0, 100],
                            ),
                        ),
                    ),
                    id="tree-health-graph",
                ),
            ],
            style={"width": "48%", "float": "left"},
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="borough-dropdown",
                    options=[
                        {"label": i, "value": i}
                        for i in [
                            "Bronx",
                            "Brooklyn",
                            "Manhattan",
                            "Queens",
                            "Staten Island",
                        ]
                    ],
                    value="Bronx",
                    searchable=False,
                ),
                dcc.Graph(
                    config={"displayModeBar": False},
                    figure=go.Figure(
                        data=get_steward_graph_data(),
                        layout=go.Layout(
                            title="Q2. ARE BRONX STEWARDS MAKING A DIFFERENCE?",
                            yaxis=go.layout.YAxis(
                                title=go.layout.yaxis.Title(text="Percent"),
                                range=[0, 100],
                            ),
                        ),
                    ),
                    id="steward-graph",
                ),
            ],
            style={"width": "48%", "float": "right"},
        ),
	######################################################################################################################################################################
	html.H2(children='Tree Health in NYC - a Dash App'),
	html.H3(children='Select species and borough.'),
	dcc.Dropdown(id='species', options=[
		{'label': i, 'value': i} for i in counts.spc_common.unique()
	],value='London planetree',multi=False, placeholder='Filter by species.'),
	dcc.Dropdown(id='borough', options=[
		{'label': j, 'value': j} for j in counts.boroname.unique()
	],value='Queens',multi=False, placeholder='Filter by borough.'),
	html.H4(children='Raw numbers of trees'),
	html.Div(id='table-container'),
	html.H4(children='Distribution of health'),
	dcc.Graph(id='health-dist'),
	html.H4(children='Health as a function of stewardship'),
	dcc.Graph(id='health-vs-steward'),
	######################################################################################################################################################################
	html.H4('Select Tree Specie'),
    dcc.Dropdown(
        id='specie', 
        options=[{'label': i, 'value': i} for i in species],
        value="'Schubert' Chokecherry",
        style={'height': 'auto', 'width': '300px'}
    ),
    dcc.Graph(id='graph-ratio'),
    dcc.Graph(id='graph-health')
])

@app.callback(
    Output(component_id="tree-health-graph", component_property="figure"),
    [Input(component_id="tree-dropdown", component_property="value")],
)
def update_tree_health_graph(spc_common):
    fig = go.Figure(
        data=get_tree_health_graph_data(spc_common),
        layout=go.Layout(
            title="Q1. ARE " + spc_common.upper() + " TREES HEALTHY?",
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(text="Percent"), range=[0, 100]
            ),
        ),
    )
    return fig

@app.callback(
    Output(component_id="steward-graph", component_property="figure"),
    [
        Input(component_id="borough-dropdown", component_property="value"),
        Input(component_id="tree-dropdown", component_property="value"),
    ],
)
def update_tree_health_graph(borough, spc_common):
    fig = go.Figure(
        data=get_steward_graph_data(borough, spc_common),
        layout=go.Layout(
            title="Q2. ARE " + borough.upper() + " STEWARDS MAKING A DIFFERENCE?",
            yaxis=go.layout.YAxis(
                title=go.layout.yaxis.Title(text="Percent"), range=[0, 100]
            ),
        ),
    )

    return fig

@app.callback(
    dash.dependencies.Output('table-container', 'children'),
    [dash.dependencies.Input('species', 'value'),
    dash.dependencies.Input('borough', 'value')])

def display_table(species_value,borough_value):
    return generate_table(counts[(counts['spc_common'] == species_value) & (counts['boroname'] == borough_value)])

@app.callback(
	dash.dependencies.Output('health-dist', 'figure'),
	[dash.dependencies.Input('species', 'value'),
	dash.dependencies.Input('borough', 'value')])

def update_graph(species_value,borough_value):
	counts_of_interest = counts[(counts['spc_common'] == species_value) & (counts['boroname'] == borough_value)]
	counts_per_health = counts_of_interest.groupby('health').sum()['count_tree_id']
	return {
		'data': [go.Bar(
			x = counts_per_health.index,
			y = counts_per_health
		)],
		'layout': go.Layout(
			xaxis = dict(title = 'Health'),
			yaxis = dict(title = 'Number of trees')
		)
	}

@app.callback(
	dash.dependencies.Output('health-vs-steward','figure'),
	[dash.dependencies.Input('species', 'value'),
	dash.dependencies.Input('borough', 'value')])

def update_graph(species_value,borough_value):
	counts_of_interest = counts[(counts['spc_common'] == species_value) & (counts['boroname'] == borough_value)]
	
	counts_per_steward = counts_of_interest.groupby('steward').sum()['count_tree_id']

	health_as_percent_steward = pd.merge(counts_of_interest,
                                     pd.DataFrame(counts_per_steward),
                                     on='steward')

	health_as_percent_steward['percent_steward'] = health_as_percent_steward['count_tree_id_x']*100/health_as_percent_steward['count_tree_id_y']

	health_as_percent_steward = health_as_percent_steward.pivot(index='steward',
                                                            columns='health',
                                                            values='percent_steward')

	#If any health levels missing, add a column of zeroes.

	for health in ['Poor','Fair','Good']:
		if health not in health_as_percent_steward.columns.tolist():
			health_as_percent_steward[health] = [0] * len(health_as_percent_steward.index)

	trace3 = go.Bar(
    	x=health_as_percent_steward['Poor'].index,
		y=health_as_percent_steward['Poor'],
		name='Poor')

	trace2 = go.Bar(
		x=health_as_percent_steward['Fair'].index,
		y=health_as_percent_steward['Fair'],
		name='Fair')

	trace1 = go.Bar(
		x=health_as_percent_steward['Good'].index,
		y=health_as_percent_steward['Good'],
		name='Good')

	return {
		'data': [trace1,trace2,trace3],
		'layout': go.Layout(
			xaxis = dict(title = 'Number of stewards'),
			yaxis = dict(title = 'Percent of trees'),
			barmode = 'stack'
		)
	}


#Display Proportion Graph 
@app.callback(
    Output('graph-ratio', 'figure'),
    [Input('specie', 'value')])
def update_figure(selected_specie):

    filtered_df = tree_proportions[tree_proportions.spc_common == selected_specie]
    #borocode: 1 (Manhattan), 2 (Bronx), 3 (Brooklyn), 4 (Queens), 5 (Staten Island)
    manhattan = filtered_df[filtered_df.borocode == 1]
    bronx = filtered_df[filtered_df.borocode == 2]
    brooklyn = filtered_df[filtered_df.borocode == 3]
    queens = filtered_df[filtered_df.borocode == 4]
    staten_island = filtered_df[filtered_df.borocode == 5]
    
    traces = []

    traces.append(go.Bar(
    x=queens['health'],
    y=queens['ratio'],
    name='Queens',
    opacity=0.9
    ))

    traces.append(go.Bar(
    x=manhattan['health'],
    y=manhattan['ratio'],
    name='Manhattan',
    opacity=0.9
    ))

    traces.append(go.Bar(
    x=bronx['health'],
    y=bronx['ratio'],
    name='Bronx',
    opacity=0.9
    ))

    traces.append(go.Bar(
    x=brooklyn['health'],
    y=brooklyn['ratio'],
    name='Brooklyn',
    opacity=0.9
    ))

    traces.append(go.Bar(
    x=staten_island['health'],
    y=staten_island['ratio'],
    name='Staten Island',
    opacity=0.9
    ))
    
    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'Health of Trees'},
            yaxis={'title': 'Proportion of Trees in Borough'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend=dict(x=-.1, y=1.2)
        )
    }


#Display Steward-Health Graph for Question 2
@app.callback(
    Output('graph-health', 'figure'),
    [Input('specie', 'value')])
def update_figure2(selected_specie):
    #print('here: ' + selected_specie)
    filtered_df = df_overall_health_index[df_overall_health_index.spc_common == selected_specie]
    traces2 = []
        
    for i in filtered_df.borough.unique():
        df_by_borough = filtered_df[filtered_df['borough'] == i]
        traces2.append(go.Scatter(
            x=df_by_borough['steward_level'],
            y=df_by_borough['overall_health_index'],
            mode='markers',
            opacity=0.7,
            marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=i
        ))
    
    return {
        'data': traces2,
        'layout': go.Layout(
            #xaxis={'title': 'Steward Level'},
            yaxis={'title': 'Overall Health Index'},
            xaxis=dict(tickvals = [1,2,3,4], ticktext = ['None', '1or2', '3or4', '4orMore'], title='Steward'),
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend=dict(x=-.1, y=1.2)
        )
    }


if __name__ == '__main__':
    app.run_server()
    
# http://127.0.0.1:8050/