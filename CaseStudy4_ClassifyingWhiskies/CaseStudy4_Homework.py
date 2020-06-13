from sklearn.cluster import SpectralCoclustering
import numpy as np
import pandas as pd
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure, output_file, show
from itertools import product

whisky = pd.read_csv("whiskies.csv", index_col=0)
correlations = pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose())
correlations = np.array(correlations)

# ----- Exercise 1 ----- #
# First, we import a tool to allow text to pop up on a plot when the cursor
# hovers over it.  Also, we import a data structure used to store arguments
# of what to plot in Bokeh.  Finally, we will use numpy for this section as well!

# Let's plot a simple 5x5 grid of squares, alternating between two colors.
plot_values = [1, 2, 3, 4, 5]
plot_colors = ['#0173b2', '#de8f05']

# How do we tell Bokeh to plot each point in a grid?  Let's use a function that
# finds each combination of values from 1-5.

grid = list(product(plot_values, plot_values))
# print(grid)
# [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5)]
# The first value is the x coordinate, and the second value is the y coordinate.

# Let's store these in separate lists.
xs, ys = zip(*grid)
# print(xs)
# print(ys)
# (1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5)
# (1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5)

# Now we will make a list of colors, alternating between the two chosen colors.
colors = [plot_colors[i % 2] for i in range(len(grid))]
# print(colors)
# ['#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2', '#de8f05', '#0173b2']

# Finally, let's determine the strength of transparency (alpha) for each point,
# where 0 is completely transparent.
alphas = np.linspace(0, 1, len(grid))

# Bokeh likes each of these to be stored in a special dataframe, called
# ColumnDataSource.  Let's store our coordinates, colors, and alpha values.
source = ColumnDataSource(
    data = {
        "x": xs,
        "y": ys,
        "colors": colors,
        "alphas": alphas,
    }
)

# We are ready to make our interactive Bokeh plot!
output_file("Basic_Example.html", title="Basic Example")
fig = figure(tools="hover")
fig.rect("x", "y", 0.9, 0.9, source=source, color="colors",alpha="alphas")
hover = fig.select(dict(type=HoverTool))
hover.tooltips = {
    "Value": "@x, @y",
    }
# show(fig)

# ----- Exercise 2 ----- #
"""
Create a dictionary region_colors with regions as keys and cluster_colors as values.
Print region_colors
"""
cluster_colors = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161']
regions = ["Speyside", "Highlands", "Lowlands", "Islands", "Campbelltown", "Islay"]

region_colors = dict()
for i in range(len(regions)):
    region_colors[regions[i]] = cluster_colors[i]

# print(region_colors)

# ----- Exercise 3 ----- #
"""
-> Edit the code to define correlation_colors for each distillery pair to 
have input 'white' if the correlation for the pair is less than 0.7.
-> Whisky is a pandas dataframe, and Group is a column consisting of distillery 
group memberships. For distillery pairs with correlations greater than 0.7, 
if they share the same whisky group, use the corresponding color from 
cluster_colors. Otherwise, the correlation_colors value for that distillery pair 
will be defined as 'lightgray'
"""
distilleries = list(whisky.Distillery)
correlation_colors = []
for i in range(len(distilleries)):
    for j in range(len(distilleries)):
        if correlations[i][j] < 0.7:                      # if low correlation,
            correlation_colors.append('white')            # just use white.
        else:                                             # otherwise,
            if whisky.Group[i] == whisky.Group[j]:         # if the groups match
                correlation_colors.append(cluster_colors[whisky.Group[i]]) # color them by their mutual group.
            else:                                         # otherwise
                correlation_colors.append('lightgray')    # color them lightgray.

# aDict = correlation_colors.count('lightgray')
# print(aDict)  # 468

# ----- Exercise 4 ----- #
"""
-> correlation_colors is a list of string colors for each pair of distilleries. 
Set this as color in ColumnDataSource.
-> Define correlations in source using correlations from the previous exercise. 
To convert correlations from a np.array to a list, use the flatten() method. 
This correlation coefficient will be used to define both the color transparency 
as well as the hover text for each square.
"""
source = ColumnDataSource(
    data = {
        "x": np.repeat(distilleries,len(distilleries)),
        "y": list(distilleries)*len(distilleries),
        "colors": correlation_colors,
        "correlations": correlations.flatten()
    }
)

output_file("Whisky Correlations.html", title="Whisky Correlations")
fig = figure(title="Whisky Correlations",
    x_axis_location="above", x_range=list(reversed(distilleries)), y_range=distilleries,
    tools="hover,box_zoom,reset")
fig.grid.grid_line_color = None
fig.axis.axis_line_color = None
fig.axis.major_tick_line_color = None
fig.axis.major_label_text_font_size = "5pt"
fig.xaxis.major_label_orientation = np.pi / 3
fig.rect('x', 'y', .9, .9, source=source,
     color='colors', alpha='correlations')
hover = fig.select(dict(type=HoverTool))
hover.tooltips = {
    "Whiskies": "@x, @y",
    "Correlation": "@correlations",
}
# show(fig)  # bottom right corner is blue

# ----- Exercise 5 ----- #
"""
Run the following code, to be adapted in the next exercise.
Compare this code to the code used in plotting the distillery correlations.
"""
points = [(0,0), (1,2), (3,1)]
xs, ys = zip(*points)
colors = ['#0173b2', '#de8f05', '#029e73']

output_file("Spatial_Example.html", title="Regional Example")
location_source = ColumnDataSource(
    data={
        "x": xs,
        "y": ys,
        "colors": colors,
    }
)

fig = figure(title = "Title",
    x_axis_location = "above", tools="hover, save")
fig.plot_width  = 300
fig.plot_height = 380
fig.circle("x", "y", size=10, source=location_source,
     color='colors', line_color = None)

hover = fig.select(dict(type = HoverTool))
hover.tooltips = {
    "Location": "(@x, @y)"
}
# show(fig)

# ----- Exercise 6 ----- #
"""
-> Adapt the given code beginning with the first comment and ending with show(fig) to create 
the function location_plot(), as described above.
-> Region is a column of in the pandas dataframe whisky, containing the regional group 
membership for each distillery. Make a list consisting of the value of region_colors for 
each distillery, and store this list as region_cols.
-> Use location_plot to plot each distillery, colored by its regional grouping.
"""
def location_plot(title, colors):
    output_file(title+".html")
    location_source = ColumnDataSource(
        data={
            "x": whisky[" Latitude"],
            "y": whisky[" Longitude"],
            "colors": colors,
            "regions": whisky.Region,
            "distilleries": whisky.Distillery
        }
    )

    fig = figure(title=title,
        x_axis_location="above", tools="hover, save")
    fig.plot_width = 400
    fig.plot_height = 500
    fig.circle("x", "y", size=9, source=location_source, color='colors', line_color=None)
    fig.xaxis.major_label_orientation = np.pi / 3
    hover = fig.select(dict(type=HoverTool))
    hover.tooltips = {
        "Distillery": "@distilleries",
        "Location": "(@x, @y)"
    }
    show(fig)