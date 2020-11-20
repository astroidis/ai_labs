from state import MapState
import graphs
import pandas as pd
import plotly.express as px


init = MapState(graphs.USA)

for k in init.map:
    move = init.get_next_move()
    # print(move)
    init.apply_move(move)

# print("\n\n", init.map)

df = pd.read_csv("usa_states_colors.csv")

for region, val in init.map.items():
    df.loc[df["Abbreviation"] == region, "Color"] = val[0]

# print(df.head())

colors = {"blue": "blue", "red": "red", "green": "green", "black": "black",
          "pink": "pink", "cyan": "cyan", "yellow": "yellow"}
fig = px.choropleth(df,  # input dataframe
                    locations="Abbreviation",  # column with locations
                    color="Color",  # column with color values
                    color_discrete_map=colors,  # custom colormap
                    locationmode="USA-states")  # plot as Us states

fig.update_layout(geo_scope="usa")  # plot only usa
fig.show()
