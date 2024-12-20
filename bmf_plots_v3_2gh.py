import streamlit as st
import pandas as pd
# import geopandas as gpd
import altair as alt
from vega_datasets import data      

from streamlit_folium import st_folium
import folium

import math
import matplotlib.pyplot as plt
import numpy as np
import pprint
import csv
import math


APP_TITLE = 'NY Nonprofits'
APP_SUB_TITLE = 'Main Data Source:  IRS Business Master File (BMF)'
APP_VERS = "3_2"



# --- defs for all reports ---

def test_report(ny_cities_df, srch_city):
    st.write("sdf")


def cities_folium(ny_cities_df, srch_city):
    """ nonprofits on openstreet map
        not implemented yet.   
        triggers reruns where I don't want
    
    """

    center = [42.54043355305221,-76.1342239379883]
    num = 0
    m2 = folium.Map(
        location=center,
        #zoom_start=15
    )
    
    nps2 = folium.FeatureGroup(name="Cities")

    filt = ny_cities_df["dc_longitude"].notnull()        

    for index, row in ny_cities_df[filt].sort_values('major_city').iterrows():
        
        num = index
        present_num = "(" + str(num) + ") "
        #if num == np_df_selected_index:
        #    color="#FF00AA"
        #    center = [  row['coord_y'], row['coord_x'] ]
        #    if row['cluster_ind'] > 0:
        #        center = [  row['cluster_lat'], row['cluster_lng'] ]
        #else:
        #    color=""

        #TODO: Handle Map selection of one NP in colocated marker
        #TODO: fix this temp workaround to map each co-located ein to a NP number
        
        #pop_msg = present_num +  row.major_city 
        pop_msg = row.major_city #TODO add other stuff

        # pop_msg += "<br>" + row.STREET
        # pop_msg += "<br> x: " + str(row.coord_x) +  "<br>  y:  " + str(row.coord_y) 


        marker_color = "blue"
        if row.major_city == srch_city:
            marker_color = "red"

        folium.CircleMarker(
            #location=[  row['dc_longitude'], row['dc_latitude'] ],
            location=[  row['dc_latitude'], row['dc_longitude'] ],
            #icon= number_DivIcon(color,num),
            # popup= folium.Popup(pop_msg, max_width=300), 
            # tooltip = present_num + row.NAME
            tooltip = pop_msg, 
            radius=3,
            color=marker_color
        ).add_to(nps2)

    m2.add_child(nps2)

    #folium.LayerControl(collapsed=False).add_to(m2)
   
    # call to render Folium map in Streamlit
    st_m2 = st_folium(m2, 
                      width=675, 
                      height=450,
                      center=center,
                      # key="city_folium",
                        zoom=7    
                      )
    # if user clicked on this map, save to sessions
    #if st_m2
    st.write ("OK, here's st_m2")
    st.write(st_m2)
    # st.session_state["city_folium_status"] = "Not"

    if "last_object_clicked_tooltip" in st_m2:
        st.write ("s2t m2 last object clicked tooltip ")
        if st_m2["last_object_clicked_tooltip"] is None:
            st.write ("ok tooltip is None")
        else:
            st.write (st_m2["last_object_clicked_tooltip"])
            st.session_state["city_folium_status"] = st_m2["last_object_clicked_tooltip"]
        
        #if not pd.isnull(st_m2["last object clicked tooltip"]):
        #    st.write (st_m2["last object clicked tooltip"])

    #st.session_state["cities_folium_city"] 
    return st_m2

def cities_on_map(ny_cities_df, srch_city):
    """
    vega in streamlit doesn't allow events on maps 
    with multiple layers
       
    """

    filt = ny_cities_df["dc_longitude"].notnull()

    # only want new york shape
    us_10m  = data.us_10m()
    # Select the geometries under states under objects, filter on id 36
    us_10m['objects']['states']['geometries']=[item for item in us_10m['objects'] \
        ['states']['geometries'] if item['id'] in [36]]
    # Make the topojson data
    states = alt.Data(
        values=us_10m, 
        format=alt.TopoDataFormat(feature='states',type='topojson'))

    # Plot background (now only NY) 
    background = alt.Chart(states).mark_geoshape(
        fill='lightgray',
        stroke='white',
        limit=1000
    ).properties(
        title='NY Nonprofits',
        width=700,
        height=400
    ) #.project("mercator")

    # Plot the points
    point_selector = alt.selection_point("point_selection", 
                                         fields=['major_city', 'uszip_county'] )

    points=alt.Chart(ny_cities_df[filt]).mark_circle(size=60).encode(
        longitude = "dc_longitude",
        latitude = "dc_latitude",
        #tooltip=['major_city', 'np_cnt_rank_seq', 'inc_rank_seq', 
        #    'inc_tot', 'np_cnt',
        #    'city_population'],

        tooltip = [
                        {"field": "major_city", 
                            "type": "ordinal", 
                            "title": "City"
                            },
                        {"field": "inc_tot",
                         "type": "quantitative", 
                         "title": "Income Total", 
                         "format" : "$,.0f"
                         },
                          {"field": "np_cnt",
                         "type": "quantitative", 
                         "title": "Nbr NPs", 
                         "format" : ",.0f"
                         },
                        {"field": "city_population",
                         "type": "quantitative", 
                         "title": "Population", 
                         "format" : ",.0f"
                         },
                        {"field": "np_cnt_rank_seq",
                         "type": "quantitative", 
                         "title": "Nbr NPs Rank", 
                         "format" : ",.0f"
                         },

                        {"field": "inc_rank_seq",
                         "type": "quantitative", 
                         "title": "Nbr NPs Rank", 
                         "format" : ",.0f"}
                        ],

        color=alt.condition(
            alt.datum.major_city == srch_city,
            alt.value('blue'),  # Color for selected NP
            alt.value('red')   # Color for all others
            #alt.value(st.session_state['highlight_color']),
            #alt.value(st.session_state['point_color'])

        ),

        size=alt.condition(
            alt.datum.major_city == srch_city,
            #alt.Size('size_var', scale=alt.Scale(range=[10, 500]))
            alt.value(120), # selected size
            alt.value(50),
        ),


        #color = "Group"
        #).project("mercator")
        ).add_params(point_selector).project("mercator")

    # Overlay the two plots
    #st.altair_chart(background+points, theme="streamlit", use_container_width=True)
    #event = st.altair_chart(background+points,  use_container_width=True, key="cities_on_map", on_select="rerun")
    
    event = st.altair_chart(points,  
                        use_container_width=True, 
                        key="alt_cities_on_map", 
                        on_select="rerun")



def Cities_Income_and_Nbr_Orgs(df, srch_city):
    
    st.write("Number of Cities:  " + str(len(df)))
    st.write ("City of Interest: " + srch_city)
    # Select columns to plot
    #x_column = st.selectbox("Select X-axis column", df.columns)
    #y_column = st.selectbox("Select Y-axis column", df.columns)

    #x_column = df['inc_rank_seq']
    x_column = 'inc_rank_seq'
    #y_column = df['np_cnt_rank_seq']
    y_column = 'np_cnt_rank_seq'
    city_name = 'major_city'
    #x_column = 'x'
    #y_column = 'y'

    #tt = {'major_city' : city_name,  
    #    'inc_rank_seq' : x_column,
    #    'np_cnt_rank_seq' : y_column}

    # change color of selected city
    # color=alt.condition("datum.species == 'setosa'", alt.value('yellow'), alt.Color('species'))

    point_selector = alt.selection_point("point_selection", 
                                         fields=['major_city', 'uszip_county'] )
    # Create the chart
    chart = alt.Chart(df).mark_circle().encode(
        x=alt.X('inc_rank_seq', axis=alt.Axis(title='Income Ranking')),
        y=alt.Y('np_cnt_rank_seq', axis=alt.Axis(title='NP Count Ranking')),
        
        tooltip = [
                        {"field": "major_city", 
                            "type": "ordinal", 
                            "title": "City"
                            },
                        {"field": "inc_tot",
                         "type": "quantitative", 
                         "title": "Income Total", 
                         "format" : "$,.0f"
                         },
                          {"field": "np_cnt",
                         "type": "quantitative", 
                         "title": "Nbr NPs", 
                         "format" : ",.0f"
                         },
                        {"field": "city_population",
                         "type": "quantitative", 
                         "title": "Population", 
                         "format" : ",.0f"
                         },
                        {"field": "np_cnt_rank_seq",
                         "type": "quantitative", 
                         "title": "Nbr NPs Rank", 
                         "format" : ",.0f"
                         },

                        {"field": "inc_rank_seq",
                         "type": "quantitative", 
                         "title": "Nbr NPs Rank", 
                         "format" : ",.0f"}
                        ],

        color=alt.condition(
            alt.datum.major_city == srch_city,
            alt.value('blue'),  # Color for selected NP
            alt.value('red')   # Color for all others
        ),

        size=alt.condition(
            alt.datum.major_city == srch_city,
            #alt.Size('size_var', scale=alt.Scale(range=[10, 500]))
            alt.value(120), # selected size
            alt.value(50),
        ),
        opacity=alt.value(0.5)  # Set the opacity to 0.5
    
    ).add_params(point_selector)

    event = st.altair_chart(chart,  use_container_width=True, 
                key="alt_cities_income_and_nbr_org",   # alt_chart",
                on_select="rerun")


def alt_city_pop_inc(df, srch_city):

    # Create a Streamlit app
    #st.write (rpt_config["Cities:  Income and Nbr Orgs"]["rpt_title"])
    #st.write( rpt_config["Cities:  Income and Nbr Orgs"]["rpt_desc"])
    st.write("Number of Cities:  " + str(len(df)))
    st.write ("City of Interest: " + srch_city)
    
    # Select columns to plot
    #x_column = st.selectbox("Select X-axis column", df.columns)
    #y_column = st.selectbox("Select Y-axis column", df.columns)

    # change color of selected city
    # color=alt.condition("datum.species == 'setosa'", alt.value('yellow'), alt.Color('species'))

    point_selector = alt.selection_point("point_selection", 
                                         fields=['major_city', 'uszip_county'] )
    # Create the chart
    chart = alt.Chart(df).mark_circle().encode(
        #x='x_column:Q',
        #y='y_column:Q',
        #x=x_column,
        #y=y_column,
        x = 'inc_tot',
        y= 'city_population',

        #tooltip=[alt.Tooltip(city_name)]
        #tooltip=['major_city', 'np_cnt_rank_seq', 'inc_rank_seq', 
        #         'inc_tot', 'np_cnt',
        #         'city_population'],

        tooltip = [
                        {"field": "major_city", 
                            "type": "ordinal", 
                            "title": "City"
                            },
                        {"field": "inc_tot",
                         "type": "quantitative", 
                         "title": "Income Total", 
                         "format" : "$,.0f"
                         },
                          {"field": "np_cnt",
                         "type": "quantitative", 
                         "title": "Nbr NPs", 
                         "format" : ",.0f"
                         },
                        {"field": "city_population",
                         "type": "quantitative", 
                         "title": "Population", 
                         "format" : ",.0f"
                         },
                        {"field": "np_cnt_rank_seq",
                         "type": "quantitative", 
                         "title": "Nbr NPs Rank", 
                         "format" : ",.0f"
                         },

                        {"field": "inc_rank_seq",
                         "type": "quantitative", 
                         "title": "Nbr NPs Rank", 
                         "format" : ",.0f"}
                        ],

        color=alt.condition(
        alt.datum.major_city == srch_city,
        alt.value('blue'),  # Color for selected NP
        alt.value('red')   # Color for all others
        ),

        size=alt.condition(
            alt.datum.major_city == srch_city,
            #alt.Size('size_var', scale=alt.Scale(range=[10, 500]))
            alt.value(120), # selected size
            alt.value(50),
        ),
        opacity=alt.value(0.5)  # Set the opacity to 0.5
    
        #color='category'
    ).add_params(point_selector)

    event = st.altair_chart(chart,  use_container_width=True, key="alt_city_pop_inc", on_select="rerun")
    # st.write (event)
    
    # Display the chart
    #st.altair_chart(chart, use_container_width=True)

def Income_and_Emphasis_Area (df, city):
    
    srch_city = city
    
    if srch_city == "(all)":
        result = df.groupby(['ntee_cat'])['INCOME_AMT'].sum().reset_index()
        nps_in_rpt = str(len(df))
    else:
        filt = df['major_city'] == srch_city
        result = df[filt].groupby(['ntee_cat'])['INCOME_AMT'].sum().reset_index()
        nps_in_rpt = str(len(df[filt]))

    # Calculate the sum of the 'Value' column for each category
    #sum_by_category = data.groupby('Category')['Value'].sum().reset_index()

    st.write ("Number of Nonprofits:  " + nps_in_rpt)

    # Create the Altair chart
    #chart = alt.Chart(result, 
    #                    title="Simple Income for NTEE").mark_bar().encode(
    #                    x='ntee_cat',
    #                    y='INCOME_AMT',
    #                    tooltip = [
    #                    {"field": "ntee_cat", 
    #                        "type": "ordinal", 
    #                        "title": "NTEE Cat"
    #                        },
    #                    {"field": "INCOME_AMT", "type": "quantitative", 
    #                     "title": "Income Total", "format" : "$,.0f"}
    #                    ]

     #               )
    
    # st.altair_chart(chart, theme="streamlit", use_container_width=True)

    # Create the Altair chart
    chart = alt.Chart(result,
                      title="Sorted by Income").mark_bar().encode(
        x=alt.X('sum(INCOME_AMT):Q', title='Sum of Income'),
        y=alt.Y('ntee_cat:N', sort='-x', title='NTEE Category'),
                        #x='INCOME_AMT',
                        #y='ntee_cat',
                        tooltip = [
                        {"field": "ntee_cat", 
                            "type": "ordinal", 
                            "title": "NTEE Cat"
                            },
                        {"field": "INCOME_AMT", "type": "quantitative", 
                         "title": "Income Total", "format" : "$,.0f"}
                        ]

    
    )
    st.altair_chart(chart, theme="streamlit", use_container_width=True)

    # Create the Altair chart
    #chart = alt.Chart(result,
    #                  title="Sort -y, format tooltip").mark_bar().encode(
    #    x=alt.X('ntee_cat:O',  sort='-y', title='NTEE Category'),
    #    y=alt.Y('sum(INCOME_AMT):Q',  title='Sum of NP Income'),
        #x=alt.X('sum(INCOME_AMT):Q', title='Sum of Value'),
        #y=alt.Y('ntee_cat:N', sort='-x', title='Category')
        #tooltip=[alt.Tooltip("INCOME_AMT:Q", format=",.0f")]
    #    tooltip=[alt.Tooltip("INCOME_AMT:Q", format="$,.0f")]
    #)
    #st.altair_chart(chart, theme="streamlit", use_container_width=True)

def rpt_cities_income_rank(ny_cities_df, city, nbr_results):
    # ------------------------------------
    # Cities and NP Income Ranking (*)
    # ------------------------------------

    # Bar Horizontal of Cities and Sum of Incomes, Ranked
    # Provide a City Name, Nbr of Results wanted
    # or ranking range

    results_cnt = nbr_results # how many cities to get

    # provide a city or give rank start/end
    if city == '(all)':
        srch_city = 'New York'        
    else:
        srch_city = city
    
    # Find Rank of City, then get half results count above and below in ranking
    filt = ny_cities_df['major_city'] == srch_city
    srch_city_rank = ny_cities_df[filt]['inc_rank_seq'].values[0]

    rank_start = ny_cities_df[filt]['inc_rank_seq'].values[0] - (int(results_cnt/2))
    rank_end = ny_cities_df[filt]['inc_rank_seq'].values[0] + (int(results_cnt/2))

    if rank_start < 1:
        rank_start = 1
    rank_end = rank_start + results_cnt

    cities_cnt = len(ny_cities_df.index) # Total Nbr of Cities
    label_format =  '{:,.2f} M'
    xlabel_format = '{:,.0f} M'

    fig, ax = plt.subplots()

    # sort out fig height depending on how rows being returned
    figh = int( ( (rank_end - rank_start) / 3)) 
    fig.set_figheight(figh)

    y_pos =np.arange (rank_end - rank_start)

    # get cities in rank range
    filt = (ny_cities_df['inc_rank_seq'] >= rank_start) & (ny_cities_df['inc_rank_seq'] < rank_end) 
    selected_cities = ny_cities_df[filt].sort_values(by=['inc_rank_seq'])

    # build bar color matrix
    bar_color = st.session_state['blueish']  
    my_colors = [bar_color] * results_cnt
    #my_colors = [bar_color] * len(selected_cities)

    # make selected city another color
    city_list = selected_cities['major_city'].to_list()
    my_colors[city_list.index(srch_city)] = st.session_state["orangey"]

    # get income for each selected city
    city_inc = selected_cities['inc_tot'] / 1000000

    # create labels for each city
    city_labs = selected_cities['major_city'] + ' ' + \
                "(" + selected_cities['inc_rank_seq'].astype(str) + ")"

    ax.barh(y_pos, city_inc, align='center', color=my_colors)

    ax.set_yticks(y_pos, labels=city_labs)

    ax.invert_yaxis()  # lowest rank (greater income) at top

    ax.set_xticks(ax.get_xticks())
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticklabels([xlabel_format.format(x) for x in ticks_loc], rotation=90)

    # label values on bars
    #matplotlib.pyplot.text(x, y, s, fontdict=None, **kwargs)
    for index, value in enumerate(city_inc):
        plt.text(value/2, index + .15,
                label_format.format(value),
                color = 'black'
                )

    ax.set_xlabel('Sum of NP Income')

    chart_title = f'City by Income Rank {rank_start} to {rank_end} (of {cities_cnt}) '
    chart_title += f'\n(Centered on {srch_city})'
    #ax.set_title(f'City by Income Rank {rank_start} to {rank_end} (of {cities_cnt}) ')
    ax.set_title(chart_title)

    plt.margins(y=0.01)

    #plt.show()
    st.pyplot(fig)

def rpt_cities_np_cnt_rank(ny_cities_df, city, nbr_results):

    # ------------------------------------
    # Cities and Number of Nonprofits Ranking 
    # ------------------------------------

    # Bar Horizontal of Cities and Sum of Incomes, Ranked
    # Provide a City Name, Nbr of Results wanted
    # or ranking range

    # Notes:
    #   Using ny_cities_df dataframe
    #TODO:  add other census data to a cities dataframe

    # provide a city or give rank start/end
    if city == '(all)':
        srch_city = 'New York'        
    else:
        srch_city = city

    results_cnt = nbr_results # how many cities to get
    # provide a city or give rank start/end

    # Find Rank of City, then get half results count above and below in ranking
    filt = ny_cities_df['major_city'] == srch_city
    srch_city_rank = ny_cities_df[filt]['np_cnt_rank_seq'].values[0]

    rank_start = ny_cities_df[filt]['np_cnt_rank_seq'].values[0] - (int(results_cnt/2))
    rank_end = ny_cities_df[filt]['np_cnt_rank_seq'].values[0] + (int(results_cnt/2))

    if rank_start < 1:
        rank_start = 1
    rank_end = rank_start + results_cnt

    cities_cnt = len(ny_cities_df.index) # Total Nbr of Cities
    label_format =  '{:,.0f} NPs'

    fig, ax = plt.subplots()

    # sort out fig height depending on how rows being returned
    figh = int( ( (rank_end - rank_start) / 3))
    fig.set_figheight(figh)

    y_pos =np.arange (rank_end - rank_start)

    # get cities in rank range
    filt = (ny_cities_df['np_cnt_rank_seq'] >= rank_start) & (ny_cities_df['np_cnt_rank_seq'] < rank_end)
    selected_cities = ny_cities_df[filt].sort_values(by=['np_cnt_rank_seq'])

    # build bar color matrix
    bar_color = st.session_state['blueish']  #"#54b1f0"
    my_colors = [bar_color] * results_cnt
    #my_colors = [bar_color] * len(selected_cities)

    # make selected city another color
    city_list = selected_cities['major_city'].to_list()
    my_colors[city_list.index(srch_city)] = st.session_state['orangey']  # "#f0b056"

    # get income for each selected city
    city_inc = selected_cities['np_cnt'] 

    # create labels for each city
    city_labs = selected_cities['major_city'] + ' ' + \
                "(" + selected_cities['np_cnt_rank_seq'].astype(str) + ")"

    ax.barh(y_pos, city_inc, align='center', color=my_colors)

    ax.set_yticks(y_pos, labels=city_labs)

    ax.invert_yaxis()  # lowest rank (greater income) at top

    ax.set_xticks(ax.get_xticks())
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticklabels([label_format.format(x) for x in ticks_loc], rotation=90)

    # label values on bars
    #matplotlib.pyplot.text(x, y, s, fontdict=None, **kwargs)
    for index, value in enumerate(city_inc):
        plt.text(value/2, index + .15,
                label_format.format(value),
                color = 'black'
                )

    ax.set_xlabel('Count of Nonprofits')

    chart_title = f'City by Number of NPs Rank {rank_start} to {rank_end} (of {cities_cnt}) '
    chart_title += f'\n(Centered on {srch_city})'
    #ax.set_title(f'City by Income Rank {rank_start} to {rank_end} (of {cities_cnt}) ')
    ax.set_title(chart_title)

    plt.margins(y=0.01)

    #plt.show()
    st.pyplot(fig)

def rpt_income_ntee(df, city, nbr_results):
    srch_city = city
    
    if srch_city == "(all)":
        result = df.groupby(['ntee_cat'])['INCOME_AMT'].sum().sort_values(ascending=False)
    else:
        filt = df['major_city'] == srch_city
        result = df[filt].groupby(['ntee_cat'])['INCOME_AMT'].sum().sort_values(ascending=False)

    # result = df[filt].groupby(['ntee_cat'])['INCOME_AMT'].sum().sort_values(ascending=False)

    label_format =  '{:,.0f} M'

    fig, ax1 = plt.subplots()
    #fig, (ax1, ax2)  = plt.subplots(2, 1)

    ax1.bar(result.index,
            #result.values,
            result.values / 1000000,
            color= st.session_state['blueish'],  #"#54b1f0",
            label='Sum of Orgs Income')
    ax1.legend()
    #ax1.set_xticklabels(result.index, rotation=90)

    # avoid weird warning...
    ax1.xaxis.set_ticks(result.index)
    ax1.xaxis.set_ticklabels(result.index,rotation=90)

    # this odd line remove warning to use formatter before locator
    ax1.set_yticks(ax1.get_yticks())
    ticks_loc = ax1.get_yticks().tolist()
    ax1.set_yticklabels([label_format.format(x) for x in ticks_loc])


    ax1.set_xlabel('NTEE Category')
    ax1.set_ylabel('Orgs Income')
    ax1.set_title(f'{srch_city} NP Orgs Income by NTEE Category')

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bar_label.html
    #ax1.bar_label(ax1.containers[0])

    # plt.show()
    st.pyplot(fig)

def rpt_orgs_category(df, city, bmf, category):
    """ Simple Plot of Org Counts in various IRS Categories
    
    Parameters: 
        df (dataframe): dataframe of NPs
        city (str): City 
        bmf (dict): 
        category (str):  which category or column in df
    
    """

    # plot different BMF codes - by category
    # counting nonprofits

    srch_city = city
  
    if srch_city == '(all)':
        #result = df.groupby(['ntee_cat'])['INCOME_AMT'].sum().sort_values(ascending=False)
        result = df.groupby([category])['p_org_id'].count()

    else:
        filt = df['major_city'] == srch_city
        result = df[filt].groupby([category])['p_org_id'].count()

    # st.write (len(result))
    # populate data dict with all lookups
    data = {}
    xlabs = []

    xy = zip(result.index, result.values)
    for cat in xy:
        data[cat[0]] = cat[1]
        # st.write(cat[0], cat[1])  # Numbers

    sorted_data = {}
    # when there is no data for a lookup category adding zeros to data and setup xlabels 
    for cd in (cd for cd in bmf[category] if cd != 'define') :
        xlabs.append(bmf[category][cd]['short'])
        cd_int = int(cd)
        # st.write (cd, bmf['INCOME_CD'][cd]['short'])
        if cd_int not in data:
            sorted_data[cd] = 0
        else:
            sorted_data[cd] = data[cd_int]

    label_format =  '{:,.0f}'

    # debug
    #print ("data keys:  ", list(data.keys()) )
    #print ("data vals:  ", list(data.values()) )
    #print ("xlabs:  ", xlabs)

    fig, ax1 = plt.subplots()

    ax1.bar(list(sorted_data.keys()),
            list(sorted_data.values()),
            color= st.session_state['blueish'], # "#54b1f0",
            label='Number Of Nonprofits')

    ax1.legend()
    #ax1.set_xticklabels(result.index, rotation=90)

    # avoid weird warning...
    ax1.xaxis.set_ticks(list(sorted_data.keys()))
    ax1.xaxis.set_ticklabels(xlabs,rotation=90)

    # this odd line remove warning to use formatter before locator
    ax1.set_yticks(ax1.get_yticks())
    ticks_loc = ax1.get_yticks().tolist()
    ax1.set_yticklabels([label_format.format(x) for x in ticks_loc])

    ax1.set_xlabel(f"IRS {category} Category")
    ax1.set_ylabel('Number of Nonprofits')
    ax1.set_title(f'{srch_city} NP Orgs and {category}')

    st.pyplot(fig)


def load_rpt_config():
    """ Reports Config
    
    Report Titles, Descriptions and def name
    
    Used to generate sidebar report list
    try using def in the config to make it  could simplify 
    would have to define all defs before loading report config

    """

    # "rpt_name" :  { "rpt_title"  : "Human Readable Title",
    #    "rpt_def_name" : "name_of_def",
    #    "rpt_desc" : "Explanation of Report"
    #    },

    rpt_config = {
                  # "test" : { "rpt_title" : "Test Report",
                  #  "rpt_def_name" : test_report,
                  #  "rpt_desc" :    "See Relationship between Total NP Income and Number or NPs.  Click on bubble to select city"
                  #},
                    # folium implmentation too clunky.  figure out later
                    #"Cities folium" : { "rpt_title" : "Cities Folium in NY",
                    #"rpt_def_name" : cities_folium,
                    #"rpt_desc" :    "Click on bubble to select city"
                  # },
    
                    "Cities on Map" : { "rpt_title" : "Cities in NY",
                    "rpt_def_name" : cities_on_map,
                    "rpt_desc" :    "Click on bubble to select city"
                  },

                  "Rank Cities Income and Nbr Orgs" : { "rpt_title" : "Rank of Number of Nonprofits and Rank of Total Income",
                    "rpt_def_name" : "not_implented",
                    "rpt_desc" :    ("See Relationship between Total NP Income and Number or NPs. " 
                                       " Click on bubble to select city")
                                       
                            },  
              "Rank Cities by NP Income" : { "rpt_title" :  "Rank Cities by NP Income",
                            "rpt_def_name" : rpt_cities_income_rank,
                            "rpt_desc" : "Where the city ranks by Total Income of NPs"
                            },                                                                                         
              "Rank Cities by NP Org Count" : { "rpt_title" : "NP Org Count Rank",
                            "rpt_def_name" : rpt_cities_np_cnt_rank,
                            "rpt_desc" : "Where the city ranks in Number of NPs"
                            },                                             
              #"NP Income by Emphasis Area" : { "rpt_title" : "NP Income by Emphasis Area",
              #              "rpt_def_name" : "name_of_def",
              #              "rpt_desc" : "Duplicate Total Income of NPs by NTEE or Emphasis Area"
              #              },
              #"Orgs and Income" : { "rpt_title" : "Orgs and Income",
              #              "rpt_def_name" : "name_of_def",
              #              "rpt_desc" : "In category rpt, Count Orgs by IRS Income Categories"
              #              },
              #"Orgs and Affiliation" : { "rpt_title" : "Orgs and Affiliation",
              #              "rpt_def_name" : "name_of_def",
              #              "rpt_desc" : "In category rpt, Count Orgs by IRS Affiliation Category - Independent vs Group"
              #              },
            "Income and Emphasis Area" : { "rpt_title" :  "Income and Emphasis Area",
                            "rpt_def_name" : "name_of_def",
                            "rpt_desc" : "Income Reported for Orgs categorized by NTEE Code"
                            },
            
            "Population and NP Income" : { "rpt_title" : "Population and NP Income",
                            "rpt_def_name" : "alt_city_pop_inc",
                            "rpt_desc" : "Relationship between population and total NP Income"
                            },

              "INCOME_CD" : { "rpt_title" : "IRS INCOME_CD",
                            "rpt_def_name" : "name_of_def",
                            "rpt_desc" : "Count Orgs by IRS Income Code"
                            },
              "AFFILIATION" : { "rpt_title" : "IRS AFFILIATION",
                            "rpt_def_name" : "name_of_def",
                            "rpt_desc" : "Count Orgs by IRS Affiliation Code"
                            },
                "ORGANIZATION" : { "rpt_title" : "ORGANIZATION",
                            "rpt_def_name" : "name_of_def",
                            "rpt_desc" : "Count Orgs by IRS by IRS Organization Code"
                            },                                                   
                "FOUNDATION" : { "rpt_title" : "FOUNDATION",
                            "rpt_def_name" : "name_of_def",
                            "rpt_desc" : "Count Orgs by IRS by IRS Affiliation Code"
                            },
                "SUBSECTION" : { "rpt_title" : "SUBSECTION",
                            "rpt_def_name" : "name_of_def",
                            "rpt_desc" : "Count Orgs by IRS Subsection Code"
                            },
                "ASSET_CD" : { "rpt_title" : "ASSET_CD",
                            "rpt_def_name" : "name_of_def",
                            "rpt_desc" : "Count Orgs by IRS by IRS Asset Code"
                            },                                                                 
            "DEDUCTIBILITY" : { "rpt_title" : "DEDUCTIBILITY",
                            "rpt_def_name" : "name_of_def",
                            "rpt_desc" : "Count Orgs by IRS Deductibility Code"
                            }             
          }
                    
    return rpt_config

# --- end report definitions and configs

@st.cache_data
def get_np_local_df():
    dtype = {"CLASSIFICATION": str,
         "EIN" : str,
         "ACTIVITY" : str,
         "AFFILIATION" : str,
         "ORGANIZATION" : str,
         "FOUNDATION" : str,
         "NTEE_CD" : str,
         "RULING" : str,
         "ZIP" : str,
         "TAX_PERIOD" : str,
         "GROUP" : str,
         "cb_BASENAME" : int, 
         "cb_BLKGRP" : int,
         "cb_BLOCK": str,
         "cb_GEOID" : str,
         "ZipCd" : str
         }

    # get nonprofit data, then setup incremental index
    np_local_df = pd.read_csv('data/np_local_df.csv')
    np_local_df.sort_values(by=['NAME'], inplace=True)
    np_local_df.reset_index(drop=True, inplace=True)

    # start index at 1 for humans, when matching org to map number
    # this matches the other local nonprofits apps, but used here 
    np_local_df.index += 1

    return np_local_df


@st.cache_data
def load_ny_cities_df():
    # get nonprofit data, then setup incremental index
    #TODO: new data structure...
    #ny_cities_df = pd.read_csv('data/ny_cities_p_df.csv')

    dtypes = {'inc_rank' : int,	
             'nbr_np_rank' : int,
             'dc_wikidataId' : str,
             'dc_usCensusGeoId' : str,
             #'city_population' : int
             }

    ny_cities_df = pd.read_csv('data/bmf_cities_p_df.csv', dtype=dtypes)

    return ny_cities_df


@st.cache_data
def load_city_list(ny_cities_df):
    #city_list = ["(all)"] + ny_cities_df['CITY'].sort_values().to_list()
    city_list = ["(all)"] + ny_cities_df['major_city'].sort_values().to_list()
    #city_list = ny_cities_df['CITY'].sort_values().to_list()
    return city_list


@st.cache_data
def load_np_ny_p_df():

    #TODO: change to all_ny_np_df... but would have to modify 
    #       existing report with code lookups...
    dtypes = {"CLASSIFICATION": str,
         "EIN" : str,
         "ACTIVITY" : str,
         #"AFFILIATION" : str,
         #"ORGANIZATION" : str,
         #"FOUNDATION" : str,
         "NTEE_CD" : str,
         "RULING" : str,
         "ZIP" : str,
         "TAX_PERIOD" : str,
         #"GROUP" : str,
         "zipcode" : str

         }

    # get nonprofit data, then setup incremental index
    # np_ny_p_df = pd.read_csv('data/np_ny_p_df.csv')
    np_ny_p_df = pd.read_csv('data/all_ny_np_df.csv', dtype=dtypes)

    # a way to insert propublica link
    # TODO: add the column in colab prep steps
    pp_link = "https://projects.propublica.org/nonprofits/organizations/" # EIN

    np_ny_p_df['pp_link'] = np_ny_p_df['EIN'].apply(
                    lambda x: pp_link + x
    )

    return np_ny_p_df

@st.cache_data
def load_ntee():
    ntee = {}
    eo = {}

    with open('data/NCCS_NTEE.csv', mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            # print(f"0 {row[0]}   1 {row[1]}    2 {row[2]}" )
            ntee[row[0]] = [row[1], row[2]]
    
    return ntee

@st.cache_data
def load_bmf_codes():
    """ Load lookup values for IRS codes in Business Master File (BMF)
    
    Returns: bmf (dict)

    """

    # Nested dictionary, like this 
    #bmf = { 'INCOME_CD': {'code1': 'XX', 
    #                              'codevalue1': '19', 
    #                              {'short' : 'brief definition', 
    #                              'full' :  'full definition'}

    bmf = {}
    bmf_set = set() # get each unique BMF field/column name
    all_rows = []

    with open('data/IRS_BMF_Lookups_wShortDesc.csv', mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            # print(f"0 {row[0]}   1 {row[1]}    2 {row[2]}  3 {row[3]}" )
            bmf_set.add(row[0])  # get unique column names
            all_rows.append(row) # get rows to add to column name dict

    for bmf_cols in bmf_set:
        #print ("Loading Data for column: ", bmf_cols)
        bmf[bmf_cols] = {}
        for row in all_rows:
            if row[0] == bmf_cols:
                #bmf[bmf_cols][row[2]] = row[3]
                bmf[bmf_cols][row[2]] = {"full" : row[3], "short": row[4]}

    return bmf
    
def show_bmf_lookups(bmf, category):



    tbl_html =  " <table> " 
    tbl_html += "     <tr> "  
    tbl_html += f"      <td colspan=2> Code Definitions:  {category} </td>"
    tbl_html += "     </tr> "
    tbl_html += "     <tr>"
    tbl_html += "       <td colspan=2> " + bmf[category]["define"]["full"] + "</td>"
    tbl_html += "     </tr>"
    tbl_html += "     <tr>"
    tbl_html += "       <td> Code </td> <td> Meaning </td>"
    tbl_html += "     </tr>"
    
    for c in (c for c in bmf[category] if c != 'define') :
        tbl_html += "   <tr>"
        tbl_html += "     <td> " + c + " </td> <td> " + bmf[category][c]["full"]  + "</td>"
        tbl_html += "            </tr>"


    tbl_html += "</table>"        

    with  st.expander("Code Lookups", expanded=False ):
        st.markdown(tbl_html, unsafe_allow_html=True)

    #for c in (c for c in bmf[category] if c != 'define') :
    #    tbl_html += "   <tr>"
    #    tbl_html += "     <td> " + c + " </td> <td> " + bmf[category][c]["full"]  + "</td>"
    #    tbl_html += "            </tr>"


    #df = pd.DataFrame(
    #    np.random.randn(10, 5), columns=("col %d" % i for i in range(5))
    #)

    # quick and dirty, shows both short and full 
    #c = pd.DataFrame(bmf[category])
    #st.table(c.T)

def data_profile(ny_cities_df, np_ny_p_df ):
    app_summary = """
    
    #### Summary
    
    This app presents data on Nonprofits in New York State.  Select a Plot in left sidebar.
     
    The core dataset was collected from the IRS in July 2024 with supplemental information
    on cities and demographics from a variety of other sources, like the [US Census](https://www.census.gov/data/developers/data-sets.html) and 
    [Google Data Commons](https://datacommons.org/).  
    
    It's an early draft of a curiosity and learning project, but demonstrates some 
    potential for creating a way to layer information about a community.  
    
    This app is an offshoot of an effort to look at more detailed info on nonprofits
    in a local area:
    - [Nonprofits in Cortland, NY](https://local-nonprofits-jzelson.streamlit.app/) 
    - [Visualization of People connections in Cortland](https://johnzelson-local-nonprofits-graph-streamlit-app-zuq6hl.streamlit.app/)

    It would be interesting to make an app that is easily refreshed and 
    extended that would enable analysis and discovery about community. (language)
    
    If you want to make it better, see the [Github repository](https://github.com/johnzelson/ny-nonprofits) 

    """
    st.markdown(app_summary)

    nbr_cities = len(ny_cities_df) 
    nbr_nps =  len(np_ny_p_df)

    data_summary = f"""
    
    #### Dataset Summary

    - Total Number of Nonprofits:  {nbr_nps} 
    - Number of Cities:  {nbr_cities} 

    
    """    
    st.markdown(data_summary)


def city_bmf_profile(city, ny_cities_df):
    """ Show a tidy profile of a city
        very drafty

    """
    # df_dict = np_local_df.filter(items=[np_df_selected_index], axis=0).to_dict('records')[0]


    flds = ['major_city',	
                'uszip_county',
                'city_population',
                'city_pop_src',
                'inc_tot',
                'np_cnt',
                'inc_rank',
                'nbr_np_rank',
                'inc_rank_seq',
                'np_cnt_rank_seq',	
                'dc_place',
                'Median_Age_Person',
                'dc_usCensusGeoId',
                'dc_wikidataId',      
                'dc_latitude',
                'dc_longitude'
        ]   


    # st.write ("City Details from dataframe")
    filt = ny_cities_df['major_city'] == city
    # st.table(ny_cities_df[filt][flds].T)

    #st.write ("City Details as Dict")

    #df_dict = np_local_df.filter(items=[np_df_selected_index], axis=0).to_dict('records')[0]
    df_dict = ny_cities_df[filt].to_dict('records')[0]
    #st.table(df_dict)
    st.markdown ("#### City Profile: " + city)

    profile_html = "<table> <tr>"
    profile_html += "<td colspan=4>" + df_dict["major_city"] + ", " + df_dict["uszip_county"]
    profile_html += "</td> </tr>"

    profile_html += "<tr> <td> Total Income of NPs </td> "
    ti = '${:,.0f}'.format(df_dict["inc_tot"])

    profile_html += "<td> " +  ti + "</td>"
    profile_html += "<td> Total Nbr of NPs </td> "
    profile_html += "<td> " +  str(df_dict["np_cnt"]) + "</td>"
    profile_html += "</tr>"


    profile_html += "<tr> <td> City Population </td>"
    cp = '{:,.0f}'.format(df_dict["city_population"])
    profile_html += "<td> " + cp + "</td>"
    # profile_html += " (" + df_dict["city_pop_src"] + "</td>"
    profile_html += "<td> Median Age </td>"
    profile_html += "<td> " + str(df_dict["Median_Age_Person"])  + "</td>"
    profile_html += "</tr>"

    profile_html += "<tr> <td> dc_usCensusGeoId </td>" 
    profile_html += "<td>" + str(df_dict["dc_usCensusGeoId"]) +  "</td>"
    profile_html += "<td> dc_wikidataId </td>" 

    profile_html += "<td> "
    
    if not pd.isna(df_dict["dc_wikidataId"]):
        profile_html += "<a href=\"https://www.wikidata.org/wiki/" + str(df_dict["dc_wikidataId"]) + "\""
        profile_html += " _target=blank>" +  str(df_dict["dc_wikidataId"]) + "</a>"
    else:
        profile_html += " "

    profile_html += "</td>"

    #TODO: add other links to external, like google map with lat lng
    
    profile_html +=  "</td>"
    profile_html += "</tr>"


    if not pd.isna(df_dict["dc_latitude"]):
        # map or search...
        #https://www.google.com/maps/@?api=1&map_action=map&center=-33.712206%2C150.311941&zoom=12
        g_url = "https://www.google.com/maps/@?api=1&map_action=map&center="
        #g_url = "https://www.google.com/maps/search/?api=1&query="
        g_url += str(df_dict["dc_latitude"]) + "%2C"
        g_url += str(df_dict["dc_longitude"])
        g_url += "&zoom=15"
        # 47.5951518%2C-122.3316393

        profile_html += "<tr> <td> Lat / Lng</td>" 
        profile_html +=  "<td>" + str(df_dict["dc_latitude"])  # +  "</td>"
        # profile_html += "<td> Lat </td>" 
        # profile_html += "<td>" + str(df_dict["dc_longitude"]) +  "</td>"
        profile_html +=    " / "
        profile_html +=  str(df_dict["dc_longitude"]) 

        profile_html += "</td>"
        profile_html += "<td colspan=2> <a href=\"" + g_url + "\" "
        profile_html += " _target=blank>Google Map &#8594;</a>"
        profile_html += "</td> </tr>"


    profile_html += "</table> "
    st.markdown(profile_html, unsafe_allow_html=True)


    # ny_cities_df[filt][flds]

def ein_click():
    #TODO: remove
    st.write(st.session_state)

def list_city_nps(city, np_ny_p_df):
    flds = ['pp_link',
        'EIN',
        'NAME',
        'STREET',
        #'major_city',
        #'uszip_county',
        'ASSET_AMT',
        'INCOME_AMT',
        'REVENUE_AMT',
        'ntee_cat',
        #'ntee_define',
        'act1_lu',
        'act2_lu',
        'act3_lu',
        'RULE_DT',
        'aff_lu',
        'org_lu',
        'found_lu',
        'deduct_lu',
        'zipcode',

        'area_code_list',
        'population',
        'population_density',
        'housing_units',
        'occupied_housing_units',
        'median_home_value',
        'median_household_income'
        ]

    # temp, insert propublica link...

    

    st.markdown ("#### Nonprofits in " + city)
    st.markdown ("###### Click on header to sort Nonprofits")
    filt = np_ny_p_df['major_city'] == city
    #st.table(np_ny_p_df[filt][flds])


    #df['B'] = df['A'].apply(lambda x: x * 2)

    st.dataframe(np_ny_p_df[filt][flds], 
                 
                 column_config={
                    "pp_link": st.column_config.LinkColumn(
                            "Propublic Profile",
                            display_text= "Open Profile",
                            width=100
                            )
                    },
        
                  hide_index=True)
    #selection = st.dataframe(np_ny_p_df[filt][flds],  
    #                                hide_index=True, key="city_profile",
    #                                 on_select=ein_click)    

def do_sidebar(reports_list, city_list, selected_city):

    with st.sidebar:
        select_tab, help_tab = st.tabs(["Create Plot", "Help"])

        if selected_city != '(all)':
            sel_city_index = city_list.index(selected_city)
        else:
            sel_city_index = 0

        with select_tab:
            with st.form("report_params"):
            #with st.expander("Reports", expanded=True): #temp exp without forms
                st.write ("Pick a Report") 
                report_name = st.selectbox("Plot", reports_list, index=0, key="sb_rpt_name_key")

                city = st.selectbox("City of Interest", city_list, index=sel_city_index, 
                                    key="sb_city_key",
                                    help="Typing the city name will jump to desired city.  (Not relevant to some plots)")

                nbr_results = st.selectbox("Number of Cities to Compare", [10, 20, 30, 40, 50], index=1)

                submitted = st.form_submit_button("Show Plot")
                #doit = st.button("Generate", key="doit")

        with help_tab:
            help_md = """
            
            This app is a window into all the active Nonprofits (over 100k nonprofits in 1522 cities) 
            in New York State.

            In left sidebar, select a plot and click "Show Button."

            The City of Interest isn't relevant to every plot.

            In some plots, user can click on cities to change city of interest.

            A city profile and list of NPs in that city is presented under plots 
            where City of Interest has been selected
                        
            """
            st.markdown (help_md)


    return report_name, city, nbr_results

def main():
    st.set_page_config(APP_TITLE) #, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUB_TITLE)

    #TODO   related to handling folium
    rpt_return = None

    # load data 
    ntee = load_ntee()
    bmf = load_bmf_codes()
    ny_cities_df = load_ny_cities_df()
    # st.write (len(ny_cities_df))  # debug: verify
    
    # create list of cities for selectbox
    city_list = load_city_list(ny_cities_df)
    # st.write (len(city_list))

    np_ny_p_df = load_np_ny_p_df()

    # set some graph params
    # TODO: implement highlight_color vs regular_color in all reports
    if 'blueish' not in st.session_state:
        st.session_state['blueish'] = "#54b1f0"
        st.session_state['orangey'] = "#f0b056"

        st.session_state['highlight_color'] = "#54b1f0"
        st.session_state['point_color'] = "#f0b056"

    # load reports list, configs
    rpt_config = load_rpt_config()
    rpt_names_list = list(rpt_config.keys())
    reports_list = ["Select..."] + rpt_names_list 

    # ----------------------------- 
    # Determine whether user clicked report/city in sidebar  
    # or has clicked on a new city on a plot
    
    # debug
    # st.write(st.session_state)
    
    #NOTE:  so far, report can only be changed in sidebar, but some future 
    # iteration would allow user to jump to related plots from a plot
    
    # on initial load, formsubmitter not in session
    if "FormSubmitter:report_params-Show Plot" in st.session_state:

        # when formsubmitter is true, then user clicked on sidebar
        if st.session_state["FormSubmitter:report_params-Show Plot"]:
            selected_rpt = st.session_state["sb_rpt_name_key"]
            selected_city = st.session_state["sb_city_key"]

        # otherwise user clicked in a plot

        # handle user clicks on altair/vega plots, get new selected city  
        # session state has the altair chart key with city in point selection

        #TODO: use report config, prob use rpt_def_name in session
        # making all reports return city, would only need to call report with return 
        elif "alt_city_pop_inc" in st.session_state: 
            selected_city = st.session_state["alt_city_pop_inc"]["selection"]["point_selection"][0]["major_city"] 
            #st.write(st.session_state["alt_city_pop_inc"]["selection"]["point_selection"][0]["major_city"])
        elif "alt_cities_on_map" in st.session_state: 
            selected_city = st.session_state["alt_cities_on_map"]["selection"]["point_selection"][0]["major_city"] 
            #st.write(st.session_state["alt_chart"]["selection"]["point_selection"][0]["major_city"])
        elif "alt_cities_income_and_nbr_org" in st.session_state: 
            selected_city = st.session_state["alt_cities_income_and_nbr_org"]["selection"]["point_selection"][0]["major_city"] 
            #st.write(st.session_state["alt_chart"]["selection"]["point_selection"][0]["major_city"])

        #TODO:  deal with integrating folium maps
        # leaving stub, but not implemented     
        # trying to city if clicked on, clunky reruns 
        # 
        # --- stub ignored ----
        elif st.session_state["sb_rpt_name_key"] == "Cities folium": 
            st.write ("user selected a city from map...")
            if "city_folium_status" in st.session_state:
                selected_city = st.session_state["city_folium_status"]
                st.write ("folium map session, set selected city")
                st.write (selected_city)

            else:
                selected_city = st.session_state["sb_city_key"]
        # --- end stub ignored ----                            

        else:
            #TODO: check: shouldn't get here...
            selected_city = st.session_state["sb_city_key"]
    else: 
        #st.write ("formsubmitter not in session.  just arrived?")
        selected_city = '(all)'
        #TODO: consider renaming to welcome()
        data_profile(ny_cities_df, np_ny_p_df)

    
    #NOTE:  so far, don't need to send selected report to sidebar
    #       in future, if user can click in one plot plot generate to another report, will have to... 
    #(report_name, city, nbr_results) = do_sidebar(reports_list, city_list, selected_city)

    (report_name, selected_city, nbr_results) = do_sidebar(reports_list, city_list, selected_city)
    
    # show plot title and description 
    if report_name != "Select...":
        #rpt_md = "#### "+ rpt_config[report_name]["rpt_title"]
        #rpt_md += "###### " + rpt_config[report_name]["rpt_desc"]
        st.markdown ("#### "+ rpt_config[report_name]["rpt_title"])
        st.markdown ("###### " + rpt_config[report_name]["rpt_desc"], unsafe_allow_html=True)
        
    # Do the selected report/plot
    # experiment with putting def names in report config
    if report_name in ["test", "Cities on Map"]:
        #TODO:add return values? 
        rpt_config[report_name]['rpt_def_name'](ny_cities_df, selected_city)
    
    #but folium maps are different    
    elif report_name == "Cities folium":    

        rpt_return = rpt_config[report_name]['rpt_def_name'](ny_cities_df, selected_city)
        
        #st.session_state["cities_folium_city"]

        #if rpt_return:
        #    if "last_object_clicked_tooltip" in rpt_return and rpt_return['last_object_clicked_tooltip']:
        #        st.write("folium map has tooltip")
        #        selected_city = rpt_return["last_object_clicked_tooltip"]
        #        st.write(rpt_return["last_object_clicked_tooltip"])
        #    else:
        #        selected_city = st.session_state["sb_city_key"]


    elif report_name == "Rank Cities by NP Income":
        rpt_cities_income_rank(ny_cities_df, selected_city, nbr_results)

    elif report_name == "Rank Cities by NP Org Count":
        rpt_cities_np_cnt_rank(ny_cities_df, selected_city, nbr_results)

    # removed dupe from config    
    elif report_name == "NP Income by Emphasis Area":
        rpt_income_ntee(np_ny_p_df, selected_city, nbr_results)


    #elif report_name == "Orgs and Income":
        # rpt_income_code(np_ny_p_df, city, bmf)
    #    rpt_orgs_category(np_ny_p_df, selected_city, bmf, 'INCOME_CD')
        #show_bmf_lookups(bmf, 'INCOME_CD')
    #elif report_name == "Orgs and Affiliation":
    #    rpt_orgs_category(np_ny_p_df, selected_city, bmf, 'AFFILIATION')
    #    show_bmf_lookups(bmf, 'AFFILIATION')
    
    
    elif report_name == "Income and Emphasis Area":
        Income_and_Emphasis_Area(np_ny_p_df, selected_city)
    
    elif report_name == "Rank Cities Income and Nbr Orgs":
        Cities_Income_and_Nbr_Orgs(ny_cities_df, selected_city)

    elif report_name == "Population and NP Income":
        alt_city_pop_inc(ny_cities_df, selected_city)

    #TODO:decide how to handle these count orgs by IRS code
    elif report_name in ['INCOME_CD', 'AFFILIATION', 'ORGANIZATION', 'FOUNDATION', 'ASSET_CD', 
                         'SUBSECTION', 'DEDUCTIBILITY']:
        rpt_orgs_category(np_ny_p_df, selected_city, bmf, report_name)
        show_bmf_lookups(bmf, report_name)


    if selected_city != '(all)':
        city_bmf_profile(selected_city, ny_cities_df)
        list_city_nps(selected_city, np_ny_p_df)


if __name__ == "__main__":
    main()
