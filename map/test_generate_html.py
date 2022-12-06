# %% package installation
import numpy as np
import pandas as pd
import pickle
from utils import *

import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os  # operating system tools (check files)

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from bokeh.models import (
    ColorBar,
    GeoJSONDataSource,
    HoverTool,
    LinearColorMapper,
)
from bokeh.palettes import brewer
from bokeh.plotting import figure, save, ColumnDataSource
from bokeh.models import Title
from bokeh.models import TabPanel, Tabs, Div
from bokeh.layouts import column, row
from bokeh.models import CustomJS, Slider

from bokeh.resources import CDN
from bokeh.embed import file_html

from utils import get_data, get_us_map, config_colorbar_range


def draw_map(colorbar_name, field, value_name, title, descip):
    # create colorbar
    tick_labels = {
        2: str(colorbar_name[2]),
        4: str(colorbar_name[4]),
        6: str(colorbar_name[6]),
        8: str(colorbar_name[9]),
    }
    color_bar = ColorBar(
        color_mapper=color_mapper,
        label_standoff=8,
        width=20,
        height=420,
        border_line_color=None,
        orientation="vertical",
        location=(0, 0),
        major_label_overrides=tick_labels,
        major_tick_line_alpha=0.25,
    )

    # Create figure object.
    p = figure(
        height=512,
        width=768,
        toolbar_location="above",
        tools="box_zoom, reset",
    )

    # add title or stuff
    p.add_layout(
        Title(text=descip, text_font_style="italic", text_font_size="9pt"), "above"
    )
    p.add_layout(Title(text=title, text_font_size="11pt"), "above")

    # author = """Created by HOIST"""
    # p.add_layout(
    #     Title(text=author, text_font_style="italic", text_font_size="9pt"), "below"
    # )

    # no line color
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    # Add patch renderer to figure.

    # add states, nan_states and state line patch
    states = p.patches(
        "xs",
        "ys",
        source=geosource,
        fill_color={"field": field, "transform": color_mapper},
        line_color="gray",
        line_width=0.25,
        fill_alpha=1,
    )
    nan_states = p.patches(
        "xs",
        "ys",
        source=nan_geosource,
        fill_color="#F5EFE6",
        line_color="gray",
        line_width=0.25,
        fill_alpha=1,
    )

    state_line = p.multi_line(
        "xs", "ys", source=state_geosource, line_color="black", line_width=0.5
    )

    # add hover tool
    TOOLTIPS = f"""
        <div style="background-color:#F5F5F5; opacity: 0.95;">
            <div style = "text-align:center;">
                <span style="font-size: 12px; font-weight: bold;">@NAME
            </div>
            <div style = "text-align:center;">
                <span style="font-size: 12px; font-weight: bold">Decomposition Score: @{value_name}</span>
            </div>
        </div>
    """
    p.add_tools(HoverTool(renderers=[states], tooltips=TOOLTIPS))

    #### Some features to make it a bit nicer.
    p.axis.visible = False
    p.background_fill_color = "grey"
    p.background_fill_alpha = 0.25

    p.border_fill_color = "#F5F5F5"
    color_bar.background_fill_color = "#F5F5F5"
    p.toolbar.autohide = False
    p.add_layout(color_bar, "right")

    return p


def draw_map_with_slider(colorbar_name, field, value_name, title, descip):
    # create colorbar
    tick_labels = {
        2: str(colorbar_name[2]),
        4: str(colorbar_name[4]),
        6: str(colorbar_name[6]),
        8: str(colorbar_name[9]),
    }
    color_bar = ColorBar(
        color_mapper=color_mapper,
        label_standoff=8,
        width=20,
        height=420,
        border_line_color=None,
        orientation="vertical",
        location=(0, 0),
        major_label_overrides=tick_labels,
        major_tick_line_alpha=0.25,
    )

    # Create figure object.
    p = figure(
        height=512,
        width=768,
        toolbar_location="above",
        tools="box_zoom, reset",
    )

    # add title or stuff
    p.add_layout(
        Title(text=descip, text_font_style="italic", text_font_size="9pt"), "above"
    )
    p.add_layout(Title(text=title, text_font_size="11pt"), "above")

    # author = """Created by HOIST"""
    # p.add_layout(
    #     Title(text=author, text_font_style="italic", text_font_size="9pt"), "below"
    # )

    # no line color
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    # Add patch renderer to figure.

    # add states, nan_states and state line patch
    states = p.patches(
        "xs",
        "ys",
        source=geosource,
        fill_color={"field": field, "transform": color_mapper},
        line_color="gray",
        line_width=0.25,
        fill_alpha=1,
    )
    nan_states = p.patches(
        "xs",
        "ys",
        source=nan_geosource,
        fill_color="#F5EFE6",
        line_color="gray",
        line_width=0.25,
        fill_alpha=1,
    )

    state_line = p.multi_line(
        "xs", "ys", source=state_geosource, line_color="black", line_width=0.5
    )

    # add hover tool
    TOOLTIPS = f"""
        <div style="background-color:#F5F5F5; opacity: 0.95;">
            <div style = "text-align:center;">
                <span style="font-size: 12px; font-weight: bold;">@NAME
            </div>
            <div>
                <img
                    src="@cases_file_location" height="280" alt="@cases_file_location" width="350"
                    style="float: center; margin: 1px 1px 1px 1px; opacity: 0.95;"
                    border="0"
                ></img>
            </div>
            <div style = "text-align:center;">
                <span style="font-size: 12px; font-weight: bold">Cases: @slider &nbsp &nbsp 
                MAE: @mae_label</span>
            </div>
        </div>
    """
    p.add_tools(HoverTool(renderers=[states], tooltips=TOOLTIPS))

    #### Some features to make it a bit nicer.
    p.axis.visible = False
    p.background_fill_color = "grey"
    p.background_fill_alpha = 0.25

    p.border_fill_color = "#F5F5F5"
    color_bar.background_fill_color = "#F5F5F5"
    p.toolbar.autohide = False
    p.add_layout(color_bar, "right")

    return p


""" get data (fill my own data) """
df = get_data()

""" config states and counties """
us_map, state_map, land_map, lake_map = get_us_map(df)

""" Cases color_bar """
(
    us_map,
    state_map,
    q_decomp,
    q_decomp2,
    q_cases,
    q_ratio,
    q_mape,
    q_margin,
    q_cost,
) = config_colorbar_range(us_map, state_map, land_map, lake_map)

print("finished configuration")


""" start drawing"""
state_geosource = GeoJSONDataSource(geojson=state_map.to_json())
nan_map = us_map[us_map["cases_label"] == "N/A"]
notnan_map = us_map[us_map["cases_label"] != "N/A"]

geosource = GeoJSONDataSource(geojson=notnan_map.to_json())
nan_geosource = GeoJSONDataSource(geojson=nan_map.to_json())
palette = brewer["RdBu"][10]

# https://docs.bokeh.org/en/latest/docs/reference/palettes.html
color_mapper = LinearColorMapper(palette=palette, low=0, high=9)

title = "First Rank Decomposition of the Tesnor"
descip = "just a test."
p = draw_map(q_decomp, "q_decomp", "decomp_label", title, descip)

title = "Second Rank Decomposition of the Tesnor"
descip = "just a test."
p2 = draw_map(q_decomp2, "q_decomp2", "decomp2_label", title, descip)

title = "Test slider bar"
descip = "test"
p3 = draw_map(q_margin, "q_margin", "slider", title, descip)

# %%
# div = Div(
#     text="""<b>Explore the full county-level results table at <a target="_blank" href="https://docs.google.com/spreadsheets/d/11-RqEKaZihnABpKk1ozPCBuFOWpcFDdL4hJAzmZr9pM/edit?usp=sharing">Here</a>.</b>"""
# )

div = Div(text="""<b>Add something here later.</b>""")

tab1 = TabPanel(child=column(p, div), title="Decomposition Score 1")
tab2 = TabPanel(child=column(p2, div), title="Decomposition Score 2")

##############
t_slider = Slider(start=1, end=12, value=1, step=1, title="Months of 2022")

callback = CustomJS(
    args=dict(
        source=geosource,
        t_slider=t_slider,
    ),
    code="""
    const data = source.data;
    const new_year = t_slider.value;
    const x = data['slider']
    const y = data['cases_' + new_year.toString()]
    for (let i = 0; i < x.length; i++) {
        x[i] = y[i];
    }
    source.change.emit();
""",
)
t_slider.js_on_change("value", callback)
##################

tab3 = TabPanel(child=column(p3, t_slider, div), title="Time-dependent Patterns")
tabs = Tabs(tabs=[tab1, tab2, tab3], align="center")

file_path = os.getcwd()

doc_path = file_path

outfp = doc_path + "/index.html"

# Save the map
save(tabs, outfp, title="HOIST Visualizations")

# Not sure if this is important, but seemed to start working once
# I ran it
html = file_html(tabs, CDN, outfp)
