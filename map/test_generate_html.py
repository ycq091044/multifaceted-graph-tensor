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

from bokeh.models import (
    ColorBar,
    GeoJSONDataSource,
    HoverTool,
    LinearColorMapper,
)
from bokeh.palettes import brewer
from bokeh.plotting import figure, save
from bokeh.models import Title
from bokeh.models import TabPanel, Tabs, Div
from bokeh.layouts import column, row
from bokeh.models import CustomJS, Slider

from bokeh.resources import CDN
from bokeh.embed import file_html

from utils import get_data, get_us_map, config_colorbar_range


def draw_map(
    geosource,
    nan_geosource,
    colorbar_name,
    field,
    value_label,
    value_name,
    title,
    descip,
):
    # create colorbar
    tick_labels = {idx: str(item) for idx, item in enumerate(colorbar_name)}
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
    p.add_layout(Title(text=title, text_font_size="11pt", align="center"), "above")

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
    _ = nan_states

    state_line = p.multi_line(
        "xs", "ys", source=state_geosource, line_color="black", line_width=0.5
    )
    _ = state_line

    # add hover tool
    TOOLTIPS = f"""
        <div style="background-color:#F5F5F5; opacity: 0.95;">
            <div style = "text-align:center;">
                <span style="font-size: 12px; font-weight: bold;">@NAME
            </div>
            <div style = "text-align:center;">
                <span style="font-size: 11px;">{value_label}: @{value_name}</span>
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


root = "/home/chaoqiy2/data/MNIST/Spatio-temporal/"
DATES = pickle.load(open(os.path.join(root, "feat_name.pkl"), "rb"))["date"]
Interval = 28

if __name__ == "__main__":
    """get data (fill my own data)"""
    df, DATE = get_data()

    """ config states and counties """
    us_map, state_map, land_map, lake_map = get_us_map(df, DATE)

    """ Cases color_bar """
    (
        us_map,
        state_map,
        q_demo,
        q_vac_12,
        q_vac_bst,
    ) = config_colorbar_range(us_map, DATE, state_map, land_map, lake_map)

    print("finished configuration")

    """ Figure 1 """
    state_geosource = GeoJSONDataSource(geojson=state_map.to_json())
    nan_map = us_map[us_map["demo_label"] == "N/A"]
    notnan_map = us_map[us_map["demo_label"] != "N/A"]

    geosource = GeoJSONDataSource(geojson=notnan_map.to_json())
    nan_geosource = GeoJSONDataSource(geojson=nan_map.to_json())
    palette = brewer["RdBu"][10]

    # https://docs.bokeh.org/en/latest/docs/reference/palettes.html
    color_mapper = LinearColorMapper(palette=palette, low=0, high=9)
    title = "County Demographics"
    descip = "Color is coded by the principle tensor factor of the county demongraphics (population, income, etc)."
    p_demo = draw_map(
        geosource,
        nan_geosource,
        q_demo,
        "q_demo",
        "Population",
        "demo_label",
        title,
        descip,
    )

    """ Figure 2 """
    # state_geosource = GeoJSONDataSource(geojson=state_map.to_json())
    # nan_map = us_map[us_map["vac_label_12_0"] == "N/A"]
    # notnan_map = us_map[us_map["vac_label_12_0"] != "N/A"]

    # geosource_vac_12 = GeoJSONDataSource(geojson=notnan_map.to_json())
    # nan_geosource_vac_12 = GeoJSONDataSource(geojson=nan_map.to_json())
    palette = brewer["RdBu"][10]

    # https://docs.bokeh.org/en/latest/docs/reference/palettes.html
    color_mapper = LinearColorMapper(palette=palette, low=0, high=9)
    title = "County Vaccinations of 1st and 2nd Shots"
    descip = "Color is aligned with the number of 1st/2nd shots."
    p_vac_12 = draw_map(
        geosource,
        nan_geosource,
        q_vac_12,
        "q_vac_12_0",
        "1st/2nd Shots",
        "vac_label_12_0",
        title,
        descip,
    )

    vac_12_slider = Slider(
        start=1, end=len(DATES) // Interval, value=1, step=1, title="", show_value=False
    )
    slider_name_vac_12 = Div(
        text="Date Range (28 days): <b>"
        + str(DATES[Interval * vac_12_slider.value])
        + " . . . "
        + str(DATES[Interval * vac_12_slider.value + Interval - 1])
        + "</b>",
        render_as_text=False,
        width=575,
    )
    callback_vac_12 = CustomJS(
        args=dict(
            source=geosource,
            slider=vac_12_slider,
            div=slider_name_vac_12,
            DATES=DATES,
            Interval=Interval,
        ),
        code="""
        const data = source.data;
        const new_time = slider.value;
        const x1 = data['q_vac_12_0'];
        const y1 = data['q_vac_12_' + new_time.toString()];
        for (let i = 0; i < x1.length; i++) {
            x1[i] = y1[i];
        }
        const x2 = data['vac_label_12_0'];
        const y2 = data['vac_label_12_' + new_time.toString()];
        for (let i = 0; i < x2.length; i++) {
            x2[i] = y2[i];
        }
        source.change.emit();
        div.text="Date Range (28 days): <b>" + DATES[Interval * new_time] + " . . . " + DATES[Interval * new_time + Interval-1] + "</b>";
    """,
    )
    vac_12_slider.js_on_change(
        "value",
        callback_vac_12,
    )

    """ Figure 3 """
    # state_geosource = GeoJSONDataSource(geojson=state_map.to_json())
    # nan_map = us_map[us_map["vac_label_bst_0"] == "N/A"]
    # notnan_map = us_map[us_map["vac_label_bst_0"] != "N/A"]

    # geosource_vac_bst = GeoJSONDataSource(geojson=notnan_map.to_json())
    # nan_geosource_vac_bst = GeoJSONDataSource(geojson=nan_map.to_json())
    palette = brewer["RdBu"][10]

    # https://docs.bokeh.org/en/latest/docs/reference/palettes.html
    color_mapper = LinearColorMapper(palette=palette, low=0, high=9)
    title = "County Vaccinations of Booster Shots"
    descip = "Color is aligned with the number of booster shots."
    p_vac_bst = draw_map(
        geosource,
        nan_geosource,
        q_vac_bst,
        "q_vac_bst_0",
        "Booster Shots",
        "vac_label_bst_0",
        title,
        descip,
    )

    vac_bst_slider = Slider(
        start=1, end=len(DATES) // Interval, value=1, step=1, title="", show_value=False
    )
    slider_name_vac_bst = Div(
        text="Date Range (28 days): <b>"
        + str(DATES[Interval * vac_bst_slider.value])
        + " . . . "
        + str(DATES[Interval * vac_bst_slider.value + Interval - 1])
        + "</b>",
        render_as_text=False,
        width=575,
    )
    callback_vac_bst = CustomJS(
        args=dict(
            source=geosource,
            slider=vac_bst_slider,
            div=slider_name_vac_bst,
            DATES=DATES,
            Interval=Interval,
        ),
        code="""
        const data = source.data;
        const new_time = slider.value;
        const x1 = data['q_vac_bst_0'];
        const y1 = data['q_vac_bst_' + new_time.toString()];
        for (let i = 0; i < x1.length; i++) {
            x1[i] = y1[i];
        }
        const x2 = data['vac_label_bst_0'];
        const y2 = data['vac_label_bst_' + new_time.toString()];
        for (let i = 0; i < x2.length; i++) {
            x2[i] = y2[i];
        }
        source.change.emit();
        div.text="Date Range (28 days): <b>" + DATES[Interval * new_time] + " . . . " + DATES[Interval * new_time + Interval-1] + "</b>";
    """,
    )
    vac_bst_slider.js_on_change(
        "value",
        callback_vac_bst,
    )

    """ demo vs vac_12 """
    div = Div(text="""<b>Add something here later.</b>""")
    tab1 = TabPanel(
        child=column(
            row(p_demo, column(p_vac_12, slider_name_vac_12, vac_12_slider)), div
        ),
        title="Demographs vs 1st/2nd Shots",
    )

    """ demo vs vac_bst """
    div = Div(text="""<b>Add something here later.</b>""")
    tab2 = TabPanel(
        child=column(
            row(p_demo, column(p_vac_bst, slider_name_vac_bst, vac_bst_slider)), div
        ),
        title="Demographs vs Booster Shots",
    )

    tabs = Tabs(tabs=[tab1, tab2], align="center")

    """ output the map """
    file_path = os.getcwd()
    doc_path = file_path
    outfp = doc_path + "/index.html"

    # Save the map
    save(tabs, outfp, title="Bronchiolitis Maps")
    # Not sure if this is important, but seemed to start working once
    # I ran it
    html = file_html(tabs, CDN, outfp)
