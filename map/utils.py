from logging.config import valid_ident
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy import stats
import os
import geopandas as gpd
import pandas as pd
import pickle

root = "/home/chaoqiy2/data/MNIST/Spatio-temporal"


def get_data():
    fips2name = pickle.load(open(f"{root}/fips2name.pkl", "rb"))
    fips2state = pickle.load(open(f"{root}/fips2state.pkl", "rb"))
    fips = pickle.load(open(f"{root}/fips.pkl", "rb"))
    decomp = pickle.load(open(f"../decomp.pkl", "rb"))

    df_dict = {
        "county": [],
        "state": [],
        "fips": [],
        "decomp": [],
        "decomp2": [],
        "cases": [],
        "slider": [],
        "cases_1": [],
        "cases_2": [],
        "cases_3": [],
        "cases_4": [],
        "mae": [],
        "mape": [],
        "ratio": [],
        "margin": [],
        "hos_nnt": [],
        "death_nnt": [],
        "cost": [],
        "cases_file_location": [],
        "ratio_file_location": [],
        "sim_file_location": [],
    }
    for i in range(len(fips)):
        if fips[i] in fips2name:
            cur_name = fips2name[fips[i]].split(",")
            df_dict["county"].append(cur_name[0])
            df_dict["state"].append(cur_name[1])
            df_dict["fips"].append(fips[i])
            df_dict["decomp"].append(decomp[i, 0])
            df_dict["decomp2"].append(decomp[i, 1])
            df_dict["cases"].append(1)
            df_dict["slider"].append(0)
            df_dict["cases_1"].append(1)
            df_dict["cases_2"].append(10)
            df_dict["cases_3"].append(100)
            df_dict["cases_4"].append(1000)
            df_dict["mae"].append(0.5)
            df_dict["mape"].append(0.5)
            df_dict["ratio"].append(0.5)
            df_dict["margin"].append(10)
            df_dict["hos_nnt"].append(10)
            df_dict["death_nnt"].append(10)
            df_dict["cost"].append(10)
            df_dict["cases_file_location"].append("png1")
            df_dict["ratio_file_location"].append("png2")
            df_dict["sim_file_location"].append("png3")
    df = pd.DataFrame(df_dict)
    return df


def config_colorbar_range(us_map, state_map, land_map, lake_map):
    q_decomp = [
        -np.inf,
        -323.7175900341559,
        -61.94701358172966,
        -11.854258854289872,
        -2.26844653180864,
        -0.43409286201199365,
        -0.08306857562181706,
        -0.015896112444451445,
        -0.0030418998746590143,
        -0.0005821019230791376,
        -0.00011139178237758834,
        -2.1316076599817513e-05,
        0,
    ]
    us_map["q_decomp"] = pd.cut(us_map["decomp"], q_decomp, labels=range(0, 12))
    us_map["q_decomp"].fillna(0, inplace=True)

    q_decomp2 = [
        0,
        8.366516072160513e-06,
        4.171941291326946e-05,
        0.00020803255002531403,
        0.0010373498906893798,
        0.0051727149132769325,
        0.025793602059426626,
        0.1286190942965734,
        0.641355611347234,
        3.1981022915471025,
        15.947249990865679,
        79.52051192370948,
        100,
    ]
    us_map["q_decomp2"] = pd.cut(us_map["decomp2"], q_decomp2, labels=range(0, 12))
    us_map["q_decomp2"].fillna(0, inplace=True)

    q_cases = [0, 1, 5, 10, 100, 250, 500, 1000, 5000, 10000, np.inf]
    us_map["q_cases"] = pd.cut(us_map["cases"], q_cases, labels=range(0, 10))
    us_map["q_cases"].fillna(0, inplace=True)

    q_ratio = [1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 3e-3, 5e-3, 7e-3, 1e-2, 5e-2, 1e-1]
    us_map["q_ratio"] = pd.cut(us_map["ratio"], q_ratio, labels=range(0, 10))
    us_map["q_ratio"].fillna(0, inplace=True)

    q_mape = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    us_map["q_mape"] = pd.cut(us_map["mape"], q_mape, labels=range(0, 10))
    us_map["q_mape"].fillna(0, inplace=True)

    q_margin = [np.inf, 100000, 50000, 10000, 5000, 1000, 500, 300, 100, 50, 10][::-1]
    us_map["q_margin"] = pd.cut(us_map["margin"], q_margin, labels=range(0, 10))
    us_map["q_margin"].fillna(0, inplace=True)

    us_map["q_hosnnt"] = pd.cut(us_map["hos_nnt"], q_margin, labels=range(0, 10))
    us_map["q_hosnnt"].fillna(0, inplace=True)

    us_map["q_deathnnt"] = pd.cut(us_map["death_nnt"], q_margin, labels=range(0, 10))
    us_map["q_deathnnt"].fillna(0, inplace=True)

    q_cost = [0, 0.2, 0.4, 0.6, 0.8, 1, 10, 20, 30, 40, 80]
    us_map["q_cost"] = pd.cut(us_map["cost"], q_cost, labels=range(0, 10))
    us_map["q_cost"].fillna(0, inplace=True)

    us_map["decomp"].replace("nan", "N/A", inplace=True)
    us_map["decomp2"].replace("nan", "N/A", inplace=True)
    us_map["cases_label"].replace("nan", "N/A", inplace=True)
    us_map["mae_label"].replace("nan", "N/A", inplace=True)
    us_map["ratio_label"].replace("nan", "N/A", inplace=True)
    us_map["margin_label"].replace("nan", ">100,000", inplace=True)
    us_map["deathnnt_label"].replace("nan", ">100,000", inplace=True)
    us_map["hosnnt_label"].replace("nan", ">100,000", inplace=True)
    us_map["cost_label"].replace("nan", "N/A", inplace=True)
    us_map["mape_label"].replace("nan", "N/A", inplace=True)

    us_map = gpd.overlay(us_map, land_map, how="intersection")
    great_lakes = [
        "Lake Superior",
        "Lake Michigan",
        "Lake Erie",
        "Lake Superior" "Lake Huron",
    ]
    us_map = gpd.overlay(
        us_map, lake_map[lake_map.name.isin(great_lakes)], how="difference"
    )
    state_map = gpd.overlay(state_map, land_map, how="intersection")

    state_map = gpd.overlay(
        state_map, lake_map[lake_map.name.isin(great_lakes)], how="difference"
    )

    return (
        us_map,
        state_map,
        q_decomp,
        q_decomp2,
        q_cases,
        q_ratio,
        q_mape,
        q_margin,
        q_cost,
    )


def get_us_map(df):
    # TODO: what is this
    map_projection = "epsg:2163"
    cwd = os.getcwd()

    # lake shape
    lake_shape = cwd + "/shapefiles/lake/ne_10m_lakes.shx"
    lake_map = gpd.read_file(lake_shape)
    lake_map = lake_map.to_crs({"init": map_projection})

    # land shape
    land_shape = cwd + "/shapefiles/land/ne_50m_land.shx"
    land_map = gpd.read_file(land_shape)
    land_map = land_map.to_crs({"init": map_projection})
    land_map = land_map.iloc[0:1200]

    # county shape
    county_shape = cwd + "/shapefiles/county/tl_2017_us_county.shx"
    us_map = gpd.read_file(county_shape)
    us_map = us_map.to_crs({"init": map_projection})
    us_map["geometry"] = us_map["geometry"].simplify(2000)
    us_map["area_fips"] = (
        us_map.STATEFP.astype(str) + us_map.COUNTYFP.astype(str)
    ).astype(int)

    # county merge with conty-level statistics
    us_map = us_map.merge(
        df, left_on="area_fips", right_on="fips", how="left", indicator=True
    )
    us_map.set_index("STATEFP", inplace=True)

    # TODO: why drop these rows?
    drop_list = [
        "02",
        "15",
        "72",
        "78",
        "69",
        "66",
        "60",
    ]
    us_map.drop(drop_list, inplace=True)

    # state shape
    state_shape = cwd + "/shapefiles/state/tl_2017_us_state.shx"
    state_map = gpd.read_file(state_shape)
    state_map = state_map.to_crs({"init": map_projection})
    state_map["geometry"] = state_map["geometry"].simplify(200)
    state_fp_dict = dict(zip(state_map.STATEFP, state_map.STUSPS))
    state_map.set_index("STATEFP", inplace=True)

    # TODO: why drop these rows?
    drop_list = [
        "02",
        "15",
        "72",
        "78",
        "69",
        "66",
        "60",
    ]
    state_map.drop(drop_list, inplace=True)

    us_map.reset_index(inplace=True)
    us_map["STSPS"] = us_map["STATEFP"].map(state_fp_dict)
    us_map["NAME"] = us_map["NAME"] + " County, " + us_map["STSPS"]
    us_map.set_index("STATEFP", inplace=True)

    us_map["cases_label"] = us_map["cases"].round(0)
    us_map["cases_label"] = us_map["cases_label"].map("{:,.0f}".format)

    us_map["decomp_label"] = us_map["decomp"].map("{:,.2f}".format)
    us_map["decomp2_label"] = us_map["decomp2"].map("{:,.2f}".format)

    us_map["mae_label"] = us_map["mae"].round(1)
    us_map["mae_label"] = us_map["mae_label"].map("{:,.1f}".format)

    us_map["mape_label"] = us_map["mape"].round(2)
    us_map.loc[us_map["mape_label"] == 0, "mape_label"] = np.nan
    us_map["mape_label"] = us_map["mape_label"].map("{:,.2f}".format)

    us_map["ratio_label"] = us_map["ratio"].round(4)

    us_map["ratio_label"] = us_map["ratio_label"].map("{:,.4f}".format)

    us_map["margin_label"] = us_map["margin"].round(0)
    us_map.loc[us_map["margin_label"] > 100000, "margin_label"] = np.nan
    us_map["margin_label"] = us_map["margin_label"].map("{:,.0f}".format)

    us_map["hosnnt_label"] = us_map["hos_nnt"].round(0)
    us_map.loc[us_map["hosnnt_label"] > 100000, "hosnnt_label"] = np.nan
    us_map["hosnnt_label"] = us_map["hosnnt_label"].map("{:,.0f}".format)

    us_map["deathnnt_label"] = us_map["death_nnt"].round(0)
    us_map.loc[us_map["deathnnt_label"] > 100000, "deathnnt_label"] = np.nan
    us_map["deathnnt_label"] = us_map["deathnnt_label"].map("{:,.0f}".format)

    us_map["cost_label"] = us_map["cost"].round(2)
    us_map["cost_label"] = us_map["cost_label"].map("{:,.2f}".format)

    return us_map, state_map, land_map, lake_map


def get_dist(lat1, lon1, lat2, lon2):
    R = 6371e3
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d


def generate_series(data, y, window_size, pred_size, date=False):
    """
    Data: N*T*F
    y: N*T
    """
    series = []
    targets = []
    idx = window_size
    while idx + pred_size < data.shape[1]:
        if date:
            series.append(data[:, idx - 1, :])
        else:
            series.append(np.sum(data[:, idx - window_size : idx, :], axis=1))
        targets.append(np.sum(y[:, idx : idx + pred_size], axis=1))
        idx += pred_size
    return np.array(series).transpose(1, 0, 2), np.array(targets).transpose(1, 0)


def temporal_split(
    x, y, static, mats, val_ratio, test_ratio, norm="z-score", norm_mat=True
):
    seq_len = x.shape[1]
    test_len = int(seq_len * test_ratio)
    val_len = int(seq_len * val_ratio)
    train_len = seq_len - val_len - test_len

    shuffle_idx = np.arange(x.shape[0])
    np.random.shuffle(shuffle_idx)
    x = x[shuffle_idx, :, :]
    y = y[shuffle_idx, :]
    static = static[shuffle_idx, :]
    for i in range(len(mats)):
        mats[i] = mats[i][shuffle_idx, :][:, shuffle_idx]

    idx = np.arange(x.shape[1])
    train_x = x[:, :train_len, :]
    train_y = y[:, :train_len]
    train_idx = idx[:train_len]
    val_x = x[:, train_len : train_len + val_len, :]
    val_y = y[:, train_len : train_len + val_len]
    val_idx = idx[train_len : train_len + val_len]
    test_x = x[:, train_len + val_len :, :]
    test_y = y[:, train_len + val_len :]
    test_idx = idx[train_len + val_len :]

    normalize_dict = {}
    for i in range(x.shape[2]):
        normalize_dict[i] = [
            np.mean(train_x[:, :, i]),
            np.std(train_x[:, :, i]),
            np.min(train_x[:, :, i]),
            np.max(train_x[:, :, i]),
        ]
    normalize_dict["y"] = [
        np.mean(train_y[:, :]),
        np.std(train_y[:, :]),
        np.min(train_y[:, :]),
        np.max(train_y[:, :]),
    ]
    normalize_dict["static"] = [
        np.mean(static, axis=0),
        np.std(static, axis=0),
        np.min(static, axis=0),
        np.max(static, axis=0),
    ]
    for i in range(len(mats)):
        normalize_dict["mat_%d" % i] = [
            np.mean(mats[i]),
            np.std(mats[i]),
            np.min(mats[i]),
            np.max(mats[i]),
        ]

    for i in range(train_x.shape[2]):
        if (normalize_dict[i][3] - normalize_dict[i][2]) == 0:
            train_x[:, :, i] = 0
            val_x[:, :, i] = 0
            test_x[:, :, i] = 0
        else:
            train_x[:, :, i] = (train_x[:, :, i] - normalize_dict[i][2]) / (
                normalize_dict[i][3] - normalize_dict[i][2]
            )
            val_x[:, :, i] = (val_x[:, :, i] - normalize_dict[i][2]) / (
                normalize_dict[i][3] - normalize_dict[i][2]
            )
            test_x[:, :, i] = (test_x[:, :, i] - normalize_dict[i][2]) / (
                normalize_dict[i][3] - normalize_dict[i][2]
            )
    train_y = (train_y - normalize_dict["y"][0]) / normalize_dict["y"][1]
    val_y = (val_y - normalize_dict["y"][0]) / normalize_dict["y"][1]
    test_y = (test_y - normalize_dict["y"][0]) / normalize_dict["y"][1]
    static = (static - normalize_dict["static"][2]) / (
        normalize_dict["static"][3] - normalize_dict["static"][2]
    )
    if norm_mat:
        for i in range(len(mats)):
            mats[i] = (mats[i] - normalize_dict["mat_%d" % i][2]) / (
                normalize_dict["mat_%d" % i][3] - normalize_dict["mat_%d" % i][2]
            )

    return (
        train_x,
        val_x,
        test_x,
        train_y,
        val_y,
        test_y,
        train_idx,
        val_idx,
        test_idx,
        static,
        mats,
        normalize_dict,
        shuffle_idx,
    )


def mse(y_true, y_pred, std=False):
    if std:
        return (
            np.mean((y_true - y_pred) ** 2, axis=1).mean(),
            np.mean((y_true - y_pred) ** 2, axis=1).std(),
        )
    else:
        return np.mean((y_true - y_pred) ** 2, axis=1).mean()


def mae(y_true, y_pred, std=False):
    if std:
        return (
            np.mean(np.abs(y_true - y_pred), axis=1).mean(),
            np.mean(np.abs(y_true - y_pred), axis=1).std(),
        )
    else:
        return np.mean(np.abs(y_true - y_pred), axis=1).mean()


def r2(y_true, y_pred, std=False):
    return r2_score(y_true, y_pred)


def ccc(y_true, y_pred, std=False):
    return stats.pearsonr(y_true.flatten(), y_pred.flatten())[0]
