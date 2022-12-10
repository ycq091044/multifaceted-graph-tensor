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

    # Figure 1 data
    demo_score, demo_num = pickle.load(open(f"../demongraphs.pkl", "rb"))
    vac_score, vac_12, vac_bst = pickle.load(open(f"../vaccine.pkl", "rb"))
    date = len(vac_12)

    df_dict = {
        "county": [],
        "state": [],
        "fips": [],
        "demo_score": [],
        "demo_num": [],
        # "vac_score_12": [],
        # "vac_num_12": [],
        # "vac_score_bst": [],
        # "vac_num_bst": [],
    }
    df_dict.update({f"vac_score_12_{i}": [] for i in range(date)})
    df_dict.update({f"vac_num_12_{i}": [] for i in range(date)})
    df_dict.update({f"vac_score_bst_{i}": [] for i in range(date)})
    df_dict.update({f"vac_num_bst_{i}": [] for i in range(date)})

    for i in range(len(fips)):
        if fips[i] in fips2name:
            cur_name = fips2name[fips[i]].split(",")
            # figure 1
            df_dict["demo_score"].append(demo_score[i, 0])
            df_dict["demo_num"].append(demo_num[i])
            for j in range(date):
                df_dict[f"vac_score_12_{j}"].append(vac_score[j][i, 0])
                df_dict[f"vac_num_12_{j}"].append(vac_12[j][i])
                df_dict[f"vac_score_bst_{j}"].append(vac_score[j][i, 1])
                df_dict[f"vac_num_bst_{j}"].append(vac_bst[j][i])
            # df_dict["vac_score_12"].append(vac_score[i, 0])
            # df_dict["vac_num_12"].append(vac_12[i])
            # df_dict["vac_score_bst"].append(vac_score[i, 1])
            # df_dict["vac_num_bst"].append(vac_bst[i])
            # others
            df_dict["county"].append(cur_name[0])
            df_dict["state"].append(cur_name[1])
            df_dict["fips"].append(fips[i])
    df = pd.DataFrame(df_dict)

    # change into [0, infty)
    df["demo_score"] = df["demo_score"].apply(lambda x: np.log(x))
    df["demo_score"] = df["demo_score"] - df["demo_score"].min()
    for j in range(date):
        df[f"vac_score_12_{j}"] = df[f"vac_score_12_{j}"].apply(lambda x: np.log(x))
        df[f"vac_score_12_{j}"] = (
            df[f"vac_score_12_{j}"] - df[f"vac_score_12_{j}"].min()
        ) / (df[f"vac_score_12_{j}"].max() - df[f"vac_score_12_{j}"].min() + 1e-8)

        df[f"vac_score_bst_{j}"] = df[f"vac_score_bst_{j}"].apply(lambda x: np.log(x))
        df[f"vac_score_bst_{j}"] = (
            df[f"vac_score_bst_{j}"] - df[f"vac_score_bst_{j}"].min()
        ) / (df[f"vac_score_bst_{j}"].max() - df[f"vac_score_bst_{j}"].min() + 1e-8)
    return df, date


def get_us_map(df, date):
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

    # figure 1
    us_map["demo_label"] = us_map["demo_num"]
    us_map["demo_label"] = us_map["demo_label"].map("{:.0f}".format)
    for j in range(date):
        us_map[f"vac_label_12_{j}"] = us_map[f"vac_num_12_{j}"]
        us_map[f"vac_label_12_{j}"] = us_map[f"vac_label_12_{j}"].map("{:.0f}".format)
        us_map[f"vac_label_bst_{j}"] = us_map[f"vac_num_bst_{j}"]
        us_map[f"vac_label_bst_{j}"] = us_map[f"vac_label_bst_{j}"].map("{:.0f}".format)

    return us_map, state_map, land_map, lake_map


def config_colorbar_range(us_map, date, state_map, land_map, lake_map):
    # figure 1
    q_demo = [
        np.round(item, 2) for item in np.linspace(0, us_map["demo_score"].max(), 10)
    ] + [np.inf]
    us_map["q_demo"] = pd.cut(us_map["demo_score"], q_demo, labels=range(0, 10))
    us_map["q_demo"].fillna(0, inplace=True)

    MAX_vac_12 = max([us_map[f"vac_score_12_{j}"].max() for j in range(date)])
    q_vac_12 = [np.round(item, 2) for item in np.linspace(0, MAX_vac_12, 10)] + [np.inf]
    for j in range(date):
        us_map[f"q_vac_12_{j}"] = pd.cut(
            us_map[f"vac_score_12_{j}"], q_vac_12, labels=range(0, 10)
        )
        us_map[f"q_vac_12_{j}"].fillna(0, inplace=True)
        # drop columns
        us_map.drop(columns=[f"vac_num_12_{j}", f"vac_score_12_{j}"], inplace=True)

    MAX_vac_bst = max([us_map[f"vac_score_bst_{j}"].max() for j in range(date)])
    q_vac_bst = [np.round(item, 2) for item in np.linspace(0, MAX_vac_bst, 10)] + [
        np.inf
    ]
    for j in range(date):
        us_map[f"q_vac_bst_{j}"] = pd.cut(
            us_map[f"vac_score_bst_{j}"], q_vac_bst, labels=range(0, 10)
        )
        us_map[f"q_vac_bst_{j}"].fillna(0, inplace=True)
        # drop columns
        us_map.drop(columns=[f"vac_num_bst_{j}", f"vac_score_bst_{j}"], inplace=True)

    # figure 1
    us_map["demo_label"].replace("nan", "N/A", inplace=True)
    for j in range(date):
        us_map[f"vac_label_12_{j}"].replace("nan", "N/A", inplace=True)
        us_map[f"vac_label_bst_{j}"].replace("nan", "N/A", inplace=True)

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
        q_demo,
        q_vac_12,
        q_vac_bst,
    )


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
