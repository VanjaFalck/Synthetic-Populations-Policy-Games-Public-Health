# -*- coding: utf-8 -*-
"""
Utilities has a set of helper functions for the
Synthetic Population Generations for Policy Games
in Public Health
"""
import torch
import pandas as pd


# Get prediction from a trained torch model on a dataset df
# Returns a pandas dataframe with predictions
def get_predictions(torch_model, df):
    torch_data = torch.tensor(df.values, dtype=torch.float32)
    with torch.no_grad():
        z, logits = torch_model(torch_data)
        predictions = logits.max(dim=1)[1]
    torch_data = torch_data.detach()
    return pd.DataFrame(predictions)


# Returns the ranked list of cluster-id (numbers) from a dataframe with predictons
def highest_cluster(df_preds):
    cluster = []
    cluster_list = df_preds.value_counts().index.tolist()
    for value in cluster_list:
        cluster.append(value[0])
    return cluster


# Returns dataset which matches cluster only
# Default level=0 selects the most frequent cluster
def get_cluster_data(df, df_predictions, level=0):
    df = df.copy()
    df["Cat"] = df_predictions
    categories = highest_cluster(df_predictions)
    if len(categories) > level:
        df = df[df["Cat"] == categories[level]]
        return df.iloc[:, :-1]  # skip column "Cat"
    else:
        print("Selected level is higher than the number of available categories!")
        return None


def get_pca_data(mod, dta, level=0):
    """
    Parameters:
    mod ...... the pca trained model
    dta ...... a pandas dataframe to extract components from
    Returns:
    A pandas dataframe with the selected level (component from pca model)
    """
    comp = mod.transform(dta).argmax(axis=1)
    dta_c = dta.copy()
    dta_c["Cat"] = comp
    dt = dta_c[dta_c["Cat"] == level]
    return dt.iloc[:, :-1]  # Skip column "Cat"


def get_cluster_index(model, df_o, df_s, number):
    df_o = df_o.copy()
    df_s = df_s.copy()
    preds_o = get_predictions(model, df_o)
    preds_o = preds_o[preds_o == number].dropna()
    categories_o = highest_cluster(preds_o)
    idx_o = categories_o.index(number)
    preds_s = get_predictions(model, df_s)
    preds_s = preds_s[preds_s == number].dropna()
    categories_s = highest_cluster(preds_s)
    idx_s = categories_s.index(number)
    return idx_o, idx_s, preds_o, preds_s


def get_cluster_by_category_list(model, df_o, df_s, df_sv,
                                 number, categories_o,
                                 categories_s, 
                                 categories_sv):
    df_o = df_o.copy()
    df_s = df_s.copy()
    df_sv = df_sv.copy()
    preds_o = get_predictions(model, df_o)
    idx_o = categories_o.index(number)
    preds_s = get_predictions(model, df_s)
    preds_sv = get_predictions(model, df_sv)
    idx_s = categories_s.index(number)
    idx_sv = categories_sv.index(number)
    # Get corrct cluster data (same cluster-number)
    data_o = get_cluster_data(df_o, preds_o, idx_o)
    data_s = get_cluster_data(df_s, preds_s, idx_s)
    data_sv = get_cluster_data(df_sv, preds_sv, idx_sv)
    return data_o, data_s, data_sv


def get_cluster_data_special(model, df, df_s, number):
    idx_o, idx_s, pred_o, pred_s = get_cluster_index(model, df, df_s, number)
    cluster_o = get_cluster_data(df, pred_o, level=idx_o)
    cluster_s = get_cluster_data(df, pred_s, level=idx_s)
    return cluster_o, cluster_s


def get_synthetic(torch_model, batch_dim, lat_feature):
    noise = torch.rand(batch_dim, lat_feature)
    with torch.no_grad():
        s_data = torch_model(noise)
    s_data = s_data.detach()
    return s_data


def combine_data(df1, df2, columns):
    combine = pd.concat([df1.mean(), df2.mean()], axis=1)
    combine.columns = columns
    return combine


def get_regions(df):
    df_ = df.copy()
    regions = list()
    regions.append(df_[df_["region_1"] == 1])
    regions.append(df_[df_["region_2"] == 1])
    regions.append(df_[df_["region_3"] == 1])
    regions.append(df_[df_["region_4"] == 1])
    regions.append(df_[df_["region_5"] == 1])
    regions.append(df_[df_["region_6"] == 1])
    return regions


def get_PH010(df):
    df_ = df.copy()
    health = list()
    health.append(df_[df_["PH010_1"] == 1])
    health.append(df_[df_["PH010_2"] == 1])
    health.append(df_[df_["PH010_3"] == 1])
    health.append(df_[df_["PH010_4"] == 1])
    health.append(df_[df_["PH010_5"] == 1])
    return health


def get_PE040(df):
    df_ = df.copy()
    edu = list()
    edu .append(df_[df_["PE040_1"] == 1])
    edu .append(df_[df_["PE040_2"] == 1])
    edu .append(df_[df_["PE040_3"] == 1])
    edu .append(df_[df_["PE040_4"] == 1])
    edu .append(df_[df_["PE040_5"] == 1])
    edu .append(df_[df_["PE040_6"] == 1])
    return edu


def check_duplicates(df_):
    df_dup = df_.copy()
    dup_number = len(df_)-len(df_.drop_duplicates())
    df_dup["dup_number"] = df_dup.groupby(df_.columns.tolist()).cumcount()+1
    frequency_list = df_dup["dup_number"].value_counts()
    print("Duplicates: {} \nFrequency:\n{}".format(dup_number, frequency_list))
    # return dup_number, frequency_list
