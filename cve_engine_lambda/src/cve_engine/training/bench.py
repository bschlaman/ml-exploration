"""
This module is a script intended for development purposes only.
"""
import json
import logging
import os
import pickle
import sys

import pandas as pd
import torch
from cve_engine.cvss_data import CVSS_BASE_METRICS
from cve_engine.data_processing import (
    clean_cvss_vector,
    create_bow,
    desc_preprocess,
    vec_parse_metric,
)
from cve_engine.engine import CVEEngineModel
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    format="[%(levelname)-8s] (%(name)s) %(message)s",
    level=logging.DEBUG,
)
log = logging.getLogger(__name__)
logging.getLogger("cve_engine.data_processing").setLevel(logging.INFO)


def load_cves():
    """Loads all cve data, indexed by cve_id"""
    cves = {}
    for subdir in ("2017", "2018", "2019", "2020", "2021", "2022", "2023"):
        path = os.path.join("../data/cve", subdir)
        for file in os.listdir(path):
            with open(os.path.join(path, file)) as f:
                cves[file.removesuffix(".json")] = json.load(f)
    return cves


def construct_training_set(cves: dict):
    """
    Scan through all CVEs for cve.source_data elements.
    For each element, couple the cve.source_data.elem.description
    with each cve.source_data.elem.score.
    """
    examples = []
    for cve_data in cves.values():
        for sd in cve_data["source_data"]:
            if "scores" not in sd:
                continue
            examples.extend(
                [{"description": sd["description"]} | score for score in sd["scores"]]
            )
    return examples


def load_raw_cve_pkl():
    pkl_path = "../cves.pkl"

    if os.path.isfile(pkl_path):
        log.info(f"{pkl_path} exists.  loading from file...")
        with open(pkl_path, "rb") as f:
            cves = pickle.load(f)
    else:
        log.info(f"{pkl_path} not found.  loading data into memory...")
        # can take a few seconds
        cves = load_cves()
        log.info(f"saving data to {pkl_path}")
        with open(pkl_path, "wb") as f:
            pickle.dump(cves, f)
    return cves


def extract_cvss_vector_components(df: pd.DataFrame, vector: pd.Series):
    for metric in CVSS_BASE_METRICS:
        df[metric] = vector.dropna().apply(lambda v: vec_parse_metric(v, metric))
    return df


def construct_clean_dataframe(cves: dict) -> pd.DataFrame:
    df = pd.DataFrame(construct_training_set(cves))

    log.info("cleaning cvss vectors")
    df["vector_clean"] = df["vector"].apply(clean_cvss_vector)
    log.info("processing descriptions")
    df["processed_desc"] = df["description"].apply(desc_preprocess)
    log.info("extracting cvss vector components")
    df = extract_cvss_vector_components(df, df["vector_clean"])

    # only this compact is version is used going forward
    df_clean = df.dropna(subset=["vector_clean"]).copy()

    for metric in CVSS_BASE_METRICS.keys():
        encoder = LabelEncoder()
        df_clean[metric + "_Y"] = encoder.fit_transform(df_clean[metric])

    return df_clean


def main():
    cves = load_raw_cve_pkl()

    print(f"size of cve data: {sys.getsizeof(cves)}")

    df = construct_clean_dataframe(cves)
    log.info("dataframe preparation complete")

    Y_np = df[[metric + "_Y" for metric in CVSS_BASE_METRICS.keys()]].values
    Y = torch.from_numpy(Y_np)

    log.info(f"Y matrix constructed with shape: {Y.shape}")

    # split the data and create Y matrices
    train_split = 0.8
    i = int(train_split * len(Y))
    X_train_raw, X_test_raw = df["processed_desc"][:i], df["processed_desc"][i:]
    Y_train, Y_test = Y[:i], Y[i:]

    # compute X_train_np just so we can examine the shape;
    # the actual X_train will be constructed just before training
    bow_vec, X_train_np = create_bow(X_train_raw.to_list())
    print(f"{X_train_np.shape=}\t{Y_train.shape=}")

    cvem = CVEEngineModel()
    cvem.new_model(bow_vec)
    cvem.display_parameters()

    exit()
    if cvem.ipex_optimized:
        cvem.train_all(X_train_raw.to_numpy(), Y_train.to("xpu"))
    else:
        cvem.train_all(X_train_raw.to_numpy(), Y_train)


if __name__ == "__main__":
    main()
