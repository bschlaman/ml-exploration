from dataclasses import dataclass
import logging
import torch
import time
from datetime import datetime
import numpy as np
import cvss
from prettytable import PrettyTable

from bpyutils.formatting.colors import bld, yel
from cve_engine.engine import CVEEngineModel
from cve_engine.data_processing import desc_preprocess
from cve_engine.cvss_data import CVSS_VERSION_STR, CVSS_BASE_METRICS

logging.basicConfig(
    format="[%(levelname)-8s] (%(name)s) %(message)s",
    level=logging.DEBUG,
)

log = logging.getLogger(__name__)


def construct_cvss_vector(pred: np.ndarray) -> str:
    elements = [CVSS_VERSION_STR]
    for i, (metric, meta) in enumerate(CVSS_BASE_METRICS.items()):
        elements.append(metric + ":" + meta.categories[int(pred[i])][0])
    return cvss.CVSS3("/".join(elements)).clean_vector()


def construct_predictions_dict(pred: np.ndarray, conf: np.ndarray) -> dict[str, dict]:
    preds = {}
    for i, (metric, meta) in enumerate(CVSS_BASE_METRICS.items()):
        preds[metric] = {
            "prediction": meta.categories[int(pred[i])],
            "confidence_interval": conf[i],
        }
    return preds


def pprint_lambda_res(res: dict):
    pt = PrettyTable()
    pt.field_names = ["Attribute", "Value"]
    pt.align = "l"

    for key, val in res.items():
        if key == "metric_predictions":
            continue
        pt.add_row((key, val))

    print(pt)

    pt = PrettyTable()
    pt.field_names = ["Metric", "Predicted Value", "Confidence"]
    pt.align = "l"

    for key, val in res["metric_predictions"].items():
        pt.add_row(
            (
                bld(CVSS_BASE_METRICS[key].name),
                yel(val["prediction"]),
                round(float(val["confidence_interval"]), 4),
            )
        )

    print(pt)


def handler(event: dict, context):
    _start = time.perf_counter()
    _start_timestamp = datetime.now()

    log.debug(f"context.aws_request_id: {context.aws_request_id}")
    log.debug(f"context.log_stream_name: {context.log_stream_name}")
    log.debug(f"event: {event}")

    log.info("Torch version: " + torch.__version__)

    cve_desc = event["cve_description"]

    log.info(desc_preprocess(cve_desc))

    cvem = CVEEngineModel()
    cvem.load_latest_models()
    cvem.display_parameters()

    x = np.array([desc_preprocess(cve_desc)])
    predictions, confidence_scores = cvem.predict(x)
    # we only used 1 desc, so take the first
    pred, conf = predictions[0], confidence_scores[0]
    print(predictions, confidence_scores)

    return {
        "start_timestamp": _start_timestamp.isoformat(),
        "exec_time_secs": round(time.perf_counter() - _start, 3),
        "cvss_version": CVSS_VERSION_STR,
        "vector_combined_prediction": construct_cvss_vector(pred),
        "average_confidence": conf.mean(),
        "metric_predictions": construct_predictions_dict(pred, conf),
    }


# for testing
def main():
    @dataclass
    class Ctx:
        aws_request_id: int
        log_stream_name: str

    res = handler(
        {
            "cve_description": "this attack is based on the network, it is very complex but poses no threat to availability.  oh no, secret data is a big problem, once an attacker gains access then can do wild things with php",
        },
        Ctx(0, "test"),
    )
    pprint_lambda_res(res)


if __name__ == "__main__":
    main()
