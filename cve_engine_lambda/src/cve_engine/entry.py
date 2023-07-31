from dataclasses import dataclass
import logging
import torch
import time
from datetime import datetime
from pprint import pprint

from cve_engine.cvss import CVSS_BASE_METRICS
from cve_engine.engine import CVEEngineModel, desc_preprocess

logging.basicConfig(
    format="[%(levelname)-8s] (%(name)s) %(message)s",
    level=logging.DEBUG,
)

log = logging.getLogger(__name__)


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
    cvem.display_parameters()

    return {
        "status": 201,
        "message": "hello!",
        "start_timestamp": _start_timestamp.isoformat(),
        "exec_time_secs": round(time.perf_counter() - _start, 3),
    }


# for testing
def main():
    @dataclass
    class Ctx:
        aws_request_id: int
        log_stream_name: str

    res = handler(
        {
            "cve_description": "desc description a with person persons HELLO an the 1 2 3",
        },
        Ctx(0, "test"),
    )
    pprint(res)


if __name__ == "__main__":
    main()
