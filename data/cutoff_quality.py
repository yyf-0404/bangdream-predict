"""Distinguish confirmed empty rankings from failed/invalid data snapshots."""
import hashlib
import json
from pathlib import Path

QUALITY_VERSION = "validated-cutoff-empty-v1"


def cutoff_digest(points):
    encoded = json.dumps(points, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def confirms_structural_zero(report):
    # Legacy merge reports already contain received counts, so old genuine
    # empty data need not be redownloaded. A filtered nonempty response is not
    # proof of insufficient participants, even if its status used to be "ok".
    if report.get("merge", {}).get("merged") != 0:
        return False
    return any(info.get("status") == "ok" and info.get("received") == 0
               for info in report.get("sources", {}).values())


def load_cutoff_snapshot(raw_dir, server, event_id, tier):
    path = Path(raw_dir) / str(server) / str(event_id) / f"{tier}.json"
    report_path = Path(raw_dir) / "_merge_reports" / str(server) / str(event_id) / f"{tier}.json"
    with path.open(encoding="utf-8") as handle:
        points = json.load(handle)
    if not isinstance(points, list):
        raise ValueError(f"Invalid cutoff list: {path}")
    previous_time, previous_ep = -1, -1
    for point in points:
        if not isinstance(point, dict):
            raise ValueError(f"Invalid cutoff point: {path}")
        timestamp, ep = point.get("time"), point.get("ep")
        if (type(timestamp) is not int or timestamp <= previous_time or timestamp <= 0
                or type(ep) is not int or ep < 0 or ep < previous_ep):
            raise ValueError(f"Invalid/non-monotone cutoff point: {path}")
        previous_time, previous_ep = timestamp, ep
    report = {}
    if report_path.exists():
        with report_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        if not isinstance(report, dict):
            raise ValueError(f"Invalid merge report: {report_path}")
        for key, value in (("server", server), ("event_id", event_id), ("tier", tier)):
            if key in report and report[key] != value:
                raise ValueError(f"Merge report identity mismatch: {report_path}")
        if report.get("cutoff_sha256") and report["cutoff_sha256"] != cutoff_digest(points):
            raise ValueError(f"Cutoff/report snapshot mismatch: {path}")
    if not points and not confirms_structural_zero(report):
        raise ValueError(f"Empty cutoff is not a confirmed structural zero: {path}; refresh its source report")
    return points
