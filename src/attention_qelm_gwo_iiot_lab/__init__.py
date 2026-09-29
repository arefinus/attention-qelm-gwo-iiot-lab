"""attention_qelm_gwo_iiot_lab.

Independent implementation of an attention encoder with a fixed-weight extreme learning
machine head and grey wolf hyperparameter search for IIoT intrusion detection.

Everything in this package runs on CPU, offline, on authored synthetic fixtures. No
dataset rows are shipped. See README.md and docs/paper-implementation-audit.md.
"""
from __future__ import annotations

__version__ = "0.1.0"
PROJECT_ID = "attention-qelm-gwo-iiot-lab"
RELEASE_DATE = "2026-09-27"
EVIDENCE_LABELS = ("paper-reported", "reproduced", "demo", "proposed")
