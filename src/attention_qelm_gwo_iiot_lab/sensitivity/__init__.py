"""Shortcut and leakage audits (evidence label: demo on the fixture, proposed on real data)."""
from __future__ import annotations

from .audits import (duplicate_flow_check, identifier_like_columns, placeholder_encoding_audit,
                     run_all_audits)

__all__ = ["duplicate_flow_check", "identifier_like_columns", "placeholder_encoding_audit",
           "run_all_audits"]
