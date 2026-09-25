#!/usr/bin/env python3
"""Print the submitted PBS array range as ``START END STEP`` (empty on failure).

PBS Pro gives each subjob a PBS_JOBID like ``12345[9].gadi-pbs`` — its own
index, no range — so the range must come from the parent job. Order:
``[a-b]`` in PBS_JOBID (parent job), then ``qstat -f $PBS_ARRAY_ID``
(``array_indices_submitted = a-b[:s]``).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys


def parse_range(text: str):
    m = re.search(r"(\d+)-(\d+)(?::(\d+))?", text or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)


def array_range(env=os.environ):
    jobid = env.get("PBS_JOBID", "")
    m = re.search(r"\[(\d+-\d+(?::\d+)?)\]", jobid)
    if m:
        return parse_range(m.group(1))
    parent = env.get("PBS_ARRAY_ID", "")
    if not parent:
        return None
    try:
        out = subprocess.run(["qstat", "-f", parent], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"array_indices_submitted\s*=\s*([0-9:-]+)", out)
    return parse_range(m.group(1)) if m else None


if __name__ == "__main__":
    rng = array_range()
    if rng:
        print(*rng)
    sys.exit(0)
