#!/usr/bin/env python3
"""Download the full AMBRE grid in small Teff/logg batches.

This script automates the ``ambre.ipynb`` example by tiling the parameter
space into small (Teff, logg) boxes and issuing a separate SSA query for each
box. Each returned FITS file is downloaded and saved locally.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, Tuple
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

import pyvo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download AMBRE spectra from the Pollux SSA service in small parameter chunks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ambre_spectra"),
        help="Directory where FITS files will be stored (default: %(default)s).",
    )
    parser.add_argument(
        "--service-url",
        default="https://pollux.oreme.org/vo/ssa",
        help="SSA service URL (default: %(default)s).",
    )
    parser.add_argument(
        "--collection", default="AMBRE", help="SSA collection name (default: %(default)s)."
    )
    parser.add_argument(
        "--spec-type", default="spec", help="SSA spectral type (default: %(default)s)."
    )
    parser.add_argument(
        "--format", dest="format_", default="application/fits", help="Desired format (default: %(default)s)."
    )
    parser.add_argument("--teff-min", type=float, default=3000.0, help="Minimum Teff to query.")
    parser.add_argument("--teff-max", type=float, default=8000.0, help="Maximum Teff to query.")
    parser.add_argument(
        "--teff-interval",
        type=float,
        default=250.0,
        help="Teff interval for each query window (default: %(default)s).",
    )
    parser.add_argument("--logg-min", type=float, default=0.0, help="Minimum logg to query.")
    parser.add_argument("--logg-max", type=float, default=5.5, help="Maximum logg to query.")
    parser.add_argument(
        "--logg-interval",
        type=float,
        default=0.25,
        help="logg interval for each query window (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download FITS files even if they already exist locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List which files would be downloaded without performing network requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of retries for failed SSA queries (default: %(default)s).",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=3.0,
        help="Seconds to wait between failed retries (default: %(default)s).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout for individual file downloads in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Path to a progress file (JSONL). Defaults to <output-dir>/download_progress.jsonl.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file to mirror console output.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def frange(start: float, stop: float, step: float) -> Iterable[Tuple[float, float]]:
    """Yield half-open numeric intervals [lower, upper)."""
    if step <= 0:
        raise ValueError("Step size must be positive.")
    idx = 0
    while True:
        lower = start + idx * step
        if lower >= stop:
            break
        upper = min(lower + step, stop)
        yield (round(lower, 6), round(upper, 6))
        idx += 1


def derive_filename(record: pyvo.dal.ssa.SSARecord) -> str:
    """Derive a sensible filename for an SSA record."""
    url = record.getdataurl()
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    candidate = qs.get("ID", [None])[0]
    name = candidate or Path(parsed.path).name or record.get("title", "spectrum")
    if not name.endswith(".fits"):
        name = f"{name}.fits"
    # Sanitize stray query characters in the path component
    return name.replace("/", "_")


def download_url(url: str, destination: Path, timeout: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout) as response, destination.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def load_state(state_file: Path) -> set[str]:
    """Load previously processed filenames from a JSONL progress file."""
    processed: set[str] = set()
    if not state_file.exists():
        return processed

    try:
        import json

        with state_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                name = entry.get("name")
                if name:
                    processed.add(name)
    except Exception:
        logging.warning("Failed to load state file %s; continuing without it.", state_file, exc_info=True)
    return processed


def append_state(state_file: Path, **fields: object) -> None:
    """Append a JSON record to the progress file."""
    try:
        import json

        state_file.parent.mkdir(parents=True, exist_ok=True)
        with state_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(fields, sort_keys=True))
            fh.write("\n")
    except Exception:
        logging.warning("Could not write to state file %s", state_file, exc_info=True)


def fetch_chunk(
    service: pyvo.ssa.SSAService,
    *,
    collection: str,
    spec_type: str,
    format_: str,
    teff_min: float,
    teff_max: float,
    logg_min: float,
    logg_max: float,
    max_retries: int = 3,
    retry_wait: float = 3.0,
) -> pyvo.dal.query.SSAQueryResult:
    """Run a single SSA query with retry/backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            return service.search(
                collection=collection,
                spec_type=spec_type,
                format=format_,
                teff_min=teff_min,
                teff_max=teff_max,
                logg_min=logg_min,
                logg_max=logg_max,
            )
        except Exception as exc:  # noqa: BLE001 - need to surface all failures
            if attempt == max_retries:
                raise
            logging.warning(
                "SSA query failed (attempt %s/%s): %s. Retrying in %.1f s...",
                attempt,
                max_retries,
                exc,
                retry_wait,
            )
            time.sleep(retry_wait)
    raise RuntimeError("Unreachable code path in fetch_chunk.")


def main() -> int:
    args = parse_args()
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=log_handlers,
    )
    output_dir: Path = args.output_dir
    state_file: Path = args.state_file or output_dir / "download_progress.jsonl"

    service = pyvo.ssa.SSAService(args.service_url)
    logging.info(
        "Initialized SSA service %s for collection=%s format=%s",
        args.service_url,
        args.collection,
        args.format_,
    )

    processed = load_state(state_file)
    if processed:
        logging.info("Loaded %d previously processed entries from %s", len(processed), state_file)

    downloaded = 0
    skipped = 0
    seen: set[str] = set(processed)

    for teff_min, teff_max in frange(args.teff_min, args.teff_max, args.teff_interval):
        for logg_min, logg_max in frange(args.logg_min, args.logg_max, args.logg_interval):
            logging.info("Querying Teff=[%.1f, %.1f], logg=[%.2f, %.2f]", teff_min, teff_max, logg_min, logg_max)
            try:
                results = fetch_chunk(
                    service,
                    collection=args.collection,
                    spec_type=args.spec_type,
                    format_=args.format_,
                    teff_min=teff_min,
                    teff_max=teff_max,
                    logg_min=logg_min,
                    logg_max=logg_max,
                    max_retries=args.max_retries,
                    retry_wait=args.retry_wait,
                )
            except Exception as exc:  # noqa: BLE001
                logging.error(
                    "Skipping chunk Teff=[%.1f, %.1f], logg=[%.2f, %.2f] due to repeated errors: %s",
                    teff_min,
                    teff_max,
                    logg_min,
                    logg_max,
                    exc,
                    exc_info=True,
                )
                append_state(
                    state_file,
                    event="chunk_error",
                    teff_range=[teff_min, teff_max],
                    logg_range=[logg_min, logg_max],
                    error=str(exc),
                    timestamp=time.time(),
                )
                continue

            if len(results) == 0:
                logging.debug("No spectra in this chunk.")
                append_state(
                    state_file,
                    event="chunk_empty",
                    teff_range=[teff_min, teff_max],
                    logg_range=[logg_min, logg_max],
                    timestamp=time.time(),
                )
                continue

            logging.info("Chunk returned %d spectrum files.", len(results))
            for record in results:
                try:
                    name = derive_filename(record)
                except Exception as exc:  # noqa: BLE001
                    logging.warning("Could not derive filename for record; using fallback: %s", exc)
                    name = f"spectrum_{downloaded + skipped:05d}.fits"

                if name in seen:
                    skipped += 1
                    continue
                seen.add(name)

                destination = output_dir / name
                if destination.exists() and not args.overwrite:
                    skipped += 1
                    append_state(
                        state_file,
                        event="exists",
                        name=name,
                        url=record.getdataurl(),
                        teff_range=[teff_min, teff_max],
                        logg_range=[logg_min, logg_max],
                        timestamp=time.time(),
                    )
                    continue

                if args.dry_run:
                    logging.info("[dry-run] Would download %s -> %s", record.getdataurl(), destination)
                    downloaded += 1
                    append_state(
                        state_file,
                        event="dry_run",
                        name=name,
                        url=record.getdataurl(),
                        teff_range=[teff_min, teff_max],
                        logg_range=[logg_min, logg_max],
                        timestamp=time.time(),
                    )
                    continue

                try:
                    download_url(record.getdataurl(), destination, timeout=args.timeout)
                except Exception as exc:  # noqa: BLE001
                    logging.error("Failed to download %s: %s", name, exc, exc_info=True)
                    append_state(
                        state_file,
                        event="error",
                        name=name,
                        url=record.getdataurl(),
                        teff_range=[teff_min, teff_max],
                        logg_range=[logg_min, logg_max],
                        error=str(exc),
                        timestamp=time.time(),
                    )
                    continue

                downloaded += 1
                logging.debug("Saved %s", destination)
                append_state(
                    state_file,
                    event="downloaded",
                    name=name,
                    url=record.getdataurl(),
                    teff_range=[teff_min, teff_max],
                    logg_range=[logg_min, logg_max],
                    timestamp=time.time(),
                )

    logging.info("Download complete. New files: %d, skipped: %d", downloaded, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
