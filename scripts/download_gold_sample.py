import pandas as pd
import pyvo
import os
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import argparse
import sys

def read_eso_credentials(env_file):
    # First, try environment variables
    user = os.environ.get("ESO_USERNAME")
    pw = os.environ.get("ESO_PASSWORD")

    if user and pw:
        return user, pw

    # If not found, try eso.env file
    if os.path.exists(env_file):
        creds = {}
        with open(env_file) as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    creds[k.strip()] = v.strip()
        user = creds.get("ESO_USERNAME")
        pw = creds.get("ESO_PASSWORD")
        if user and pw:
            return user, pw

    raise EnvironmentError(
        "Please set ESO_USERNAME and ESO_PASSWORDs environment variables "
        "or provide them in an eso.env file for authentication."
    )


def download_data(instrument, target, output_dir, USER, PASS):
    tap = pyvo.dal.TAPService("https://archive.eso.org/tap_obs")

    import re
    def safe_str(s):
        return re.sub(r"[^a-zA-Z0-9_\- ]", "", str(s))

    safe_instrument = safe_str(instrument)
    safe_target = safe_str(target)

    query = f"""
    SELECT obs_publisher_did, access_url
    FROM ivoa.obscore
    WHERE instrument_name='{safe_instrument}'
    AND target_name='{safe_target}'
    """

    result = tap.search(query)

    outdir = Path(output_dir) / safe_instrument / safe_target
    outdir.mkdir(parents=True, exist_ok=True)

    for row in result:
        obs_id = safe_str(row["obs_publisher_did"])
        datalink_url = row["access_url"]

        fname = f"{safe_target}_{obs_id}.fits"
        final_path = outdir / fname
        tmp_path = final_path.with_suffix(".fits.part")

        # -----------------------------
        # FAST SKIP: already downloaded
        # -----------------------------
        if final_path.exists():
            continue

        # -----------------------------
        # Fetch DataLink
        # -----------------------------
        try:
            dl = requests.get(datalink_url, auth=(USER, PASS), timeout=60)
            dl.raise_for_status()
        except requests.HTTPError as err:
            if err.response is not None and err.response.status_code == 401:
                print("401 Unauthorized while fetching DataLink — check ESO credentials")
                return
            print(f"DataLink fetch failed: {err}")
            continue

        # -----------------------------
        # Parse XML
        # -----------------------------
        try:
            root = ET.fromstring(dl.text)
        except ET.ParseError as err:
            print(f"XML parse error: {err}")
            continue

        fits_url = None
        for td in root.iter("{http://www.ivoa.net/xml/VOTable/v1.3}TD"):
            if td.text and "dataPortal/file" in td.text:
                fits_url = td.text
                break

        if not fits_url:
            print("No FITS URL found in DataLink")
            continue

        # -----------------------------
        # Download FITS (atomic)
        # -----------------------------
        try:
            r = requests.get(fits_url, auth=(USER, PASS), stream=True, timeout=300)
            r.raise_for_status()
        except requests.HTTPError as err:
            if err.response is not None and err.response.status_code == 401:
                print("401 Unauthorized while downloading FITS — check ESO credentials")
                return
            print(f"FITS download failed: {err}")
            continue
        except requests.Timeout as err:
            print(f"FITS download timed out: {err}")
            continue
        except Exception as err:
            print(f"Unexpected error downloading FITS: {err}")
            continue

        try:
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            tmp_path.rename(final_path)
            print("Saved:", final_path.name)

        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Download ESO gold sample FITS files matching input table.")
    parser.add_argument("--env", default="eso.env", help="Path to eso.env file (default: eso.env)")
    parser.add_argument("--input", default="fgk.txt", help="Path to input file in TSV format (default: fgk.txt)")
    parser.add_argument("--output", default="fgk_spectra", help="Output directory for downloaded spectra (default: fgk_spectra)")
    args = parser.parse_args()

    try:
        USER, PASS = read_eso_credentials(args.env)
    except Exception as e:
        print(str(e))
        sys.exit(1)

    fgk = pd.read_csv(args.input, sep="\t")
    output_dir = args.output

    for star in tqdm(fgk.to_records()):
        download_data(star['origin'], star['star'], output_dir, USER, PASS)
        if star['star_alt1'] != '-':
            download_data(star['origin'], star['star_alt1'], output_dir, USER, PASS)
        if star['star_alt2'] != '-':
            download_data(star['origin'], star['star_alt2'], output_dir, USER, PASS)

if __name__ == "__main__":
    main()