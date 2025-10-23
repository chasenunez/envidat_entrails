#!/usr/bin/env python3
"""
envidat_filetype_tool.py

Single-file tool to: (1) crawl S3-style XML list endpoints for EnviDat-style buckets,
(2) compile a CSV of file metadata across buckets, and (3) visualize file-type
distributions (sunburst + sankey) from that CSV.

Usage examples (from shell):

# Fetch listings for the default buckets and write to all_s3_files.csv
python envidat_filetype_tool.py fetch --out all_s3_files.csv

# Fetch listings for a custom list of buckets (comma-separated)
python envidat_filetype_tool.py fetch --buckets https://os.zhdk.cloud.switch.ch/envicloud/,https://os.zhdk.cloud.switch.ch/chelsav1/ --out out.csv

# Visualize the CSV created by the fetch step
python envidat_filetype_tool.py visualize --csv all_s3_files.csv --out-prefix envidat_viz

# Do both (fetch then visualize)
python envidat_filetype_tool.py run-all --out all_s3_files.csv --out-prefix envidat_viz

Notes:
- Internet access is required for the `fetch` step. The `visualize` step works offline
  from the CSV produced by `fetch`.
- The script expects S3 ListBucketResult XML pages. It handles pagination using
  <IsTruncated> and <NextMarker> (or last Key) as described in the S3 API.

Dependencies:
- Python 3.8+
- requests
- pandas
- plotly

Install dependencies with:
pip install requests pandas plotly

"""

import argparse
import csv
import os
import re
import sys
import time
import logging
from typing import Optional

try:
    import requests
except Exception as e:
    print("ERROR: requests is required. Install with: pip install requests")
    raise

try:
    import pandas as pd
except Exception as e:
    print("ERROR: pandas is required. Install with: pip install pandas")
    raise

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception as e:
    print("ERROR: plotly is required. Install with: pip install plotly")
    raise

# ----- Configuration / defaults -----
DEFAULT_BUCKETS = [
    "https://os.zhdk.cloud.switch.ch/envidat-doi/",
    "https://os.zhdk.cloud.switch.ch/envicloud/",
    "https://os.zhdk.cloud.switch.ch/chelsav1/",
    "https://os.zhdk.cloud.switch.ch/chelsav2/",
    "https://os.zhdk.cloud.switch.ch/edna/",
]

CSV_HEADERS = [
    'bucket_url', 'bucket_name', 'key', 'last_modified', 'etag', 'size', 'storage_class',
    'owner_id', 'owner_display_name', 'type'
]

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('envidat_tool')

# ----- Helpers for XML parsing (remove namespace, robust find) -----

def _strip_s3_xml_namespace(xml_text: str) -> str:
    """Remove the s3 xmlns attribute to simplify ElementTree parsing.

    Many S3 listing XML responses use the namespace
    xmlns="http://s3.amazonaws.com/doc/2006-03-01/" which makes tag lookup messy.
    This function strips that namespace declaration.
    """
    # remove xmlns=... (only the attribute, not other occurrences)
    return re.sub(r"\sxmlns=\"[^\"]+\"", '', xml_text, count=1)


def _safe_find_text(elem, tag):
    """Find subelement text or return empty string if not present."""
    child = elem.find(tag)
    return child.text if child is not None else ''


# ----- Core: fetch/list S3-style bucket pages -----

def list_s3_bucket_to_csv(bucket_url: str, csv_writer: csv.DictWriter, session: requests.Session, sleep: float = 0.0, max_pages: Optional[int] = None):
    """
    Crawl a single S3-style bucket listing endpoint and write rows to csv_writer.

    Parameters
    ----------
    bucket_url : str
        The root URL that returns S3 ListBucketResult XML (e.g. https://.../)
    csv_writer : csv.DictWriter
        A writer already initialized with CSV_HEADERS.
    session : requests.Session
        Session object to allow connection reuse.
    sleep : float
        Seconds to sleep between page requests to be polite.
    max_pages : Optional[int]
        If set, limit the number of pages fetched (useful for testing).
    """
    logger.info("Starting bucket: %s", bucket_url)
    bucket_name = bucket_url.rstrip('/').split('/')[-1]
    marker: Optional[str] = None
    page_count = 0

    while True:
        # Build URL and params. S3 ListObjects v1 uses 'marker' for pagination.
        params = {}
        if marker:
            params['marker'] = marker
            logger.debug('Requesting page with marker=%s', marker)
        try:
            resp = session.get(bucket_url, params=params, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            logger.error('Failed to GET %s (marker=%s): %s', bucket_url, marker, e)
            raise

        xml_text = _strip_s3_xml_namespace(resp.text)

        # Parse XML with ElementTree
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_text)
        except Exception as e:
            logger.error('Failed to parse XML for bucket %s (marker=%s): %s', bucket_url, marker, e)
            raise

        # Iterate over Contents entries
        contents = root.findall('Contents')
        logger.info('Got %d Contents entries (page %d)', len(contents), page_count + 1)

        for content in contents:
            key = _safe_find_text(content, 'Key')
            last_modified = _safe_find_text(content, 'LastModified')
            etag = _safe_find_text(content, 'ETag')
            size = _safe_find_text(content, 'Size')
            storage_class = _safe_find_text(content, 'StorageClass')
            owner = content.find('Owner')
            owner_id = _safe_find_text(owner, 'ID') if owner is not None else ''
            owner_display = _safe_find_text(owner, 'DisplayName') if owner is not None else ''
            type_ = _safe_find_text(content, 'Type')

            csv_writer.writerow({
                'bucket_url': bucket_url,
                'bucket_name': bucket_name,
                'key': key,
                'last_modified': last_modified,
                'etag': etag,
                'size': size,
                'storage_class': storage_class,
                'owner_id': owner_id,
                'owner_display_name': owner_display,
                'type': type_,
            })

        page_count += 1
        # Pagination control: check <IsTruncated>
        is_truncated_tag = root.find('IsTruncated')
        is_truncated = (is_truncated_tag is not None and is_truncated_tag.text.lower() == 'true')

        if not is_truncated:
            logger.info('No more pages for bucket %s', bucket_url)
            break

        # Determine next marker. Use <NextMarker> if present, otherwise use last Key
        next_marker_tag = root.find('NextMarker')
        if next_marker_tag is not None and next_marker_tag.text:
            marker = next_marker_tag.text
        else:
            # fallback: use last content's Key
            if contents:
                last_key = contents[-1].find('Key')
                marker = last_key.text if last_key is not None else None
            else:
                logger.warning('IsTruncated true but no Contents entries found; stopping to avoid infinite loop')
                break

        # optional limits for testing
        if max_pages is not None and page_count >= max_pages:
            logger.info('Reached max_pages=%d for bucket %s; stopping early', max_pages, bucket_url)
            break

        if sleep and sleep > 0:
            time.sleep(sleep)

    logger.info('Finished bucket: %s (pages fetched=%d)', bucket_url, page_count)


# ----- Command: fetch (create the CSV) -----

def cmd_fetch(buckets, out_csv, sleep_between_requests=0.0, max_pages=None):
    """Fetch listings for all given buckets and write to out_csv."""
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    session = requests.Session()

    with open(out_csv, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
        writer.writeheader()

        for bucket in buckets:
            list_s3_bucket_to_csv(bucket, writer, session, sleep=sleep_between_requests, max_pages=max_pages)

    logger.info('All buckets processed. CSV written to: %s', out_csv)


# ----- Command: visualize (reads CSV and outputs visualization html files) -----

def cmd_visualize(csv_path, out_prefix='envidat_viz', top_n_extensions: Optional[int] = None):
    """Read the CSV produced by fetch and create visualizations.

    Produces two files:
    - {out_prefix}_sunburst.html
    - {out_prefix}_sankey.html
    """
    logger.info('Reading CSV: %s', csv_path)
    df = pd.read_csv(csv_path, dtype={'bucket_url': str, 'key': str, 'size': object})

    # derive extension (take last dot). treat files without extension as '<no_ext>'
    def get_ext(k):
        if not isinstance(k, str) or k.strip() == '':
            return '<no_ext>'
        base = os.path.basename(k)
        if '.' not in base:
            return '<no_ext>'
        return os.path.splitext(base)[1].lower() or '<no_ext>'

    df['extension'] = df['key'].apply(get_ext)

    # Aggregate counts per bucket/extension
    df_counts = df.groupby(['bucket_name', 'extension'], as_index=False).size().rename(columns={'size': 'count'})
    # Pandas < 2 compatibility: if .size returns series, fix
    if 'count' not in df_counts.columns:
        df_counts = df.groupby(['bucket_name', 'extension']).size().reset_index(name='count')

    logger.info('Total rows (files): %d', len(df))
    logger.info('Unique (bucket, extension) rows: %d', len(df_counts))

    # Optionally reduce to top N extensions overall (makes charts simpler)
    if top_n_extensions is not None:
        total_by_ext = df.groupby('extension').size().reset_index(name='total').sort_values('total', ascending=False)
        top_exts = set(total_by_ext['extension'].iloc[:top_n_extensions].tolist())
        df_counts.loc[~df_counts['extension'].isin(top_exts), 'extension'] = '<other>'
        df_counts = df_counts.groupby(['bucket_name', 'extension'], as_index=False)['count'].sum()

    # ---------- Sunburst chart: bucket -> extension ----------
    sunburst_path = ['bucket_name', 'extension']
    sunburst_fig = px.sunburst(df_counts, path=sunburst_path, values='count', title='File types by bucket (sunburst)')
    sunburst_out = f"{out_prefix}_sunburst.html"
    sunburst_fig.write_html(sunburst_out)
    logger.info('Sunburst written to %s', sunburst_out)

    # ---------- Sankey chart: Total -> extension ----------
    total_by_extension = df.groupby('extension').size().reset_index(name='count').sort_values('count', ascending=False)

    labels = ['Total files'] + total_by_extension['extension'].tolist()
    source = []
    target = []
    value = []

    for i, row in enumerate(total_by_extension.itertuples(index=False)):
        source.append(0)  # from 'Total files' node
        target.append(1 + i)
        value.append(int(row.count))

    sankey_fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=15, thickness=20),
        link=dict(source=source, target=target, value=value)
    ))
    sankey_fig.update_layout(title='File type breakdown (Total -> file extension)')
    sankey_out = f"{out_prefix}_sankey.html"
    sankey_fig.write_html(sankey_out)
    logger.info('Sankey written to %s', sankey_out)

    logger.info('Visualization complete.')


# ----- CLI wiring -----

def main(argv=None):
    p = argparse.ArgumentParser(description='EnviDat S3 listing crawler and visualizer')
    sub = p.add_subparsers(dest='cmd', required=True)

    # fetch
    pf = sub.add_parser('fetch', help='Fetch S3 listings and save to CSV')
    pf.add_argument('--buckets', type=str, default=','.join(DEFAULT_BUCKETS),
                    help='Comma-separated list of bucket root URLs (default: built-in list)')
    pf.add_argument('--out', type=str, default='all_s3_files.csv', help='Output CSV path')
    pf.add_argument('--sleep', type=float, default=0.0, help='Seconds to sleep between page requests')
    pf.add_argument('--max-pages', type=int, default=None, help='Limit pages per bucket (for testing)')

    # visualize
    pv = sub.add_parser('visualize', help='Visualize from an existing CSV produced by fetch')
    pv.add_argument('--csv', type=str, required=True, help='CSV path produced by fetch')
    pv.add_argument('--out-prefix', type=str, default='envidat_viz', help='Prefix for output HTML files')
    pv.add_argument('--top-n-extensions', type=int, default=None, help='If set, collapse extensions to top-N and group others')

    # run-all convenience
    pr = sub.add_parser('run-all', help='Run fetch then visualize in sequence')
    pr.add_argument('--buckets', type=str, default=','.join(DEFAULT_BUCKETS),
                    help='Comma-separated list of bucket root URLs (default: built-in list)')
    pr.add_argument('--out', type=str, default='all_s3_files.csv', help='Output CSV path')
    pr.add_argument('--out-prefix', type=str, default='envidat_viz', help='Prefix for output HTML files')
    pr.add_argument('--sleep', type=float, default=0.0, help='Seconds to sleep between page requests')
    pr.add_argument('--max-pages', type=int, default=None, help='Limit pages per bucket (for testing)')
    pr.add_argument('--top-n-extensions', type=int, default=None, help='Collapse extensions to top-N for visualization')

    args = p.parse_args(argv)

    if args.cmd == 'fetch':
        buckets = [b.strip() for b in args.buckets.split(',') if b.strip()]
        cmd_fetch(buckets, args.out, sleep_between_requests=args.sleep, max_pages=args.max_pages)

    elif args.cmd == 'visualize':
        cmd_visualize(args.csv, out_prefix=args.out_prefix, top_n_extensions=args.top_n_extensions)

    elif args.cmd == 'run-all':
        buckets = [b.strip() for b in args.buckets.split(',') if b.strip()]
        cmd_fetch(buckets, args.out, sleep_between_requests=args.sleep, max_pages=args.max_pages)
        cmd_visualize(args.out, out_prefix=args.out_prefix, top_n_extensions=args.top_n_extensions)


if __name__ == '__main__':
    main()
