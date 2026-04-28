#!/usr/bin/env python3
"""Crawl S3-style XML list endpoints, write file metadata to CSV, and produce
sunburst + sankey visualizations of file-type distribution.

Run `python entrails.py --help` for CLI usage.
"""

import argparse
import csv
import io
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests

DEFAULT_BUCKETS = [
    "https://os.zhdk.cloud.switch.ch/envidat-doi/",
    "https://os.zhdk.cloud.switch.ch/envicloud/",
    "https://os.zhdk.cloud.switch.ch/chelsav1/",
    "https://os.zhdk.cloud.switch.ch/chelsav2/",
    "https://s3-zh.os.switch.ch/pointclouds",
    "https://s3-zh.os.switch.ch/drone-data",
    "https://os.zhdk.cloud.switch.ch/edna/",
]

CSV_HEADERS = [
    'bucket_url', 'bucket_name', 'key', 'last_modified', 'etag', 'size', 'storage_class',
    'owner_id', 'owner_display_name', 'type'
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('envidat_tool')


# - need to show human readable byte amount (why can i never do this in my head?)
def human_bytes(n):
    n = int(n)
    for unit in ['B','KiB','MiB','GiB','TiB','PiB']:
        if abs(n) < 1024.0:
            return f'{n:3.1f}{unit}'
        n /= 1024.0
    return f'{n:.1f}EiB'

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


def _parse_le_int(b: bytes) -> int:
    """Parse little-endian unsigned integer from bytes (len 2 or 4)."""
    return int.from_bytes(b, byteorder='little', signed=False)

def _http_range_get(session: requests.Session, url: str, start: int, length: int, allow_full_fallback: bool = True):
    """
    Perform an HTTP ranged GET for bytes [start, start+length-1].
    Returns bytes or raises.
    If server responds with 200 (full content) and allow_full_fallback=True, returns entire content.
    """
    end = start + max(0, length) - 1
    headers = {'Range': f'bytes={start}-{end}'}
    resp = session.get(url, headers=headers, timeout=60)
    # 206 Partial Content is expected; 200 may mean server ignored Range and returned full file
    if resp.status_code in (200, 206):
        return resp.content, resp.status_code
    # Other statuses: raise for visibility
    resp.raise_for_status()
    return resp.content, resp.status_code

# woof this one was a heavy lift. If it works, itll be a miracle.
def inspect_zip_entries_remote(bucket_root_url: str, key: str, size: int, session: requests.Session,
                              max_eocd_search: int = 1024*64,  # how many bytes to read from the end to find EOCD
                              max_full_download: int = 1024*1024*50):
    """
    Inspect a remote ZIP accessible at bucket_root_url + key using ranged GETs.
    Returns a list of dicts describing zip entries:
      [{'filename': ..., 'compress_size': ..., 'file_size': ..., 'compress_type': ...}, ...]
    Behavior:
    - Try to fetch only EOCD and Central Directory via Range requests.
    - If server ignores Range and returns full file (status 200) and file is small (<max_full_download),
      we'll parse it as a normal zip.
    - If ZIP64 or other unusual cases occur, function will fall back to safer behavior or return [].
    """
    full_url = bucket_root_url.rstrip('/') + '/' + key.lstrip('/')
    entries = []

    # 1) Read tail of file to find EOCD signature (0x06054b50)
    #    EOCD record minimum size is 22 bytes; comment can extend it. ZIP64 requires special handling.
    read_len = min(size, max_eocd_search)
    start = max(0, size - read_len)
    try:
        tail_bytes, status = _http_range_get(session, full_url, start, read_len)
    except requests.RequestException as e:
        logger.warning("Range GET for EOCD failed for %s: %s", full_url, e)
        return entries

    # If server returned whole file (200) and file is small, hand to zipfile directly
    if status == 200:
        if len(tail_bytes) <= max_full_download:
            try:
                zf = zipfile.ZipFile(io.BytesIO(tail_bytes))
                for zi in zf.infolist():
                    entries.append({'filename': zi.filename, 'compress_size': zi.compress_size,
                                    'file_size': zi.file_size, 'compress_type': zi.compress_type})
                return entries
            except zipfile.BadZipFile as e:
                logger.warning("Full-download parsing failed for %s: %s", full_url, e)
                return entries
        else:
            logger.warning("Server ignored Range and file too large to download (%d bytes): %s", len(tail_bytes), full_url)
            return entries

    # Search for EOCD signature (last occurrence)
    eocd_sig = b'PK\x05\x06'  # 0x06054b50 little-endian
    idx = tail_bytes.rfind(eocd_sig)
    if idx == -1:
        # Possibly ZIP64 or EOCD beyond max_eocd_search. Try a larger read if feasible.
        logger.warning("EOCD signature not found in last %d bytes of %s", read_len, full_url)
        return entries

    # EOCD structure (from signature position):
    # offset 0: 4 bytes signature
    # offset 4: 2 bytes disk number
    # offset 6: 2 bytes disk with start of central dir
    # offset 8: 2 bytes num entries on this disk
    # offset 10: 2 bytes total entries
    # offset 12: 4 bytes central directory size
    # offset 16: 4 bytes central directory offset (start)
    # offset 20: 2 bytes comment length
    try:
        eocd = tail_bytes[idx: idx + 22]
        cd_size = _parse_le_int(eocd[12:16])
        cd_start = _parse_le_int(eocd[16:20])
    except (IndexError, ValueError) as e:
        logger.warning("Failed to parse EOCD in %s: %s", full_url, e)
        return entries

    # 2) Fetch central directory
    try:
        cd_bytes, status2 = _http_range_get(session, full_url, cd_start, cd_size)
    except requests.RequestException as e:
        logger.warning("Failed to fetch central directory for %s: %s", full_url, e)
        return entries

    # If server returned full file (status2 == 200), and cd_bytes is whole file, parse via zipfile if small
    if status2 == 200 and len(cd_bytes) <= max_full_download:
        try:
            zf = zipfile.ZipFile(io.BytesIO(cd_bytes))
            for zi in zf.infolist():
                entries.append({'filename': zi.filename, 'compress_size': zi.compress_size,
                                'file_size': zi.file_size, 'compress_type': zi.compress_type})
            return entries
        except zipfile.BadZipFile as e:
            logger.debug("Full-file zip parse failed (will fall back to CD+EOCD): %s", e)

    # 3) Create a minimal fake zip by concatenating CD + EOCD and feed to ZipFile
    try:
        # We need full EOCD bytes; we extracted it from tail_bytes at idx.
        full_eocd = tail_bytes[idx: idx + 22]
        fakezip = io.BytesIO(cd_bytes + full_eocd)
        zf = zipfile.ZipFile(fakezip)
        for zi in zf.infolist():
            entries.append({'filename': zi.filename, 'compress_size': zi.compress_size,
                            'file_size': zi.file_size, 'compress_type': zi.compress_type})
        return entries
    except zipfile.BadZipFile as e:
        # Could be ZIP64 or other complicated cases; we skip in that case.
        logger.warning("Failed to build fake zip from CD+EOCD for %s: %s", full_url, e)
        return entries


def list_s3_bucket_to_csv(bucket_url: str, csv_writer: csv.DictWriter, session: requests.Session, sleep: float = 0.0, max_pages: Optional[int] = None):
    """
    Crawl a single S3-style bucket listing endpoint and write rows to csv_writer.
    """
    logger.info("Starting bucket: %s", bucket_url)
    bucket_name = bucket_url.rstrip('/').split('/')[-1]
    marker: Optional[str] = None
    page_count = 0

    # Counters for skipped rows (for informative logging of what we are weeding out following IIE's comment)
    skipped_envidat1 = 0      # count of keys skipped because they contain 'envidat.1'
    skipped_doi_meta = 0      # count of keys skipped from the envidat-doi bucket for certain extensions

    while True:
        # Build URL and params. S3 ListObjects v1 uses 'marker' for pagination.
        params = {}
        if marker:
            params['marker'] = marker
            logger.debug('Requesting page with marker=%s', marker)
        try:
            resp = session.get(bucket_url, params=params, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error('Failed to GET %s (marker=%s): %s', bucket_url, marker, e)
            raise

        xml_text = _strip_s3_xml_namespace(resp.text)

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
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

            lower_key = key.lower() if isinstance(key, str) else ''

            # Rule 1: skip 'envidat.1' DOI-only datasets
            if 'envidat.1' in lower_key:
                skipped_envidat1 += 1
                continue

            # Rule 2: skip .html/.json/.xml metadata files in the envidat-doi bucket
            if bucket_name == 'envidat-doi':
                _, ext = os.path.splitext(key if key is not None else '')
                if ext.lower() in ('.html', '.json', '.xml'):
                    skipped_doi_meta += 1
                    continue

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


            # Only attempt to inspect inner ZIP contents for .zip files with a valid numeric size
            _, ext = os.path.splitext(key if key is not None else '')
            ext = ext.lower()
            try:
                size_int = int(size) if (size is not None and str(size).strip() != '') else None
            except (ValueError, TypeError):
                size_int = None

            if ext == '.zip' and size_int is not None and size_int > 0:
                # inspect_zip_entries_remote handles its own errors and returns [] on failure
                inner_entries = inspect_zip_entries_remote(bucket_url, key, size_int, session)
                if inner_entries:
                    logger.info("Found %d entries inside ZIP %s", len(inner_entries), key)
                    for ie in inner_entries:
                        inner_key = f"{key}::{ie['filename']}"
                        # We reuse bucket-level metadata; for size we write compressed size
                        csv_writer.writerow({
                            'bucket_url': bucket_url,
                            'bucket_name': bucket_name,
                            'key': inner_key,
                            'last_modified': last_modified,
                            'etag': etag,
                            'size': str(ie.get('compress_size', '')),
                            'storage_class': storage_class,
                            'owner_id': owner_id,
                            'owner_display_name': owner_display,
                            'type': type_,
                        })
                else:
                    logger.debug("No inner entries discovered (or skipped due to complexity) for ZIP %s", key)

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

    # Log skipped counts so the user knows what's been excluded
    if skipped_envidat1:
        logger.info("Skipped %d objects containing 'envidat.1' in their path for bucket %s", skipped_envidat1, bucket_name)
    if skipped_doi_meta:
        logger.info("Skipped %d metadata files (.html/.json/.xml) in envidat-doi bucket %s", skipped_doi_meta, bucket_name)

    logger.info('Finished bucket: %s (pages fetched=%d)', bucket_url, page_count)


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

    df_counts = df.groupby(['bucket_name', 'extension']).size().reset_index(name='count')

    logger.info('Total rows (files): %d', len(df))
    logger.info('Unique (bucket, extension) rows: %d', len(df_counts))

    if top_n_extensions is not None:
        total_by_ext = df.groupby('extension').size().reset_index(name='total').sort_values('total', ascending=False)
        top_exts = set(total_by_ext['extension'].iloc[:top_n_extensions].tolist())
        df_counts.loc[~df_counts['extension'].isin(top_exts), 'extension'] = '<other>'
        df_counts = df_counts.groupby(['bucket_name', 'extension'], as_index=False)['count'].sum()

    df_counts['count'] = df_counts['count'].astype(int)
    df_counts['bucket_name'] = df_counts['bucket_name'].astype(str)
    df_counts['extension'] = df_counts['extension'].astype(str)

    # Build hierarchical ids, labels, parents, and values for sunburst (root -> bucket -> extension)
    labels = []
    ids = []
    parents = []
    values = []

    # root node
    ids.append('root')
    labels.append('All files')
    parents.append('')
    values.append(int(df_counts['count'].sum()))

    # bucket nodes (one per bucket)
    buckets = df_counts.groupby('bucket_name', as_index=False)['count'].sum()
    for row in buckets.itertuples(index=False):
        bid = f"bucket:{row.bucket_name}"
        ids.append(bid)
        labels.append(str(row.bucket_name))
        parents.append('root')
        values.append(int(row.count))

    # extension nodes under each bucket
    for row in df_counts.itertuples(index=False):
        bid = f"bucket:{row.bucket_name}"
        eid = f"{row.bucket_name}|{row.extension}"
        ids.append(eid)
        labels.append(str(row.extension))
        parents.append(bid)
        values.append(int(row.count))

    sunburst_fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total'
    ))
    sunburst_fig.update_layout(title='File types by bucket (sunburst)')
    sunburst_out = f"{out_prefix}_sunburst.html"
    sunburst_fig.write_html(sunburst_out)
    logger.info('Sunburst written to %s', sunburst_out)

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

    # Coerce size column to int64 (errors → 0) for byte-weighted charts
    df['size_bytes'] = pd.to_numeric(df.get('size', df.get('size_bytes', None)), errors='coerce').fillna(0).astype('int64')

    # Aggregate total bytes per extension
    total_by_extension_bytes = (
        df.groupby('extension', as_index=False)['size_bytes']
          .sum()
          .sort_values('size_bytes', ascending=False)
    )

    # Build Sankey labels and links: Total bytes -> extension
    labels_bytes = ['Total bytes'] + total_by_extension_bytes['extension'].tolist()
    source = []
    target = []
    value = []

    for i, row in enumerate(total_by_extension_bytes.itertuples(index=False)):
        source.append(0)           # from 'Total bytes' node
        target.append(1 + i)      # to each extension node
        value.append(int(row.size_bytes))

    sankey_bytes_fig = go.Figure(go.Sankey(
        node=dict(label=labels_bytes, pad=15, thickness=20),
        link=dict(source=source, target=target, value=value)
    ))
    sankey_bytes_fig.update_layout(
        title=f'File type breakdown by total bytes (Total -> extension) — total bytes = {int(df["size_bytes"].sum()):,}'
    )
    sankey_bytes_out = f"{out_prefix}_sankey_size.html"
    sankey_bytes_fig.write_html(sankey_bytes_out)
    logger.info('Sankey (by bytes) written to %s', sankey_bytes_out)

    # Sunburst by bytes: group by (bucket_name, extension) and sum bytes
    df_counts_bytes = (
        df.groupby(['bucket_name', 'extension'], as_index=False)['size_bytes']
          .sum()
    )

    # Coerce types to safe Python int / str
    df_counts_bytes = pd.DataFrame(df_counts_bytes)
    df_counts_bytes['size_bytes'] = df_counts_bytes['size_bytes'].astype('int64')
    df_counts_bytes['bucket_name'] = df_counts_bytes['bucket_name'].astype(str)
    df_counts_bytes['extension'] = df_counts_bytes['extension'].astype(str)

    # Build hierarchical ids, labels, parents, and values for sunburst (root -> bucket -> extension)
    labels = []
    ids = []
    parents = []
    values = []

    # root node
    ids.append('root')
    labels.append('All bytes')
    parents.append('')
    values.append(int(df_counts_bytes['size_bytes'].sum()))

    # bucket nodes (one per bucket)
    buckets_bytes = df_counts_bytes.groupby('bucket_name', as_index=False)['size_bytes'].sum()
    for row in buckets_bytes.itertuples(index=False):
        bid = f"bucket:{row.bucket_name}"
        ids.append(bid)
        labels.append(str(row.bucket_name))
        parents.append('root')
        values.append(int(row.size_bytes))

    # extension nodes under each bucket
    for row in df_counts_bytes.itertuples(index=False):
        bid = f"bucket:{row.bucket_name}"
        eid = f"{row.bucket_name}|{row.extension}"
        ids.append(eid)
        labels.append(str(row.extension))
        parents.append(bid)
        values.append(int(row.size_bytes))

    # Create sunburst. Add a hovertemplate to show bytes with thousands separators.
    sunburst_bytes_fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',
        hovertemplate='%{label}<br>Bytes: %{value:,}<extra></extra>'
    ))
    sunburst_bytes_fig.update_layout(
        title=f'File types by bucket (sunburst) — bytes as wedge size (total = {int(df_counts_bytes["size_bytes"].sum()):,})'
    )
    sunburst_bytes_out = f"{out_prefix}_sunburst_size.html"
    sunburst_bytes_fig.write_html(sunburst_bytes_out)
    logger.info('Sunburst (by bytes) written to %s', sunburst_bytes_out)


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
