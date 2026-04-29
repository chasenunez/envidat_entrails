# Entrails

Crawls EnviDat's S3 buckets, builds a CSV of every file's metadata, and renders sunburst + Sankey diagrams of what's in there. The point: see at a glance which file types and which buckets are eating the storage.

## What it does

1. **Fetch.** Walks several S3-style listing endpoints, including peeking inside ZIP archives via HTTP range requests, and writes a unified CSV.
2. **Visualize.** Reads that CSV and produces interactive Plotly charts: Sankey by count and by bytes, sunburst by count and by bytes.

## Run

```bash
pip install requests pandas plotly

# fetch listings
python3 entrails.py fetch --out all_s3_files.csv

# visualize
python3 entrails.py visualize --csv all_s3_files.csv --out-prefix envidat_viz

# both at once
python3 entrails.py run-all --out all_s3_files.csv --out-prefix envidat_viz
```

Outputs land in the working directory:

| File | What |
|---|---|
| `all_s3_files.csv` | One row per file (or per ZIP-inner entry) |
| `envidat_viz_sankey.html` | Sankey by count |
| `envidat_viz_sankey_size.html` | Sankey by bytes |
| `envidat_viz_sunburst.html` | Bucket → extension by count |
| `envidat_viz_sunburst_size.html` | Bucket → extension by bytes |

## Buckets crawled

Five SWITCH Cloud buckets that make up [EnviDat](https://www.envidat.ch/):

| Bucket | Description |
|---|---|
| `envidat-doi` | Published EnviDat datasets with DOIs |
| `envicloud` | Internal staging and mirrored datasets |
| `edna` | Elevation-derived hydrological data |
| `pointclouds` | Drone-derived point clouds |
| `drone-data` | Drone-derived imagery |

CHELSA (`chelsav1`, `chelsav2`) is excluded — too big to be useful in a count-or-bytes overview.

## Filtering

Two rules apply during the fetch step (rows are dropped before the CSV is written):

1. Keys containing `envidat.1` are skipped. Those are the DOI datasets that contain thousands of `.raw` files; including them buries everything else.
2. In the `envidat-doi` bucket only, `.html` / `.json` / `.xml` files are skipped — they're machine-to-machine metadata, not dataset content.

## Peeking inside ZIPs

Lots of EnviDat files are `.zip`. Rather than download them, the script fetches the ZIP's End-of-Central-Directory and Central Directory via HTTP range requests, builds a tiny in-memory ZIP from those bytes, and reads the entry list from `zipfile.ZipFile`. The result is one extra CSV row per inner entry, with `key` set to `archive.zip::inner/path.csv` and `size` set to the compressed size.

The trick came from [this StackOverflow answer](https://stackoverflow.com/questions/51351000/read-zip-files-from-s3-without-downloading-the-entire-file) (Janaka Bandara) — much kinder to your bandwidth than downloading the archive.

### Caveats

- Needs the server to honour `Range:` headers. If it returns the full file (HTTP 200), the script gives up on archives larger than 50 MiB rather than swallow the bandwidth.
- ZIP64 and exotic ZIP layouts where EOCD isn't in the last ~64 KiB are skipped (with a warning).
- Inner-entry sizes are **compressed** sizes. Switching to uncompressed (`file_size`) is a one-line change.
- A ZIP with thousands of entries grows the CSV proportionally. Use `--max-pages` for testing.
- The script only lists filenames and sizes. It does not extract or execute anything.

## Logs

After a fetch you'll see something like:

```
2025-10-23 10:05:12 INFO: Got 1000 Contents entries (page 1)
...
2025-10-23 10:12:43 INFO: Skipped 2345 objects containing 'envidat.1' for bucket envidat-doi
2025-10-23 10:12:43 INFO: Skipped 412 metadata files (.html/.json/.xml) in envidat-doi
2025-10-23 10:12:43 INFO: Finished bucket: https://os.zhdk.cloud.switch.ch/envidat-doi/ (pages fetched=27)
```

## License

MIT — use, remix, or extend freely.
