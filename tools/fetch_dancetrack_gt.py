"""
Fetch only the ground-truth annotation files (<seq>/gt/gt.txt and seqinfo.ini)
from the official DanceTrack archives on Hugging Face, without downloading the
multi-GB image archives.

How: a zip's central directory sits at the end of the file, so we open the
remote archive through HTTP Range requests and read just the directory plus the
bytes of the requested members. zipfile verifies each member's CRC-32.

Source: https://huggingface.co/datasets/noahcao/dancetrack (train1.zip, train2.zip)

Example:
    python tools/fetch_dancetrack_gt.py --archives train1.zip --max_seqs 2 \
        --out datasets_gru/dancetrack_gt
Output layout: <out>/train/<seq>/gt/gt.txt, <out>/train/<seq>/seqinfo.ini,
plus <out>/train/FETCH_MANIFEST.json (source URL, archive, CRC, sizes).
"""
import argparse
import io
import json
import os
import urllib.request
import zipfile
import zlib

BASE = "https://huggingface.co/datasets/noahcao/dancetrack/resolve/main/"
UA = {"User-Agent": "python-urllib (dancetrack-gt-fetch)"}


class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP Range requests."""

    def __init__(self, url):
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req) as r:
            self.url = r.geturl()                 # final URL after redirects
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.bytes_fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers=dict(UA, Range="bytes={}-{}".format(self.pos, end)))
        with urllib.request.urlopen(req) as r:
            if r.status != 206:
                raise IOError("server ignored Range request (status {})".format(r.status))
            data = r.read()
        self.pos += len(data)
        self.bytes_fetched += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives", nargs="+", default=["train1.zip", "train2.zip"],
                    help="train archives only (train1.zip / train2.zip)")
    ap.add_argument("--seqs", nargs="*", default=None, help="explicit sequence names (default: all, sorted)")
    ap.add_argument("--max_seqs", type=int, default=None, help="first N sequences (sorted) across archives")
    ap.add_argument("--out", default="datasets_gru/dancetrack_gt")
    ap.add_argument("--list", action="store_true", help="only enumerate sequences in the archives")
    args = ap.parse_args()
    for a in args.archives:
        if not a.startswith("train"):
            raise SystemExit("refusing non-train archive {}: GRU training uses the train split only".format(a))

    out_split = os.path.join(args.out, "train")
    os.makedirs(out_split, exist_ok=True)
    manifest_path = os.path.join(out_split, "FETCH_MANIFEST.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    remaining = args.max_seqs
    seen = {}                                     # seq -> archive (detect duplicates across archives)
    for arc in args.archives:
        rf = HTTPRangeFile(BASE + arc)
        zf = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 16))
        names = zf.namelist()
        all_seqs = sorted({n.split("/")[-3] for n in names if n.endswith("gt/gt.txt")})
        for sq in all_seqs:
            if sq in seen:
                print("WARNING: {} appears in both {} and {}".format(sq, seen[sq], arc))
            seen.setdefault(sq, arc)
        print("{}: {} sequences with gt/gt.txt ({:.0f} MB archive): {}".format(
            arc, len(all_seqs), rf.size / 2**20, " ".join(all_seqs)))
        if args.list:
            continue
        seqs = [s for s in all_seqs if s in set(args.seqs)] if args.seqs else all_seqs
        if remaining is not None:
            seqs = seqs[:remaining]
        n_fetched = n_skipped = 0
        for seq in seqs:
            for suffix in ("gt/gt.txt", "seqinfo.ini"):
                cands = [n for n in names if n.endswith("/{}/{}".format(seq, suffix)) or n == "{}/{}".format(seq, suffix)]
                if len(cands) != 1:
                    raise SystemExit("{}: expected exactly one member for {}/{}, found {}".format(arc, seq, suffix, cands))
                member = cands[0]
                info = zf.getinfo(member)
                dst = os.path.join(out_split, seq, suffix)
                if os.path.isfile(dst) and os.path.getsize(dst) == info.file_size and \
                        zlib.crc32(open(dst, "rb").read()) & 0xFFFFFFFF == info.CRC:
                    n_skipped += 1                # already present and intact: no re-download
                else:
                    data = zf.read(member)        # raises BadZipFile on CRC mismatch
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with open(dst, "wb") as f:
                        f.write(data)
                    n_fetched += 1
                manifest["{}/{}".format(seq, suffix)] = {
                    "source": BASE + arc, "archive": arc, "member": member,
                    "crc32": "{:08x}".format(info.CRC), "bytes": info.file_size}
        print("{}: {} sequences | {} files fetched, {} already present (CRC ok) | {:.2f} MB transferred".format(
            arc, len(seqs), n_fetched, n_skipped, rf.bytes_fetched / 2**20))
        if remaining is not None:
            remaining -= len(seqs)
            if remaining <= 0:
                break
    if not args.list:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
