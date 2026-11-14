#!/usr/bin/env python3
"""
nxmodel_pack.py -- turn Nexus native model FOLDERS (*.nxmodel) into single .nxm files the engine can open.

  python3 nxmodel_pack.py MyModel.remote.nxmodel                 -> MyModel.remote.nxm   (next to the folder)
  python3 nxmodel_pack.py A.nxmodel B.nxmodel --out D:\\models   -> D:\\models\\A.nxm, D:\\models\\B.nxm
  python3 nxmodel_pack.py *.nxmodel --delete                      -> remove each folder after it was packed successfully

Nothing is re-quantized: the weight files, manifest and metadata are copied into the .nxm exactly as the builder's own
"Single native file" mode lays them out (same header, manifest, file table and payload order).  The folder is never modified.
"""
import argparse, json, os, shutil, struct, sys
from pathlib import Path

NXM_MAGIC = b"NEXUS_NXM_V1\0\0\0\0"
NXM_HEADER_STRUCT = "<16sQQQ"


def fmt(n):
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or u == "TiB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024.0


def pack(package_dir: Path, nxm_path: Path):
    manifest_path = package_dir / "nexus_model.json"
    if not manifest_path.exists():
        raise SystemExit(f"{package_dir}: no nexus_model.json - not a Nexus native model folder")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["container"] = {
        "format": "nexus_native_single_file", "version": 1, "magic": "NEXUS_NXM_V1", "file": nxm_path.name,
        "manifest_embedded_after_header": True, "file_table_embedded_after_manifest": True,
        "payload_offsets_relative_to": "payload_start",
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")

    files = sorted(p for p in package_dir.rglob("*") if p.is_file())
    entries, offset, sizes = [], 0, {}
    for p in files:
        rel = str(p.relative_to(package_dir)).replace(os.sep, "/")
        size = len(manifest_bytes) if rel == "nexus_model.json" else p.stat().st_size
        kind = "manifest" if rel == "nexus_model.json" else "metadata" if rel.startswith("metadata/") else \
               "weights" if rel.startswith("weights_") and rel.endswith(".nxw") else "file"
        entries.append({"path": rel, "kind": kind, "offset": offset, "nbytes": size})
        sizes[p] = size; offset += size
    table_bytes = json.dumps({"format": "nexus_native_single_file_table", "version": 1, "file_count": len(entries),
                              "payload_bytes": offset, "files": entries}, indent=2).encode("utf-8")

    tmp = nxm_path.with_suffix(nxm_path.suffix + ".tmp")
    if tmp.exists(): tmp.unlink()
    total, done, last = max(1, offset), 0, -1
    with tmp.open("wb") as out:
        out.write(struct.pack(NXM_HEADER_STRUCT, NXM_MAGIC, len(manifest_bytes), len(table_bytes), 0))
        out.write(manifest_bytes); out.write(table_bytes)
        for p in files:
            rel = str(p.relative_to(package_dir)).replace(os.sep, "/")
            if rel == "nexus_model.json":
                out.write(manifest_bytes); done += len(manifest_bytes); continue
            with p.open("rb") as src:
                while True:
                    chunk = src.read(16 * 1024 * 1024)
                    if not chunk: break
                    out.write(chunk); done += len(chunk)
                    pct = done * 100 // total
                    if pct >= last + 5:
                        last = pct; print(f"  {pct}%  ({fmt(done)} of {fmt(total)})", flush=True)
    if nxm_path.exists(): nxm_path.unlink()
    tmp.rename(nxm_path)
    print(f"  wrote {nxm_path}  ({fmt(nxm_path.stat().st_size)})")


def main():
    ap = argparse.ArgumentParser(description="Pack Nexus .nxmodel folders into single .nxm files.")
    ap.add_argument("folders", nargs="+", help="one or more *.nxmodel folders")
    ap.add_argument("--out", help="folder for the .nxm files (default: next to each .nxmodel folder)")
    ap.add_argument("--delete", action="store_true", help="delete each folder after it was packed")
    a = ap.parse_args()
    for f in a.folders:
        d = Path(f)
        if not d.is_dir():
            print(f"skip {f}: not a folder"); continue
        name = d.name[:-len(".nxmodel")] if d.name.lower().endswith(".nxmodel") else d.name
        dest_dir = Path(a.out) if a.out else d.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"Packing {d} ...")
        pack(d, dest_dir / (name + ".nxm"))
        if a.delete:
            shutil.rmtree(d); print(f"  removed folder {d}")
    print("Done.")


if __name__ == "__main__":
    main()
