import re
import ast
import argparse
import os
import csv


def extract(file_path):
    """Extract Python-style dicts from a logfile and classify them into
    training-step dicts (contain 'loss'), evaluation dicts (contain 'eval_loss'),
    and overall training summary dicts (contain 'train_runtime' or 'train_loss').
    Returns three lists of dicts.
    """
    with open(file_path, "r", encoding="utf-8") as fh:
        content = fh.read()

    # match single-line brace-delimited dicts (no nesting expected in these logs)
    pattern = re.compile(r"\{[^{}]*\}")

    train_records = []
    eval_records = []
    overall_records = []

    for m in pattern.finditer(content):
        s = m.group(0)
        lineno = content.count("\n", 0, m.start()) + 1

        # replace bare nan (and variants) with None so ast.literal_eval can parse it reliably
        s_fixed = re.sub(r"(?<=:)\s*nan(?=[,}])", "None", s, flags=re.IGNORECASE)

        try:
            d = ast.literal_eval(s_fixed)
        except Exception:
            # Skip unparseable fragments
            continue

        if not isinstance(d, dict):
            continue

        # add metadata
        d["_line"] = lineno

        if "eval_loss" in d:
            eval_records.append(d)
        elif "loss" in d and "eval_loss" not in d:
            train_records.append(d)
        elif any(k.startswith("train_") for k in d.keys()) or "train_loss" in d:
            overall_records.append(d)
        else:
            # fallback: classify as overall
            overall_records.append(d)

    return train_records, eval_records, overall_records


def write_csv(dicts, outpath):
    # ensure parent dir exists
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    if not dicts:
        # create an empty file
        open(outpath, "w", encoding="utf-8").close()
        return

    # union of all keys, keep deterministic order with '_line' last
    keys = sorted({k for d in dicts for k in d.keys() if k != "_line"})
    keys.append("_line")

    with open(outpath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for d in dicts:
            row = {k: d.get(k, "") for k in keys}
            writer.writerow(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None, help="the filepath to the file to parse")
    ap.add_argument("--output", default=None, help="output prefix or directory")
    args = ap.parse_args()

    if not args.file:
        raise SystemExit("Please provide --file PATH to the logfile")

    train, evals, overall = extract(args.file)

    base = os.path.splitext(os.path.basename(args.file))[0]

    if args.output:
        # if output is a directory, use base name as prefix
        if os.path.isdir(args.output):
            prefix = os.path.join(args.output, base)
        else:
            # use given output as prefix (strip extension if provided)
            prefix = os.path.splitext(args.output)[0]
    else:
        prefix = os.path.join(os.path.dirname(args.file) or ".", base)

    train_out = prefix + "_train.csv"
    eval_out = prefix + "_eval.csv"
    overall_out = prefix + "_overall.csv"

    write_csv(train, train_out)
    write_csv(evals, eval_out)
    write_csv(overall, overall_out)

    print(f"Wrote: {train_out} ({len(train)}), {eval_out} ({len(evals)}), {overall_out} ({len(overall)})")
