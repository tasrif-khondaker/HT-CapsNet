import json, glob, statistics as st, sys, pathlib

dataset = sys.argv[1] if len(sys.argv)>1 else "CIFAR-100"
base = pathlib.Path("logs")/dataset
files = sorted(base.glob("seed-*/results.json"))
metrics = ["hier_acc", "hier_f1", "consistency", "exact_match"]

rows = []
for f in files:
    with open(f) as fh: rows.append(json.load(fh))

def mpm(values):
    mu = st.mean(values)
    sd = st.pstdev(values)  # population stdev over the seeds you ran
    return f"{mu:.2f} ± {sd:.2f}"

print(f"Dataset: {dataset}")
for m in metrics:
    vals = [r[m] for r in rows if m in r]
    if vals:
        print(f"{m}: {mpm(vals)}")
