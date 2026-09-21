# Dataset placement

The benchmark datasets used by the clean experiments are included in this repository. The directory structure is:

```text
data/
├── AIOPS/
├── KPI/
├── NAB/
├── TODS/
├── WSD/
└── Yahoo/
```

The exact file layout is detected by `experiments/data_pipeline.py`. Before the final public release is frozen, verify the license and redistribution terms for every included dataset. If a dataset cannot be redistributed with the code, replace its files with the official download URL, version information, and preprocessing instructions.
