# Frozen paper results

`2016/` and `2021/` contain the final 100-design artifacts used by the paper and revised supplementary material. They include the aggregate UCED performance table, fixed train/test split, GP metadata and models, ARD screening outputs, 20,000-point search, low-mismatch candidates, and six original-UCED revalidation records per year.

These directories are treated as immutable records. Use `reproduction/prepare_analysis.py` or `reproduction/reproduce_surrogate.sh` to create a writable copy under `experiment/`.

Large weekly UCED dispatch and flow files are excluded. Their aggregate outcomes are preserved in `final_performance_summary.csv`; selected original-UCED revalidations are preserved in `back_check_archive/`.
