# Cross-year parameter concentration main figure

This folder contains the cross-year figure comparing the three active
parameters retained in both years at the fixed low-mismatch tolerance
`epsilon = 10%`.

The local reporting CSV contains summaries calculated from
`results/2016/dense_surrogate_search.csv` and
`results/2021/dense_surrogate_search.csv`.
The epsilon-10 sets contain 260 points in 2016 and 3,420 points in 2021. Each panel shows the
median, IQR, descriptive 5th–95th
percentile range, surrogate-selected minimum, temporal median connector, and
the signed change in the median from 2016 to 2021.

The main title is intentionally omitted from the figure canvas.

Run from the repository root:

```bash
MPLBACKEND=Agg python3 paper/figures/cross_year_parameter_concentration/generate_cross_year_parameter_concentration.py
```

Outputs:

- `fig_cross_year_parameter_concentration_eps10.pdf`
- `fig_cross_year_parameter_concentration_eps10.png`
