# Data documentation

This directory contains the model inputs and observed monthly targets used by the released 2016 and 2021 experiments. The data are separate from the MIT-licensed source code.

## Directory roles

- `ne_<year>_maininput/`: hourly UCED inputs for generators, availability, demand, fuel costs, network limits, interregional contract schedules, reserves, heating, and model constants.
- `ne_<year>_PriorityMLT/`: non-served-energy adjustment used by the PriorityMLT scenario.
- `hist_data_<year>/`: observed monthly coal, wind, solar, and interregional-exchange quantities used in the mismatch calculation.

## Core fields and units

| File | Main content | Typical units |
|---|---|---|
| `Generators_data.csv` | unit type, region, capacity, costs, minimum output, ramping and UC times | MW, MWh, CNY/MWh, hours, fractions |
| `Generators_variability.csv` | hourly generator availability factors | fraction |
| `Load_data.csv` | hourly zonal demand | MW |
| `Fuels_data.csv` | hourly fuel costs and emissions factors | model input units documented by column |
| `Network_forward.csv`, `Network_reverse.csv` | directed network definitions and transfer limits | MW |
| `Transmission_MLT.csv` | hourly interregional contract schedule | MW |
| `operating_reserve.csv` | reserve requirements | fraction |
| `heating_mw.csv` | heating-season requirements | MW |
| `hist_*_monthly.csv` | observed monthly targets | source-series units shown by column |

## Provenance and redistribution

**SOURCE TO CONFIRM BEFORE PUBLICATION.** Add the formal source, access date, transformation description, and redistribution permission for each of the following groups:

1. Generator identities, capacities, technologies, retrofit classifications, and operating assumptions.
2. Hourly demand and renewable-availability profiles.
3. Fuel-cost series.
4. Network and MLT contract series.
5. Monthly observed generation and exchange targets.

If any input cannot be redistributed, remove it from the public repository and provide a documented acquisition/transformation procedure or a redistributable synthetic fixture. The archived aggregate results can still support the solver-free verification and surrogate/reporting checks.

## Excluded files

Known backup and superseded inputs (`*_wrong*`, `Generators_data_forMLT.csv`, notes, and `.DS_Store`) were intentionally excluded from this release candidate.
