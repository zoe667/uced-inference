# Main-paper text sync for the final 100-run results

Reference manuscript: `Methodological_Framework_Paper_before_submission.pdf`.

Authoritative result folders:

- `experiment/runs_2016_100`
- `experiment/runs_2021_100`

Both years use 100 UCED evaluations, split into 80 development observations and 20 independent held-out observations.

## 1. Surrogate performance

Replace the existing performance paragraph with:

> Mean five-fold pooled cross-validated R² values are 0.9988 for 2016 and 0.9853 for 2021. On the independent held-out UCED evaluations, the corresponding R² values are 0.9987 and 0.9933, with root-mean-square errors of 1.08×10⁻⁴ and 1.42×10⁻⁴, respectively, on the original L(θ) scale (n = 20 for each year).

The final kernels are Matérn 5/2 ARD for 2016 and Matérn 3/2 ARD for 2021.

The methods sentence saying that the GP specification is selected by cross-validation should be revised: the saved final workflow uses the year-specific fixed kernel and applies five-fold CV and ARD screening within the development set.

## 2. Retained dimensions

The 2016 reduced surrogate retains three dimensions: the MLT band and the non-retrofitted CHP minimum-output parameters for the 300–660 MW and 0–300 MW classes.

The 2021 reduced surrogate retains six dimensions: those same three dimensions, retrofitted-coal minimum output for the 300–660 MW and 0–300 MW classes, and retrofitted-coal minimum up/down time for the 0–300 MW class.

The current main-paper sentence describing five 2021 dimensions must therefore be replaced.

## 3. Full-versus-reduced comparison

Replace the existing dimensional-reduction paragraph with:

> Using the same held-out observations for the full and reduced fits, test R² changes from 0.9979 to 0.9987 in 2016 and from 0.9936 to 0.9933 in 2021. The corresponding RMSE changes from 1.39×10⁻⁴ to 1.08×10⁻⁴ and from 1.39×10⁻⁴ to 1.42×10⁻⁴, respectively. The reduced surrogates therefore preserve held-out predictive performance while providing lower-dimensional representations.

Do not state that RMSE declines in both years: the 2021 change is a small increase.

## 4. Six-point original-UCED revalidation

Replace the existing revalidation paragraph with:

> To verify that the reduced-GP low-mismatch region corresponds to low mismatch in the original forward model, six points from S₁₀% are re-evaluated using the full UCED formulation for each year. The resulting original-UCED losses range from 0.01009 to 0.01119 in 2016 and from 0.007065 to 0.007624 in 2021. The median absolute relative discrepancy between reduced-GP predictions and revalidated losses is 1.97% and 3.41%, respectively. These results indicate that the surrogate reliably locates low-mismatch regions, although local approximation error can affect the exact ranking of nearby candidates.

At the exact rounded parameters passed to UCED, the GP overpredicts the loss at all six points in both years.

## 5. Representative point and Fig. 3 discussion

Replace the aggregate-loss sentence with:

> The aggregate loss falls from 0.25253 to 0.01109 in 2016 and from 0.18548 to 0.007161 in 2021.

The qualitative explanation remains valid. The final NRMSE changes are:

| Target | 2016 | 2021 |
|---|---:|---:|
| Coal | −57.6% | −44.8% |
| Wind | −13.1% | +1.5% |
| Solar | −9.7% | +0.4% |
| MLT exchange | −92.1% | −91.2% |

The Fig. 3 caption remains valid.

## 6. Parameter concentration

Replace the normalized-IQR paragraph with:

> The physical MLT contract band is substantially more concentrated relative to its ex ante range than the two non-retrofitted CHP minimum-output dimensions. Its normalized IQR is 0.039 in 2016 and 0.095 in 2021, compared with 0.192 and 0.473 for the 300–660 MW CHP parameter and 0.208 and 0.495 for the 0–300 MW parameter, respectively. The CHP dimensions therefore remain compatible with substantially broader portions of their tested domains, even though both are retained by the ARD screening step. This contrast illustrates that predictive relevance to the mismatch surface does not imply tight recoverability from the available quantity observations.

Replace the cross-year medians with:

- MLT band: 0.0601 in 2016 and 0.0805 in 2021; change ≈ +0.020.
- CHP minimum output, 300–660 MW: 0.6773 and 0.5276; change ≈ −0.150.
- CHP minimum output, 0–300 MW: 0.6690 and 0.5298; change ≈ −0.139.
- The 2021 surrogate-selected MLT minimum is 0.0589, rather than 0.050.

For tolerance sensitivity, replace the 2021 MLT sentence with:

> In 2021, the normalized IQR of the MLT band increases from 0.047 at ε = 5% to 0.242 at ε = 30%, whereas those of the two shared CHP dimensions remain near 0.44–0.50 across the examined tolerances.

The Fig. 4 caption remains valid; its three numerical delta labels have been updated in the figure.

## 7. Fig. 2 caption

If the four-panel combined figure is used, replace the old two-panel caption with:

> Held-out surrogate validation and retained parameter dimensions. Panels (a) and (b) compare the mismatch obtained from an original UCED evaluation with the corresponding reduced-GP prediction for observations excluded from model development; dashed lines denote perfect agreement and shaded bands denote ±5%. Panels (c) and (d) summarize the fold-wise ARD screen. Filled markers denote retained dimensions, open markers denote excluded dimensions, and the shaded area marks the four-or-five upper-bound hits that trigger exclusion. R and CHP denote retrofitted coal and non-retrofitted CHP; L, M, and S denote the 660–1000, 300–660, and 0–300 MW classes.

## 8. Final ARD robustness results

The completed `04b` outputs change the earlier robustness interpretation:

- 2016: `FAIL_REVIEW`. The three baseline-retained parameters remain retained at 0.5×, 1×, and 2× ceilings, but non-retrofitted CHP minimum up/down time for the 660–1000 MW class changes from excluded to retained at the doubled ceiling. The complete retain/drop set is therefore ceiling-sensitive.
- 2021: `PASS`. The six-parameter baseline retained set and all exclusion decisions remain unchanged across the three tested ceilings.
- Cross-fitted reduced-model RMSE decreases by 20.74% in 2016 and 8.26% in 2021. Cross-fitted R² changes from 0.9981 to 0.9988 and from 0.9790 to 0.9824, respectively.

Suggested main-paper wording:

> Alternative length-scale ceilings and a fully cross-fitted selection check provide additional evidence on ARD stability. The 2021 retain/drop set is unchanged across the tested ceilings. In 2016, the three baseline-retained dimensions remain retained, but one excluded CHP duration dimension becomes retained when the ceiling is doubled. Cross-fitted reduced models improve aggregate out-of-fold RMSE in both years. Full fold-level results are reported in the Supplementary Material.

Do not state that both complete retained sets are invariant to the alternative ceilings. The baseline production GPs and all downstream region and revalidation results remain unchanged by this diagnostic.

## 9. Other draft cleanup

- Remove the unresolved bracketed phrase `[OR: in the Supplemental Material]`.
- Keep the 100-point design and 80/20 split.
- Do not reuse the old statement that 2016 Interior 2 lies outside S₁₀%; all twelve final revalidation points satisfy their year-specific final cutoffs.
- Replace “original-UCED admissibility and revalidation” with “original-UCED revalidation,” because the final analysis does not define an admissibility classifier.
- Replace “UCED-evaluated near-optimal candidates” with “revalidated low-mismatch candidates.”
- The claim that alternative target weights were examined still needs either matching final-run evidence or deletion; the `_100` results inspected here do not establish that robustness result.
