# Publication checklist

Complete these items before pushing the repository publicly or creating the Zenodo record.

- [ ] Confirm that every file under `data/` may be redistributed publicly.
- [ ] Fill all `SOURCE TO CONFIRM` entries in `data/README.md` with citations or public URLs.
- [ ] Confirm that the MIT license is acceptable to all code contributors and the institution.
- [ ] Add the final paper title, full author list, DOI, and publication year to `CITATION.cff`.
- [ ] Create the GitHub repository `zoe667/uced-inference` and set the local remote.
- [ ] Run `python3 reproduction/verify_results.py` on a fresh clone.
- [ ] Run Julia installation on a clean depot and retain the generated `Manifest.toml`.
- [ ] Check the Gurobi smoke test on a licensed machine.
- [ ] Enable the repository in Zenodo before publishing the tagged GitHub release.
- [ ] Create a release tag and add the Zenodo DOI badge to `README.md`.
