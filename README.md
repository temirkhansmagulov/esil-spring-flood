# Upper Esil spring floods: soil and land-cover controls

Code for the article *Modelling Soil and Land-Cover Controls on Spring Flood Runoff in
the Upper Esil Basin, Kazakhstan* (Bekmagambetov, Makhmut, Smagulov). It builds the
dataset in Section 7.1 and produces Tables 2–6 for the three research questions:

| RQ | Question | Method | Output |
|---|---|---|---|
| RQ1 | How do spring runoff coefficients differ among soil–land-cover classes? | SCS-CN on WorldCover × SoilGrids/HYSOGs, AMC adjustment, land-cover-adjusted Ks | Tables 3, 4 |
| RQ2 | How do degradation and recovery scenarios change spring peak, volume and baseflow? | SWAT+ scenario runs, paired Wilcoxon over 23 years | Table 5 |
| RQ3 | Which drivers explain observed spring discharge? | XGBoost (5 seeds) + SHAP vs climatology, linear regression and SWAT+ | Table 6 |

> **Status.** The code runs end to end on synthetic demo data (`--demo`). Results for
> the article require the real inputs listed below. Nothing in `demo/` is a research result.

## Repository layout

```
config.yaml               all settings: periods, splits, curve numbers, scenarios, paths
scripts/run_pipeline.py   command-line entry point
scripts/download_static.py  Copernicus DEM, ESA WorldCover and SoilGrids for the study box
src/esilflood/
  era5.py                 ERA5-Land download (CDS API), hourly -> daily, basin average
  discharge.py            Esil–Turgen discharge screening (gaps, ice days, mm/day)
  basin.py                basin polygon: load, or delineate from the DEM (pysheds)
  static_layers.py        common grid, USDA texture, Ks, HSG, curve numbers, SCS-CN (Eqs. 1–4)
  rq1.py                  runoff coefficients, AMC comparison, Ks sensitivity (Tables 3–4)
  rq2.py                  scenario land-cover maps, SWAT+ output reader, effects (Table 5)
  rq3.py                  lagged features, XGBoost, baselines, SHAP, Spearman test (Table 6)
  metrics.py              NSE, RMSE, KGE, PBIAS, Ep, Δtp, block bootstrap, Wilcoxon (Eqs. 5–8)
  report.py               Tables 2–6 as CSV + Markdown, figures
  synthetic.py            synthetic inputs for the demo
tests/                    unit tests (pytest)
data/raw, data/processed, data/swat   inputs (not tracked by git)
results/tables, results/figures       outputs
```

## Installation

```bash
git clone https://github.com/<user>/esil-spring-flood.git
cd esil-spring-flood
conda env create -f environment.yml && conda activate esil-flood
# or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
pytest -q                                   # 20 unit tests
python scripts/run_pipeline.py --demo       # full synthetic run, about 15 s
```

Tested with Python 3.13, NumPy 2.5, pandas 3.0, scikit-learn 1.9, XGBoost 3.4, SHAP 0.54,
rasterio 1.5, geopandas 1.2, xarray 2026.9 (see `requirements.txt`).

## Data

| Data | Source | Where it goes | How |
|---|---|---|---|
| Precipitation, snowmelt, SWE, 2-m temperature, soil moisture and temperature | ERA5-Land hourly, Copernicus CDS, DOI 10.24381/cds.e2161bac | `data/raw/era5_land/` | `python scripts/run_pipeline.py --steps download` (needs a CDS account and `~/.cdsapirc`; accept the ERA5-Land licence on the CDS site first) |
| Daily discharge, Esil–Turgen | Kazhydromet hydrological database, https://meteo.kazhydromet.kz/database_hydro/ | `data/raw/discharge/esil_turgen_daily.csv` | Export manually (see below) |
| Elevation | Copernicus DEM GLO-30 | `data/raw/copernicus_dem/` | `python scripts/download_static.py --only dem` |
| Land cover 2021 | ESA WorldCover v200 | `data/raw/worldcover/` | `python scripts/download_static.py --only worldcover` |
| Soil texture | SoilGrids 2.0 (sand, silt, clay; 0–30 cm) | `data/raw/soilgrids/` | `python scripts/download_static.py --only soilgrids` |
| Hydrologic soil groups (optional) | HYSOGs250m, ORNL DAAC, DOI 10.3334/ORNLDAAC/1566 | set `static.hysogs_path` | Earthdata login required |
| Curve numbers for cross-check (optional) | GCN250, DOI 10.6084/m9.figshare.7756202 | set `static.gcn250_path` | Figshare |
| Basin outline (optional) | your own polygon, or HydroBASINS/HydroATLAS | `data/raw/basin/upper_esil_basin.gpkg` | otherwise delineated from the DEM |

**Discharge data.** Save the Esil–Turgen daily series as a CSV with columns
`date,q_m3s,ice`, where `ice` is 1 for days flagged as ice-affected and 0 otherwise:

```
date,q_m3s,ice
2001-09-01,4.12,0
2001-09-02,4.05,0
```

Gaps of up to 7 days are interpolated, longer gaps and ice days are removed, and the counts
go to Table 2. Before running, set `basin.gauge_lon` and `basin.gauge_lat` from the station
passport and check `basin.era5_area`.

## Running

```bash
python scripts/run_pipeline.py --steps download          # ERA5-Land (slow; runs once)
python scripts/run_pipeline.py --steps prepare table2 rq1 scenarios
# ... build and run SWAT+ (next section) ...
python scripts/run_pipeline.py --steps rq2 rq3
```

`prepare` writes `data/processed/forcing_daily.csv`, `discharge_clean.csv`, the static
grids (`static/landcover.tif`, `hsg.tif`, `ks_cm_h.tif`, `cn2.tif`) and
`class_statistics.csv`. Each later step writes its tables to `results/tables/`.

## SWAT+ (RQ2)

SWAT+ is run outside Python:

1. `scenarios` writes `data/processed/scenarios/landcover_{baseline,degradation,recovery}.tif`.
2. In QGIS with QSWAT+, build the baseline project from the Copernicus DEM, the baseline land
   cover, the SoilGrids-based soil map and ERA5-Land weather, with the outlet at Turgen.
3. Calibrate on 2001/02–2012/13 and validate on 2013/14–2016/17 (SWAT+ Toolbox or SWAT-CUP).
4. Copy the calibrated project twice, replace the land-use map with each scenario raster and
   re-run with identical weather and parameters.
5. Put each run's `channel_sd_day.txt` in `data/swat/<scenario>/` and set
   `swat.outlet_gis_id` to the outlet channel. A CSV with `date,q_m3s` also works.
6. For the RQ3 agreement test, perturb snowmelt, soil moisture and soil temperature inputs by
   one standard deviation in SWAT+, and save the change in spring peak as
   `data/swat/perturbation_sensitivity.csv` with columns `variable,delta_peak_pct`.

`rq2` reports SWAT+ performance per subset (`rq2_swat_performance.csv`) and Table 5. If
the validation NSE is below 0.50, report the scenario results as relative changes only.

## Outputs

| File | Content |
|---|---|
| `table2_final_dataset.md/.csv` | Final dataset, cleaned/removed days, split, spring quantiles |
| `table3_runoff_coefficients.md/.csv` | Area share, HSG, Ks, CN_II, C [95% CI], AMC comparison |
| `table4_ks_sensitivity.md/.csv` | ΔC with land-cover-adjusted Ks |
| `table5_scenarios.md/.csv` | Δ peak, volume, baseflow [95% CI], Wilcoxon p |
| `rq2_scenarios_by_peak_magnitude.csv` | Table 5 for the 3 largest and 3 smallest peak years |
| `table6_model_comparison.md/.csv` | NSE, KGE, PBIAS, Ep, Δtp for all models |
| `rq3_shap_ranking.md/.csv` | SHAP ranking of drivers |
| `figures/rq3_test_hydrographs.png`, `rq3_shap_ranking.png` | Test-year hydrographs, SHAP bars |

## Methodological choices made in the code

These follow Sections 6–7 of the article where the text is specific. Where it is not, the
code makes the choices below; state them in the Methods if you keep them.

- **Runoff coefficient (RQ1).** C = ΣQ/ΣP over all spring days (1 March–31 May), with
  P = precipitation + snowmelt from the basin-mean ERA5-Land series and Q from SCS-CN for
  every curve number in the class. The 95% CI resamples whole hydrological years.
- **Antecedent moisture.** AMC I/II/III from the 33rd/67th percentiles of ERA5-Land
  layer-1 soil moisture in spring; frozen topsoil (layer-1 temperature < 0 °C) is AMC III.
  The AMC comparison pairs basin-wide C per year (n = 23).
- **Curve numbers.** WorldCover classes are mapped to the nearest NRCS TR-55 cover types
  in `config.yaml`. Edit them to match the table you cite.
- **HSG.** From HYSOGs250m if provided (dual groups taken as D), otherwise from SoilGrids
  texture → Rawls Ks → HSG thresholds. The Ks sensitivity (Table 4) needs SoilGrids.
- **Scenarios.** Random, reproducible (seed 42) pixel conversions: 30% of grassland →
  cropland; 30% of cropland → grassland plus 5% of cropland → trees.
- **RQ3 features.** Basin-mean precipitation, snowmelt, SWE, 2-m temperature, layer-1 soil
  moisture and temperature at lags 0–7 days, plus day of year. XGBoost is early-stopped on
  the validation years. Ep and Δtp are averaged over test years; the Wilcoxon test pairs
  per-year RMSE.

## Reproducibility

Before submitting, tag a release on GitHub and archive it on Zenodo to get a DOI for the
Code Availability statement, and record the exact environment with
`pip freeze > requirements-lock.txt`.

## Licence

MIT (see `LICENSE`). Data sets keep their own licences (ERA5-Land: Copernicus licence;
WorldCover and SoilGrids: CC BY 4.0; Copernicus DEM GLO-30: free licence; Kazhydromet data:
terms of the provider).
