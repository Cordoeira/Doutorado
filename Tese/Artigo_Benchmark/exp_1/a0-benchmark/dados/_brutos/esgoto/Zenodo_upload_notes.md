# Tracer curve dataset for longitudinal dispersion in urban sewer systems


## Description
60 tracer experiments in 37 different sewer reaches have been analyzed for longitudinal dispersion under dry weather flow conditions. Tracer (NaCl, Uranine, or Rhodamine WT) was injected upstream, conductivity or fluorescence was recorded at a downstream point, and further upstream point where available. 
Suitable equations were identified to predict reasonable dispersion coefficients in sewer
reaches with uniform geometry and stable flow conditions. 

Attached dataset to:
> J. Rieckermann, M. Neumann, C. Ort, J.L. Huisman, W. Gujer; Dispersion coefficients of sewers from tracer experiments. Water Sci Technol 1 September 2005; 52 (5): 123–133. doi: https://doi.org/10.2166/wst.2005.0124

Not all 60 experiments are included: IDs 56–59 have no preserved tracer curves. 56 datasets are included in total.

## Files

**`aaa_paper_std/`**: 81 tracer curve files, one folder per dataset ID (01-55, 60). Each folder has `{ID}_dn.dat` (downstream) and, where available, `{ID}_up.dat` (upstream). Columns: `time_s` and either `C_mScm` (conductivity, mS/cm) or `C_ppb` (fluorescence).

**`dataset_parameters.csv`**: One row per dataset:
- `mass_kg` — injection mass [kg]
- `L_m` — reach length [m]
- `Q_m3s` — discharge [m3/s]
- `v_init` — velocity [m/s]
- `Dx_init` — initial dispersion coefficient [m2/s]
- `cal_coeff` — calibration coefficient converting sensor signal to mass concentration (0.55 for mS/cm, 1000 for uS/cm)
- `fit_start_s` / `fit_end_s` — fitting window [s]

