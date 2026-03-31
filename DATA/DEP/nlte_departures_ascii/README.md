Place generated NLTE ASCII departure files here for the Fe NLTE examples.

Expected filename pattern:

`000001_<model_stem>_abu+7.500.dat`

Example `model_stem`:

`p5000_g+4.0_m0.0_t01_st_z-0.50_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00`

You can populate this directory with:

`python3 scripts/export_nlte_grid_ascii.py <grid_file> <aux_file> -o DATA/DEP/nlte_departures_ascii`
