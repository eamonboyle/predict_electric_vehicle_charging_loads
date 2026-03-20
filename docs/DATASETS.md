# Datasets

All CSVs use `;` as the separator and Norwegian decimal commas in numeric fields where noted.

## Core (session + traffic model)

| File | Role |
|------|------|
| `datasets/EV charging reports.csv` | Charging sessions: plug times, `El_kWh` (target), duration, user type, calendar fields. |
| `datasets/Local traffic distribution.csv` | Hourly traffic counts at five locations; merged to sessions on floored plug-in hour (`Date_from`). |

## Optional (exploration and extra features)

| File | Role |
|------|------|
| `datasets/AMS data from garage Bl2.csv` | Hourly AMS energy and synthetic charger scenarios for one garage. Load with `ev_charging.extra_data.load_ams_garage` for analysis. |
| `datasets/Hourly EV loads - Aggregated private.csv` | Hourly synthetic/flex loads for the private fleet. Can be merged into the session table when `use_hourly_private_features: true` and `data.hourly_private_csv` is set in config. |
| `datasets/Hourly EV loads - Aggregated shared.csv` | Same structure for shared users. |
| `datasets/Hourly EV loads - Per user.csv` | Finer-grained series; suitable for future per-user or sequence models, not used in the default pipeline. |

## Redistribution

If you publish this repository, confirm you have the right to redistribute the CSVs or replace them with a download script and document provenance.
