Trained artifacts are **not** committed by default (see `.gitignore`).

After `python -m ev_charging train --config configs/default.yaml --root .` you should see:

- `mlp.pt`
- `preprocessor.joblib`
- `baselines.joblib`

For Docker or the API, bind-mount or copy this directory into the container.
