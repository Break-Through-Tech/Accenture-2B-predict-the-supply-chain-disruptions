# Supply Chain Risk Radar (Streamlit)

Dashboard around your disruption Random Forest: enter a date, industry, origin, route and US destination;
it scores next-week disruption risk, flags risky drivers on a US map and suggests workarounds.

## 1. Run locally
```bash
pip install -r requirements.txt
python train_model.py --csv "C:\Users\ngair\Downloads\model_ready_dataset.csv"
streamlit run app.py
```
`train_model.py` writes `artifacts/` (model, thresholds, reference data). Add `--drop-time-features` to
exclude `year` / `weeks_since_start` (recommended, see "About the model" tab in the app).

## 2. Deploy (Streamlit Community Cloud, free)
1. Create a GitHub repo and push this folder **including `artifacts/`** (about 10-50 MB, under GitHub's 100 MB file limit).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repo, branch `main`, main file `app.py`, then **Deploy**.
If your CSV is private, keep the repo private; Community Cloud supports private repos.

## Files
- `app.py` dashboard UI · `engine.py` scoring, flags, what-ifs · `config.py` ports, routes, industry profiles, workaround playbook (edit freely)
- `train_model.py` reproduces your notebook model and exports artifacts
