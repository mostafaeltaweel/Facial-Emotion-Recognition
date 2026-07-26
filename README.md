# Facial Emotion Recognition — Camera App

## 1. Put the supplied files in this folder

Copy these two files beside `app.py`:

- `C:\Users\Admin\Downloads\best_model_combined (4).pth` → rename to `best_model_combined.pth`
- `C:\Users\Admin\Downloads\temperature.json` → keep this exact name

`state.db` is not needed for running the model.

## 2. Run locally (recommended first)

In PowerShell, inside this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the `Local URL` shown by Streamlit (normally `http://localhost:8501`), click **START**, then allow camera access in the browser.

## 3. Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload the files in this folder (including the renamed model and `temperature.json`). The model is ~48 MB, below GitHub's 100 MB file limit.
2. At [share.streamlit.io](https://share.streamlit.io), deploy the repository with main file `app.py`.
3. Use the generated HTTPS URL; camera permission is available there. Do not use an HTTP public URL, because browsers block camera access on insecure sites.

## Camera troubleshooting

- Use Chrome or Edge, press the lock/camera icon by the address bar, and set Camera to **Allow**.
- Close Zoom, Teams, or any other program using the camera.
- The first model load can take 20–60 seconds on CPU.
- If it is slow, set `DETECT_EVERY_N_FRAMES = 10` in `app.py`.
- If `load_state_dict` reports missing/unexpected keys, share the complete error: it means the checkpoint architecture differs from the combined model expected here.
