# Multimodal Fake News Detector: Beginner Setup Guide

This guide takes you from an empty folder to a running app. Every step says what to type and what you should see. Commands are for **Windows 11 + VS Code + PowerShell** (Mac/Linux differences are noted).

**Time needed:** about 30–45 minutes, mostly waiting for downloads.
**Internet needed:** during installation, and once when the image model downloads. After that, the detectors run offline.

---

## What is in this folder

```
fake-news-detector/
├── streamlit_app.py            <- the whole app
├── requirements.txt            <- Python packages to install
├── packages.txt                <- system programs (used by Streamlit Cloud; see Step 2 for Windows)
├── .gitignore                  <- keeps secrets and venv out of GitHub
├── .streamlit/
│   └── secrets.toml.example    <- template for optional API keys
└── models/
    └── README.txt              <- where your trained text models go (optional)
```

---

## Step 1: Check Python and open the folder in VS Code

1. Open VS Code → **File → Open Folder…** → choose the `fake-news-detector` folder.
2. Open the terminal: **Terminal → New Terminal** (it opens at the bottom, in PowerShell).
3. Check your Python version:

```powershell
python --version
```

✅ You should see `Python 3.12.x` (3.10–3.12 all work). If you see "not recognized", install Python from python.org and tick **Add Python to PATH**.

---

## Step 2: Install the two Windows programs the app calls

`packages.txt` lists three Linux packages. That file only matters on Streamlit Cloud. On Windows you install the equivalents yourself:

| Needed for | What to install | How |
|---|---|---|
| Reading text from screenshots (OCR) | **Tesseract** | Download the Windows installer from the "UB Mannheim Tesseract" page on GitHub. Run it and keep the **default install folder** (`C:\Program Files\Tesseract-OCR`). The app finds it automatically. |
| Reading video and audio files | **ffmpeg** | In PowerShell run: `winget install Gyan.FFmpeg` then **close and reopen VS Code** |
| Audio file reading (`libsndfile`) | Nothing | Already bundled inside the `soundfile` Python package on Windows |

Check both:

```powershell
tesseract --version
ffmpeg -version
```

✅ Each prints a version number. If `tesseract` says "not recognized" but you installed it in the default folder, that is fine: the app does not need PATH for it.

> **Skipping this step is OK.** The app will just show a yellow warning ("OCR unavailable") and carry on.

---

## Step 3: Create a virtual environment (a private box for this project's packages)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

✅ Your prompt now starts with `(venv)`.

**If PowerShell says "running scripts is disabled":** run this once, then repeat the activate command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

*(Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`.)*

If VS Code asks "We noticed a new environment… use it?", click **Yes**.

---

## Step 4: Install the packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

⏳ This takes **5–20 minutes** and downloads a few GB (PyTorch is the big one). Wait for the prompt to come back.

✅ The last line should say `Successfully installed …`.

**If it fails**, read the last red lines:

* Names `jabberjay` or `fast-c2pa-python`: those two are optional features. Open `requirements.txt`, put a `#` in front of the offending line, run `pip install -r requirements.txt` again, then try `pip install jabberjay` (or the other) on its own. The app works without them; that detector is simply skipped.
* "No matching distribution for torch": confirm `python --version` is 3.10–3.12 (64-bit Python).

---

## Step 5 (optional): Add your trained text models

The app has two local text detectors. **Without them the app still runs**; text ML is skipped with a warning.

Put these in the `models/` folder:

| File or folder | What it is |
|---|---|
| `models/tfidf_logreg.joblib` | a scikit-learn Pipeline (TF-IDF + Logistic Regression) |
| `models/distilbert_fake_news/` | a folder saved from your fine-tuned DistilBERT |

If your training code has them in memory, save them like this:

```python
import joblib
joblib.dump(pipeline, "models/tfidf_logreg.joblib")        # pipeline = TF-IDF + LogisticRegression

model.save_pretrained("models/distilbert_fake_news")        # your fine-tuned DistilBERT
tokenizer.save_pretrained("models/distilbert_fake_news")
```

If your files have different names, change `TFIDF_MODEL_PATH` and `DISTILBERT_DIR` near the top of `streamlit_app.py`.

### ⚠️ Check which class means "fake" (important)

Datasets disagree: some use `1 = fake`, others `0 = fake`. If the app has this backwards, every verdict is flipped. Check yours:

```python
import joblib
p = joblib.load("models/tfidf_logreg.joblib")
print(p.classes_)                                   # e.g. [0 1]
print(p.predict_proba(["BREAKING!!! Miracle cure the government hides, share now!!!"]))
```

Then for DistilBERT: `print(model.config.id2label)`.

If your labels are words like `FAKE`/`REAL`, nothing to do. If they are numbers and **0 means fake**, set `FAKE_CLASS_INDEX = 0` in the config section of `streamlit_app.py`.

---

## Step 6 (optional): Add API keys for the online "LLM judges"

Skip this if you want a fully offline app. When a key is present, that AI model gives its opinion and counts for **3 weight points** in the vote.

1. Copy `.streamlit/secrets.toml.example` and name the copy `.streamlit/secrets.toml` (VS Code: right-click → Copy, Paste, Rename).
2. Open it and replace the placeholder with your real key(s). **Only fill in the ones you have.**

```toml
GEMINI_API_KEY = "your-key"
OPENROUTER_API_KEY = "your-key"
SARVAM_API_KEY = "your-key"
```

> 🔒 Never share `secrets.toml` or upload it to GitHub. The included `.gitignore` already blocks it.
> 🔒 Privacy: when a key is set, the text you analyse is sent to that company's server.

---

## Step 7: Run the app

```powershell
streamlit run streamlit_app.py
```

✅ Your browser opens `http://localhost:8501`. If it doesn't, copy that address into your browser. To stop the app, click the terminal and press **Ctrl + C**.

⏳ **First time only:** the first image analysis downloads the deepfake model (a few hundred MB). The page shows a "Loading…" message. Later runs are fast.

---

## Step 8: Use it

1. **Text or link:** paste a headline, message or article URL into the left box.
2. **Files:** drop images, videos, audio or `.txt` files in the right box. For a **screenshot**, click inside the box and press **Ctrl+V** to paste it directly (works in recent Streamlit versions; otherwise save it as a PNG and upload).
3. Click **🔍 Analyze**.
4. Read the result:
   * The **verdict banner** (FAKE / no strong sign of fakeness / inconclusive).
   * The **Signals table** showing every detector's vote and weight.
   * The **Details** panels with the frames, ELA picture, OCR text and search results.
   * A yellow **"Skipped or degraded parts"** box lists anything that didn't run. Warnings are normal when you haven't added optional models or keys.
5. Click **🔄 Refresh** in the sidebar to clear everything and start over.

### A quick self-test

| Try this | What should happen |
|---|---|
| Paste any 2–3 sentence news paragraph | Search runs; text models run if you added them |
| Upload a JPG photo | Deepfake probability, ELA picture, OCR text |
| Upload a short MP4 | Three frames (20%, 50%, 80%) each get a score |
| Upload a WAV of someone speaking | Jabberjay gives a synthetic-voice probability |
| Click Refresh | Text box and uploaded files vanish |

---

## How the verdict is decided

Each detector that produced a result casts a vote between 0 (real) and 1 (fake), multiplied by its weight:

| Detector type | Weight |
|---|---|
| Image / video / audio detectors | 2 each |
| Local text models (TF-IDF, DistilBERT) | 2 each |
| Web fact-check search | 1 |
| LLM judges (Gemini, OpenRouter, Sarvam) | 3 each |

**Fake share = (sum of weight × vote) ÷ (sum of weights of detectors that actually ran).** If it is **above 50 %** the verdict is FAKE. Detectors that did not run are left out of both sums, so a missing model never pushes the result either way. If a valid C2PA content-credential declares the file AI-generated, the verdict is forced to FAKE.

Example: image model says 90 % fake (2 × 0.9 = 1.8), DistilBERT says 80 % (2 × 0.8 = 1.6), search finds nothing suspicious (1 × 0 = 0). Total weight = 5, fake = 3.4, share = 68 %, so the verdict is **FAKE**.

Two deliberate details:

* **ELA only votes for JPEG files.** PNG screenshots and video frames always look "edited" to ELA, which would cause false alarms. For those it is shown but does not vote (`ELA_VOTES_ON_NON_JPEG` in the config changes this).
* **ELA is a weak clue.** If real photos get flagged, raise `ELA_THRESHOLD`.

---

## Common problems

| What you see | Fix |
|---|---|
| `streamlit : not recognized` | The venv isn't active. Run `.\venv\Scripts\Activate.ps1` first |
| Yellow "TF-IDF / DistilBERT unavailable" | Normal if you skipped Step 5. Or the path/folder name doesn't match the config |
| Yellow "OCR unavailable" | Install Tesseract (Step 2) in the default folder |
| Video says "could not read any frames" | Install ffmpeg (Step 2), reopen VS Code, or try an MP4 (H.264) file |
| Image model won't load | Needs internet once to download. Check your connection and click Refresh |
| "Web search skipped" | DuckDuckGo rate-limited you or you're offline. The app retried 3 times, then continued without it |
| Every verdict looks backwards | Fix `FAKE_CLASS_INDEX` (Step 5) |
| Terminal spams `torch.classes` / file-watcher errors | Harmless. Silence with `streamlit run streamlit_app.py --server.fileWatcherType none` |
| Port already in use | `streamlit run streamlit_app.py --server.port 8502` |
| Added a model but the app still skips it | Click Refresh. If it persists, stop (Ctrl+C) and run again |

---

## Optional: put it online with Streamlit Community Cloud

1. Push the project to GitHub (don't include `venv/` or `secrets.toml`; `.gitignore` handles this). Large model files may need Git LFS.
2. Go to share.streamlit.io → **New app** → pick your repo → main file `streamlit_app.py`.
3. In **Advanced settings**, choose Python 3.12 and paste your API keys (same format as `secrets.toml`) into **Secrets**.
4. Deploy. Streamlit installs `requirements.txt` and, from `packages.txt`, the Linux packages automatically.

⚠️ Free hosting has limited memory. PyTorch plus several models may exceed it and the app will restart or crash. If that happens, run it locally or use a host with more RAM.

---

## After it works: freeze your versions

New library releases can break things months later. Once the app runs, save the exact versions that worked:

```powershell
pip freeze > requirements-lock.txt
```

---

## Be honest about what this tool can do

* Results are **statistical indicators, not proof**. The deepfake and voice detectors make mistakes, especially on new generation tools, heavy compression or noisy audio.
* The web-search check counts keywords in result snippets. It is a rough hint, which is why it only has weight 1.
* Use the verdict as one input to your own judgement and check important claims against trusted sources.
