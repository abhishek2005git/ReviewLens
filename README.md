# ReviewLens

A small local web app: write an English review, click **Predict rating**, and get **1–5 stars**.

The app uses **TF-IDF + a linear text classifier**, with a plain HTML/CSS/JavaScript frontend and a Flask backend. The full-data workflow trains a linear SVM; the small-data/demo workflow uses logistic regression. No API key, Node.js, TensorFlow, business dataset, or cloud service is needed to run predictions. Reviews are processed locally and are not saved by the app.

## Get ReviewLens from GitHub

The trained model is included. You do not need the large Yelp dataset or a training run to use the app.

**With Git installed:**

```bash
git clone https://github.com/abhishek2005git/ReviewLens.git
cd ReviewLens
```

Then follow the macOS or Windows setup below, starting with the Python version check because you are already in the project folder. For a private network or a computer without Git, choose **Code → Download ZIP** on the repository page, extract it, and open the extracted folder. Installation requires Python and an internet connection to download packages; predictions run locally afterward.

## Setup on macOS and Windows

### Before you start (both systems)

1. Use this **updated project folder**, not a fresh download of the original research repository. You can clone this repository with Git or choose **Code → Download ZIP** on its GitHub page and extract it. Put the folder inside `Desktop/Projects`. The commands below assume the folder is named `ReviewLens`; change the `cd` / `Set-Location` path if your clone/ZIP folder has another name.
2. Install **standard 64-bit Python 3.13**. This project and its saved model were tested with Python 3.13.5. Use the pinned versions in `requirements-lock.txt` when loading the supplied model; that file includes the optional training dependencies too.
3. The repository includes **`models/review_model.joblib`** (about 20 MB), its evaluation report, and the original small-model baseline. A normal clone or downloaded ZIP includes the improved model, so **retraining is not required to run the app**. The large Yelp datasets and virtual environment are excluded from Git. If you make a source-only copy without the saved model, copy that trusted model separately or run the full training steps below.
4. **Do not copy `.venv` between computers or operating systems.** Create a new virtual environment on the destination computer. The dataset is optional when you already have the saved model. Internet is needed for installing packages and downloading training data; predictions run locally.

### macOS: first-time setup

1. Install Python 3.13 using the official [macOS Python downloads](https://www.python.org/downloads/macos/). After installation, open `/Applications/Python 3.13/` and double-click **Install Certificates.command** to finish certificate setup. See the [official macOS installation instructions](https://docs.python.org/3.13/using/mac.html).
2. Open **Terminal** and run these commands one at a time:

```bash
cd "$HOME/Desktop/Projects/ReviewLens"
python3.13 --version
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python app.py
```

3. Open **http://127.0.0.1:5001/** in your browser. Enter a review and click **Predict rating**. Leave Terminal open while using the app.
4. To stop the server, press **Control+C** in that Terminal window.

For later starts, open Terminal in the project folder and run:

```bash
.venv/bin/python app.py
```

You can also double-click **Start.command** after setup. If its executable permission was lost while copying the project, run `chmod +x Start.command` from the project folder first. The launcher uses the existing `.venv` and installs missing runtime packages if necessary.

### Windows: first-time setup (PowerShell)

1. Install standard 64-bit Python 3.13 from the official [Windows Python downloads](https://www.python.org/downloads/windows/). Include **pip** and the **Python launcher (`py`)**, and select **Add python.exe to PATH** if the installer offers it. Close and reopen PowerShell after installing. See the [official Windows installation instructions](https://docs.python.org/3.13/using/windows.html).
2. Open **PowerShell** and run these commands one at a time:

```powershell
Set-Location "$HOME\Desktop\Projects\ReviewLens"
py -3.13 --version
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe app.py
```

3. Open **http://127.0.0.1:5001/** in your browser. Enter a review and click **Predict rating**. Leave PowerShell open while using the app.
4. To stop the server, press **Ctrl+C** in that PowerShell window.

For later starts, open PowerShell in the project folder and run:

```powershell
.\.venv\Scripts\python.exe app.py
```

These instructions call the virtual environment's Python directly, so you do **not** need to activate it or change PowerShell execution policy. If your Desktop is inside OneDrive, replace the `Set-Location` path with the actual project path shown in File Explorer. You can also open the project folder in File Explorer, type `powershell` in its address bar, and press Enter; then start at the `py -3.13 --version` command.

### Check that the improved model is loaded

Open **http://127.0.0.1:5001/api/health** in a browser. For the supplied model, expect:

```json
{
  "status": "ok",
  "mode": "trained",
  "model_name": "TF-IDF + Linear SVM",
  "training_rows": 649879
}
```

The response also includes evaluation metrics. The page should say **Trained on 649,879 labeled reviews**. If it says **Demo model**, `models/review_model.joblib` is missing; the app has fallen back to 25 handwritten example reviews. Copy the saved model or train it, then stop and restart the server.

### Common setup problems

| Problem | What to do |
| --- | --- |
| `python3.13` or `py` is not found | Finish installing Python 3.13 and reopen the terminal. On Windows, ensure the Python launcher was installed. |
| The project folder cannot be found | Use the actual folder path in the first command; quote paths containing spaces. Check whether Windows redirected Desktop to OneDrive. |
| Packages are missing | Run the platform's virtual-environment Python with `-m pip install -r requirements-lock.txt` from the project folder. |
| A copied `.venv` does not work | Move that copied `.venv` aside and create a new one using the setup commands for the destination system. |
| A model version warning appears | Use the pinned dependencies and Python 3.13 in a new virtual environment, or retrain the model in your chosen environment. |
| `Address already in use` / port 5001 is busy | If this app is already running, use its browser page. Stop the previous copy with Ctrl+C before starting another. |
| The browser cannot connect | Keep the server terminal open and check it for errors. Open the URL only after it prints `Running on http://127.0.0.1:5001`. |
| An HTTPS certificate error occurs on macOS | Complete **Install Certificates.command** from the Python 3.13 Applications folder, then retry installation. |

The macOS setup was checked in this workspace. The GitHub Actions workflow checks Python 3.13 installation, tests, and saved-model loading on Windows, macOS, and Linux. Check the repository's Actions tab for actual run results; configuring the workflow alone does not confirm those remote checks passed. [Virtual environment documentation](https://docs.python.org/3.13/library/venv.html).

### Current local model

This checkout has a saved **TF-IDF + Linear SVM** trained on **649,879 usable reviews**, using the full 650,000-review YelpReviewFull training partition (about 130 times the original training data). The separate 50,000-review test partition retained **49,984** reviews after cleaning and removing training overlap.

| Metric on the same independent test reviews | Previous 5k model | Full-data model |
| --- | ---: | ---: |
| Exact accuracy | 52.48% | **61.32%** |
| Within one star | 88.68% | **94.83%** |
| Mean absolute error | 0.645 stars | **0.455 stars** |

Exact accuracy improved by **8.84 percentage points**. The original 51.8% result used a smaller, different test split; the table above is the fair comparison on the same independent reviews. The selected regularization is C=0.25. The full selection, refitting, and evaluation run took 24.7 minutes on this Mac. All **12 automated tests** passed. These results are benchmark measurements, not guaranteed accuracy on new user reviews. See `models/full_training_report.json` for per-rating precision/recall, the confusion matrix, validation results, and cleaning counts.

## Train on the full benchmark

The complete YelpReviewFull benchmark contains **650,000 training reviews and 50,000 separate test reviews**, about **323 MB compressed**. **Skip retraining if you already have the supplied model and only want to use the app.** For full training, use a machine with at least 16 GB of RAM as a practical starting point; this run was completed on a 16 GB Mac. Allow extra disk space beyond the compressed download for packages and model files. Training time depends on the machine; the recorded run took about 25 minutes after downloading.

First stop the running app with Ctrl+C. Open a terminal in the project folder. Check that curl is available with `curl --version` on macOS or `curl.exe --version` in Windows PowerShell. Then run the appropriate commands:

**macOS (Terminal):**

```bash
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python download_full.py
.venv/bin/python improve_model.py
.venv/bin/python app.py
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe download_full.py
.\.venv\Scripts\python.exe improve_model.py
.\.venv\Scripts\python.exe app.py
```

Wait for each command to finish before starting the next. The last command keeps running to serve the app. Reopen **http://127.0.0.1:5001/** when it starts. For a new experiment that does not need the exact saved-model environment, `requirements-training.txt` provides compatible dependency ranges instead of the pinned lock file.

`download_full.py` uses curl to resume interrupted downloads and verifies Parquet row counts before activating the files. It skips already downloaded files. macOS includes curl; on Windows, use a curl installation available on PATH.

`improve_model.py` removes unusable, duplicate, and conflicting reviews. It reserves 30,000 reviews from the training partition for validation, learns up to 400,000 unigram/bigram features using only the remaining training reviews, and compares linear SVM regularization settings C=0.25, 1, and 4. It selects settings by validation accuracy, refits on all usable training reviews, and evaluates once on the separate official test partition after removing train/test text overlap. It evaluates the saved 5k-review baseline on those same test reviews for a fair comparison. No test reviews enter training or model selection.

The app model is replaced only if the new model beats that baseline. `models/full_training_report.json` records settings, cleaning counts, validation scores, paired test results, per-rating metrics, and a confusion matrix. The old model remains in `models/baseline_5k.joblib`. Training takes substantially longer and uses more RAM than the demo; this workflow was run on a Mac with 16 GB of RAM. Prediction remains a lightweight local operation. **Restart the app after training.**

## Optional small-data workflow

This replaces the improved model with a smaller logistic-regression model. It is intended for experiments, not required setup.

The examples in this section and the custom-data section use `python` after activating the environment. On macOS, run `source .venv/bin/activate` first. In Windows **Command Prompt**, run `.venv\Scripts\activate.bat` first; or stay in PowerShell and replace each `python` below with `.\.venv\Scripts\python.exe` without activating.

Download just 5,000 real, labeled Yelp reviews, then train:

```bash
python download_sample.py
python train.py data/yelp_sample.csv
python app.py
```

The download script fetches small batches from the public Hugging Face dataset viewer API. It does **not** fetch the complete Yelp archive. This step needs internet; predictions afterward work locally. Choose a sample size with `python download_sample.py --rows 20000 --output data/yelp_20k.csv` (up to 50,000). The service may rate-limit requests or be unavailable; interrupted downloads are discarded safely. Existing files are preserved.

Data source: [YelpReviewFull](https://huggingface.co/datasets/Yelp/yelp_review_full), constructed by Xiang Zhang et al. from the Yelp Dataset Challenge 2015. Its labels 0–4 are converted to stars 1–5. This is an older benchmark, not the complete current Yelp Open Dataset. See the dataset card and the [official Yelp dataset terms](https://business.yelp.com/data/resources/open-dataset/) for permitted use. The downloader retrieves the first N training rows; this convenience sample is not a claim of representativeness.

## Train on your own data

Use a CSV or TSV with `text` and `stars` columns, or the official Yelp review JSONL file (one JSON object per line):

```csv
text,stars
"Great food and friendly service!",5
"Cold food and a very long wait.",1
```

```bash
python train.py path/to/reviews.csv
# Or read only the first 50,000 unique reviews from a large Yelp file:
python train.py path/to/yelp_academic_dataset_review.json --max-rows 50000
```

This lightweight `train.py` command trains logistic regression and **replaces the current saved app model**, including a model previously trained by `improve_model.py`. It reads the input incrementally and caps it at 100,000 unique reviews by default. It skips empty or punctuation-only reviews, removes duplicate texts, and rejects conflicting labels. It requires at least five distinct usable reviews per star class and reports exact accuracy, accuracy within one star, and mean absolute error on a stratified 20% held-out split. The vectorizer is fitted only on the training split for evaluation. After evaluation, the final model is fitted on all loaded reviews and saved to `models/review_model.joblib`; evaluation results and class counts are saved in `models/review_model.json`. **Restart the app after training.** Only load model files that you trained yourself or otherwise trust.

The model can confuse adjacent ratings, negation, sarcasm, and mixed reviews. More training data improves coverage but does not make subjective star ratings unambiguous. These are estimates, not an objective score. No confidence percentage is displayed because the classifier's probabilities have not been calibrated.

## Project files

```text
app.py                 Flask server: page, prediction, health check
model.py               Shared ML pipeline, validation, model loading
train.py               Dataset loading, evaluation, model saving
improve_model.py       Full-data SVM selection and independent test evaluation
download_full.py       Resumable full-benchmark downloader
download_sample.py     Small real Yelp sample downloader
requirements.txt       Runtime dependencies
requirements-training.txt  Optional Parquet training dependency
requirements-lock.txt  Exact versions used for verification
templates/index.html   Single-page interface
static/                CSS and frontend JavaScript
data/demo_reviews.csv  Tiny handwritten fallback dataset
models/                Locally trained model (generated)
tests/                 API, download, and evaluation-integrity checks
```

The original notebook and `images/` are retained for reference and are not needed to run the app. The original README is preserved in `docs/original-project.md`.

## API

`POST /api/predict` with `Content-Type: application/json`:

```json
{"review": "Delicious food and wonderful service!"}
```

Returns an integer `rating` from 1 to 5 and `mode` (`demo` or `trained`). Reviews must contain words, have at least three characters, and be no longer than 5,000 characters. Inputs with no recognized vocabulary return a helpful error. `GET /api/health` reports model mode, training row count, model name, and evaluation metrics.

## Verify

Run from the project folder after setup.

**macOS:**

```bash
.venv/bin/python -m unittest discover -s tests -v
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected result: **12 tests pass**. This checks the API, dataset validation, duplicate removal, train/test overlap handling, and baseline creation; it does not rerun full training.

## Original work

Adapted from Mansi Patel's [Yelp Business Stars’ Rating Prediction](https://github.com/mansipatel2508/Yelp-Review-Stars-Prediction-with-Machine-Learning) (2019). That research project combines business reviews and compares several models. This app simplifies the task to predicting the rating of **one review**. The original notebook is preserved with attribution; no new license is asserted over the original material or Yelp data.
