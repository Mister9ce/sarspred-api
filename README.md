# SARSPred API

SARSPred API is a machine learning–powered service designed to predict the antiviral activity of chemical compounds against SARS-CoV-2. It uses the XGBoost algorithm to classify compounds based on molecular descriptors and provides prediction confidence, applicability domain, and visual outputs such as molecular structure and Williams plots.

### Features

* Accepts `.csv`, `.txt`, and `.xlsx` files containing molecular data.
* Generates activity predictions with confidence scores.
* Evaluates applicability domain for model reliability.
* Returns visual molecular representations and Williams plots.

### Tech Stack

* **Backend:** FastAPI
* **Model:** XGBoost
* **Language:** Python 3.10+
* **Environment:** uv / virtualenv

### Running Locally

```bash
uvicorn main:app --reload
```

Then open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
