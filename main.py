import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from joblib import load
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from starlette.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SARSPred: COVID-19 Drug Activity Prediction API",
    description="Predicts drug-like compound activity against SARS-CoV-2 using an XGBoost model and applicability domain analysis.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

METRICS_FILE = "./metrics.csv"
XGBOOST_MODEL_PATH = "./weights/XGBoost_model.pkl"

dp = None
xgb_model = None


@app.on_event("startup")
async def load_resources():
    """Load model and normalization metrics."""
    global dp, xgb_model

    if not os.path.exists(METRICS_FILE) or not os.path.exists(XGBOOST_MODEL_PATH):
        raise FileNotFoundError("Metrics or model file missing.")

    dp = pd.read_csv(METRICS_FILE)
    dp.drop("metrics", axis=1, inplace=True)
    xgb_model = load(XGBOOST_MODEL_PATH)
    print("Resources loaded successfully (XGBoost + Metrics)")


def compute_morgan_fingerprints(smiles_list: List[str]) -> pd.DataFrame:
    """Generate 2048-bit Morgan fingerprints for SMILES."""
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        if mol is not None else np.zeros(2048)
        for mol in mols
    ]
    fp_array = [np.array(fp) for fp in fps]
    df = pd.DataFrame(fp_array, columns=[f"morgan_{i}" for i in range(2048)])
    df.insert(0, "SMILES", smiles_list)
    return df


def normalize_fingerprints(fingerprints: pd.DataFrame) -> pd.DataFrame:
    """Normalize fingerprints using training metrics."""
    fingerprints = fingerprints[dp.columns]
    df = pd.DataFrame()
    for col in dp.columns:
        mean, std = dp[col][0], dp[col][1]
        df[col] = (fingerprints[col] - mean) / std
    return df.fillna(0)


def smiles_to_image_base64(smiles: str) -> str:
    """Convert SMILES to molecule image (Base64-encoded)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(250, 250))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def applicability_domain_analysis(smiles_list: List[str]) -> tuple[pd.DataFrame, float, str]:
    """Perform applicability domain analysis (return results and Williams plot)."""
    metrics = dp.columns
    train_data = pd.read_csv("./Training Data.csv")[metrics]
    train_data["Compounds"] = "Train"

    new_data = compute_morgan_fingerprints(smiles_list)[metrics]

    df_standardized = (new_data - dp.loc[0]) / dp.loc[1]
    df_standardized = df_standardized.fillna(0)
    df_standardized["Compounds"] = "New Compound"

    combined_data = pd.concat([train_data, df_standardized], ignore_index=True)

    X_all = combined_data.drop(columns=["Compounds"]).fillna(0)
    X_standardized = (X_all - dp.loc[0]) / dp.loc[1]
    pca = PCA(n_components=1)
    Y = pca.fit_transform(X_standardized).flatten()
    X_data = X_standardized

    X_with_const = sm.add_constant(X_data)
    model = sm.OLS(Y, X_with_const)
    results = model.fit()

    standardized_residuals = results.resid_pearson
    leverage = results.get_influence().hat_matrix_diag

    results_df = pd.DataFrame({
        "Standardized Residual": standardized_residuals,
        "Leverage": leverage,
        "Compounds": combined_data["Compounds"].values
    })

    n_features = X_data.shape[1]
    n_samples = len(X_data)
    critical_leverage = 3 * (n_features + 1) / n_samples

    new_results = results_df[results_df["Compounds"] == "New Compound"].copy()
    new_results["AD_Status"] = np.where(
        (abs(new_results["Standardized Residual"]) <= 3)
        & (new_results["Leverage"] <= critical_leverage),
        "INSIDE AD",
        "OUTSIDE AD",
    )

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=results_df,
        x="Leverage",
        y="Standardized Residual",
        hue="Compounds",
        palette={"Train": "blue", "New Compound": "orange"},
        s=120,
        alpha=0.7,
    )
    plt.axhline(y=3, ls="--", c="red", alpha=0.7)
    plt.axhline(y=-3, ls="--", c="red", alpha=0.7)
    plt.axvline(x=critical_leverage, ls="--", c="red", alpha=0.7)
    plt.title("Applicability Domain - Williams Plot")
    plt.xlabel("Leverage")
    plt.ylabel("Standardized Residual")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG")
    plt.close()
    buf.seek(0)
    plot_b64 = base64.b64encode(buf.read()).decode("utf-8")

    return new_results, critical_leverage, plot_b64


class SmilesList(BaseModel):
    smiles: List[str]


@app.post("/predict/")
async def predict_with_ad(smiles_list: SmilesList):
    """Predict molecular activity + applicability domain."""
    if not smiles_list.smiles:
        raise HTTPException(status_code=400, detail="No SMILES provided.")

    fingerprints = compute_morgan_fingerprints(smiles_list.smiles)
    df_norm = normalize_fingerprints(fingerprints)

    preds = xgb_model.predict(df_norm)
    probs = xgb_model.predict_proba(df_norm)[:, 1]

    ad_results, critical_leverage, ad_plot_b64 = applicability_domain_analysis(smiles_list.smiles)

    final_results = []
    for smi, pred, prob, (_, row) in zip(smiles_list.smiles, preds, probs, ad_results.iterrows()):
        mol_img_b64 = smiles_to_image_base64(smi)
        final_results.append({
            "smiles": smi,
            "prediction": "Active" if pred == 1 else "Inactive",
            "confidence": round(float(prob), 4),
            "applicability_domain": row["AD_Status"],
            "molecule_image": mol_img_b64,
        })

    return JSONResponse(content={
        "results": final_results,
        "critical_leverage": critical_leverage,
        "williams_plot": ad_plot_b64,
    })


@app.post("/predict_file/")
async def predict_from_file(file: UploadFile = File(...)):
    """Predict molecular activity from uploaded file."""
    temp_path = f"./temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in [".smi", ".txt"]:
            with open(temp_path, "r", encoding="utf-8") as f:
                smiles_list = [l.strip() for l in f.readlines() if l.strip()]
        elif ext == ".csv":
            df = pd.read_csv(temp_path)
            smiles_list = df[df.columns[0]].dropna().astype(str).tolist()
        elif ext == ".xlsx":
            df = pd.read_excel(temp_path)
            smiles_list = df[df.columns[0]].dropna().astype(str).tolist()
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")
    finally:
        os.remove(temp_path)

    return await predict_with_ad(SmilesList(smiles=smiles_list))
