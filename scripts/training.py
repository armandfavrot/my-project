from pathlib import Path
import sys
import os
import json
import pickle
import random

# Pour rendre les calculs CUDA reproductibles.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use ("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "functions"))

import python_functions as mf
from mlp_module import AmmoniaMLP

RESULTS_DIR = SCRIPT_DIR / "results"
IMAGES_DIR = SCRIPT_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "models").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "losses").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "summaries").mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print (DEVICE)

directory = str(RESULTS_DIR)

# --------------------------------------------------------------------
### Parameters

seed = 1
num_epochs = 100
patience = 100
batch_size = 32
learning_rate = 1e-3
response = "e_rel_72"
output_size = 1
model_name = "mlp_ammonia"

cat_vars = [
    "soil_type2",
    "man_source", "man_trt1",
    "app_method", "incorp",
]

cont_vars = [
    "air_temp_24", "wind_2m_24", "rain_rate_24",
    "man_tan", "man_ph", "man_dm",
    "app_rate", "tan_app",
]

random.seed (seed)
np.random.seed (seed)
torch.manual_seed (seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all (seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms (True)

# --------------------------------------------------------------------
### Data

# Le CSV exporté par R utilise ici l'encodage latin-1.
# keep_default_na=False conserve les libellés comme "None".
data = pd.read_csv (
    SCRIPT_DIR / "data.csv",
    encoding = "latin-1",
    keep_default_na = False,
    na_values = ["NA", "NaN"],
)

# Garder le TAN appliqué dans son unité d'origine avant standardisation.
data["tan_app_origin"] = data["tan_app"]

cat_dims = [int(data[x].max()) + 1 for x in cat_vars]
embedding_dims = [3] * len(cat_vars)
input_size = len(cont_vars) + len(cat_vars)

# --------------------------------------------------------------------
### Split train / validation / test sur les pmid : 70% / 15% / 15%

pmids = np.random.default_rng(seed).permutation (data["pmid"].unique())

n_train = int (0.70 * len(pmids))
n_val = int (0.15 * len(pmids))

pmids_train = pmids[:n_train]
pmids_val = pmids[n_train:n_train + n_val]
pmids_test = pmids[n_train + n_val:]

data_train = data[data["pmid"].isin(pmids_train)].copy()
data_val = data[data["pmid"].isin(pmids_val)].copy()
data_test = data[data["pmid"].isin(pmids_test)].copy()

print (f"Train : {len(data_train)} | Validation : {len(data_val)} | Test : {len(data_test)}")

# --------------------------------------------------------------------
### Standardisation des variables continues

# On calcule les paramètres uniquement sur le train.
means = data_train[cont_vars].mean()
stds = data_train[cont_vars].std(ddof=0).replace (0, 1)

data_train[cont_vars] = (data_train[cont_vars] - means) / stds
data_val[cont_vars] = (data_val[cont_vars] - means) / stds
data_test[cont_vars] = (data_test[cont_vars] - means) / stds

# --------------------------------------------------------------------
### Tenseurs

x_cont_train, x_cat_train, y_train = mf.make_tensors (
    data_train, response, cont_vars, cat_vars, DEVICE
)
x_cont_val, x_cat_val, y_val = mf.make_tensors (
    data_val, response, cont_vars, cat_vars, DEVICE
)

x_train = (x_cont_train, x_cat_train)
x_val = (x_cont_val, x_cat_val)

# --------------------------------------------------------------------
### Training

model = AmmoniaMLP (
    input_size = input_size,
    output_size = output_size,
    cat_dims = cat_dims,
    embedding_dims = embedding_dims,
).to(DEVICE)

training_summary = mf.train_model (
    model, num_epochs, learning_rate,
    x_train, y_train, x_val, y_val,
    DEVICE, directory, model_name,
    print_epoch = True,
    batch_size = batch_size,
    patience = patience,
)

print (f"Temps d'exécution : {training_summary['duration_seconds']:.2f} secondes")

with open (RESULTS_DIR / "summaries" / f"training_summary_{model_name}.json", "w") as f:
    json.dump (training_summary, f, indent=4)

# --------------------------------------------------------------------
### Prédictions sur le jeu de test

# Recharger le modèle de l'époque avec la meilleure perte de validation.
model.load_state_dict (
    torch.load (
        RESULTS_DIR / "models" / f"model_{model_name}.pth",
        weights_only = True,
        map_location = DEVICE,
    )
)

data_predictions = mf.predict_emissions (
    data_test, model, cont_vars, cat_vars, response, DEVICE
)

# La sortie du modèle est une fraction du TAN appliqué.
# Revenir aux émissions absolues avec le TAN non standardisé.
data_predictions = data_predictions.rename (columns={"prediction": "prediction_rel"})
data_predictions["prediction"] = (
    data_predictions["prediction_rel"] * data_predictions["tan_app_origin"]
)

metrics = mf.compute_metrics (
    data_predictions["e_72"], data_predictions["prediction"]
)
print (metrics)

data_predictions[["pmid", response, "prediction_rel", "tan_app_origin", "e_72", "prediction"]].to_csv (
    RESULTS_DIR / "test_predictions.csv", index=False
)

with open (RESULTS_DIR / "summaries" / "test_metrics.json", "w") as f:
    json.dump (metrics, f, indent=4)

# --------------------------------------------------------------------
### Figure : émissions observées vs prédites sur le jeu de test

observed = data_predictions["e_72"]
predicted = data_predictions["prediction"]

limits = [
    min (0, observed.min(), predicted.min()),
    max (observed.max(), predicted.max()) * 1.05,
]

fig, ax = plt.subplots (figsize=(6, 5.5))

ax.scatter (observed, predicted, color="steelblue", alpha=0.75,
            edgecolors="white", linewidths=0.5)
ax.plot (limits, limits, color="black", linestyle="--", linewidth=1)

ax.set (
    xlabel = "Émissions observées à 72 h (kg N/ha)",
    ylabel = "Émissions prédites à 72 h (kg N/ha)",
    title = "Prédictions du MLP — jeu de test",
    xlim = limits,
    ylim = limits,
)
ax.set_aspect ("equal", adjustable="box")
ax.grid (alpha=0.2)
ax.set_axisbelow (True)

ax.text (
    0.05, 0.95,
    f"RMSE = {metrics['RMSE']:.2f} kg N/ha\n"
    f"R² = {metrics['R2']:.2f}\n"
    f"n = {len(data_predictions)}",
    transform = ax.transAxes,
    va = "top",
)

fig.tight_layout()
fig.savefig (IMAGES_DIR / "observed_vs_predicted.pdf", bbox_inches="tight")
fig.savefig (IMAGES_DIR / "observed_vs_predicted.png", dpi=200, bbox_inches="tight")
plt.close (fig)

# --------------------------------------------------------------------
### Sauvegarde des paramètres pour réutiliser le modèle

preprocessing = {
    "cont_vars": cont_vars,
    "cat_vars": cat_vars,
    "response": response,
    "means": means,
    "stds": stds,
    "cat_dims": cat_dims,
    "embedding_dims": embedding_dims,
    "category_labels": data[cat_vars + [x + "_origin" for x in cat_vars]].drop_duplicates(),
}

with open (RESULTS_DIR / "models" / "preprocessing.pkl", "wb") as f:
    pickle.dump (preprocessing, f)

splits = pd.concat ([
    data_train[["pmid"]].assign(split="train"),
    data_val[["pmid"]].assign(split="validation"),
    data_test[["pmid"]].assign(split="test"),
])
splits.to_csv (RESULTS_DIR / "splits.csv", index=False)

