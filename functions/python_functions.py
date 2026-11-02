import time
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def make_tensors (data, response, cont_vars, cat_vars, DEVICE):
    """Créer les tenseurs pour toutes les observations d'un sous-ensemble."""

    x_cont = torch.tensor (
        data[cont_vars].values, dtype=torch.float32
    ).to(DEVICE)

    x_cat = torch.tensor (
        data[cat_vars].values, dtype=torch.long
    ).to(DEVICE)

    y = torch.tensor (
        data[[response]].values, dtype=torch.float32
    ).to(DEVICE)

    return (x_cont, x_cat, y)


def make_mini_batches (x_training, y_training, batch_size):
    """Mélanger les lignes en conservant l'alignement des trois tenseurs."""

    x_cont, x_cat = x_training
    indices = torch.randperm (len(y_training), device=y_training.device)

    mini_batches_x = []
    mini_batches_y = []

    for i in range (0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]

        mini_batches_x.append (
            (x_cont[batch_indices], x_cat[batch_indices])
        )
        mini_batches_y.append (y_training[batch_indices])

    return (mini_batches_x, mini_batches_y)


def compute_validation_loss (x_validation, y_validation, model, criterion):
    """Calculer la perte sur le jeu de validation."""

    model.eval()

    with torch.no_grad():
        x_cont, x_cat = x_validation
        predictions = model (x_cont, x_cat)
        validation_loss = criterion (predictions, y_validation).item()

    model.train()

    return validation_loss


def train_model (
    model,
    num_epochs,
    learning_rate,
    x_training,
    y_training,
    x_validation,
    y_validation,
    DEVICE,
    directory,
    model_name,
    print_epoch=True,
    batch_size=32,
    patience=50,
):
    """Entraîner le modèle et sauvegarder la meilleure époque."""

    start_time = time.time()
    model.to (DEVICE)

    optimizer = torch.optim.Adam (model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    validation_loss_best = np.inf
    validation_loss_list = []
    train_loss_list = []
    counter = 0
    best_epoch = 0

    model.train()

    for epoch in range (num_epochs):
        epoch_loss = 0
        n_obs = 0

        mini_batches_x, mini_batches_y = make_mini_batches (
            x_training, y_training, batch_size
        )

        for x_batch, targets in zip (mini_batches_x, mini_batches_y):
            x_cont, x_cat = x_batch
            predictions = model (x_cont, x_cat)
            batch_train_loss = criterion (predictions, targets)

            optimizer.zero_grad()
            batch_train_loss.backward()
            optimizer.step()

            batch_n = targets.shape[0]
            epoch_loss += batch_train_loss.item() * batch_n
            n_obs += batch_n

        train_loss = epoch_loss / n_obs
        train_loss_list.append (train_loss)

        validation_loss = compute_validation_loss (
            x_validation, y_validation, model, criterion
        )
        validation_loss_list.append (validation_loss)

        if validation_loss < validation_loss_best:
            validation_loss_best = validation_loss
            best_epoch = epoch + 1
            counter = 0

            torch.save (
                model.state_dict(),
                Path(directory) / "models" / f"model_{model_name}.pth",
            )
        else:
            counter += 1

        if print_epoch and (epoch + 1) % max(1, num_epochs // 10) == 0:
            print (
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"train = {train_loss:.4f} | "
                f"validation = {validation_loss:.4f}"
            )

        if counter >= patience:
            break

    with open (Path(directory) / "losses" / f"train_losses_model_{model_name}.pkl", "wb") as f:
        pickle.dump (train_loss_list, f)

    with open (Path(directory) / "losses" / f"validation_losses_model_{model_name}.pkl", "wb") as f:
        pickle.dump (validation_loss_list, f)

    return {
        "duration_seconds": time.time() - start_time,
        "epochs_completed": len(train_loss_list),
        "best_epoch": best_epoch,
        "best_validation_mse": validation_loss_best,
    }


def predict_emissions (data, model, cont_vars, cat_vars, response, DEVICE):
    """Ajouter une prédiction par observation, dans l'ordre du tableau."""

    data_predictions = data.copy()

    x_cont, x_cat, y = make_tensors (
        data_predictions, response, cont_vars, cat_vars, DEVICE
    )

    model.eval()

    with torch.no_grad():
        predictions = model (x_cont, x_cat)

    data_predictions["prediction"] = predictions.cpu().numpy().reshape(-1)

    return data_predictions


def compute_metrics (observed, predicted):
    """Calculer les métriques sur l'échelle des observations fournies."""

    observed = np.asarray (observed)
    predicted = np.asarray (predicted)
    errors = predicted - observed

    mse = np.mean (errors ** 2)
    total = np.sum ((observed - observed.mean()) ** 2)

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(1 - np.sum(errors ** 2) / total) if total > 0 else None,
        "MBE": float(errors.mean()),
    }
