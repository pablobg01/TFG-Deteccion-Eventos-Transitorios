
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Rutas
lightcurves_path = "transient_lightcurves.csv"
labels_path = "transient_labels.csv"
output_path = "."

# Cargar datos
lightcurves = pd.read_csv(lightcurves_path)
labels = pd.read_csv(labels_path)
lightcurves['ID'] = lightcurves['ID'].astype(str)
labels['TransientID'] = labels['TransientID'].astype(str)

# Renombrar para poder fusionar
lightcurves = lightcurves.rename(columns={'ID': 'TransientID'})
data = pd.merge(lightcurves, labels, on="TransientID")

# Clasificación robusta
def map_class_robusta(label):
    label = str(label).lower().replace('?', '').strip()
    if "sn" in label: return "SN"
    if "cv" in label or "nova" in label: return "CV"
    if "agn" in label: return "AGN"
    if "blazar" in label: return "Blazar"
    if "flare" in label: return "Flare"
    if "hpm" in label: return "HPM"
    if any(x in label for x in ["lpv", "yso", "agb", "variable", "var", "mira", "rrl", "rcorb", "rrlyrae", "amcvn", "carbon"]):
        return "StellarVar"
    if "ast" in label or "comet" in label: return "Asteroid"
    return None

labels["ClassGroup"] = labels["Classification"].apply(map_class_robusta)
data["ClassGroup"] = data["Classification"].apply(map_class_robusta)
data = data[data["ClassGroup"].notnull()]

# Agregación para obtener features
features = data.groupby("TransientID").agg({
    "Mag": ['mean','median','std','max','min'],
    "Magerr": ['mean','std'],
    "MJD": ['min','max'],
    "observation_id": 'count'
})
features.columns = ['mag_mean','mag_median','mag_std','mag_max','mag_min',
                    'magerr_mean','magerr_std','mjd_min','mjd_max','count']
features = features.reset_index()
features["amplitude"] = features["mag_max"] - features["mag_min"]
features["duration"] = features["mjd_max"] - features["mjd_min"]

# Fusionar con clases
dataset = pd.merge(features, labels, on="TransientID")
dataset["Class"] = dataset["Classification"].apply(map_class_robusta)
dataset = dataset.dropna(subset=["Class"])

# Variables predictoras y etiquetas
X = dataset[['mag_mean','mag_median','mag_std','mag_max','mag_min',
             'magerr_mean','magerr_std','amplitude','duration','count']]
y = dataset["Class"]

# Dividir y escalar
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Guardar
X.to_csv(os.path.join(output_path, "dataset_features.csv"), index=False)
y.to_csv(os.path.join(output_path, "dataset_labels.csv"), index=False)
