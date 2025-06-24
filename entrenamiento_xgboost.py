
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib

# Cargar datos
X = pd.read_csv("../../preprocesamiento/dataset_features.csv")
y = pd.read_csv("../../preprocesamiento/dataset_labels.csv")["Class"]

# Eliminar NaN
X = X.dropna()
y = y[X.index]

# Codificar etiquetas
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Dividir y escalar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train_enc, y_test_enc = train_test_split(X, y_enc, stratify=y_enc, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Modelo
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb.fit(X_train_scaled, y_train_enc)

# Guardar modelo
joblib.dump(xgb, "codigo/xgb_model.pkl")

# Predicción
y_pred_enc = xgb.predict(X_test_scaled)
y_pred = le.inverse_transform(y_pred_enc)
y_test = le.inverse_transform(y_test_enc)

# Reporte
report = classification_report(y_test, y_pred)
print(report)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=le.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig("graficos/confusion_matrix_xgb.png")
plt.close()

# Curvas ROC
y_score = xgb.predict_proba(X_test_scaled)
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
fpr, tpr, roc_auc = dict(), dict(), dict()
for i, class_label in enumerate(lb.classes_):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"{class_label} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curvas ROC - XGBoost")
plt.legend()
plt.tight_layout()
plt.savefig("graficos/roc_xgb.png")
plt.close()
