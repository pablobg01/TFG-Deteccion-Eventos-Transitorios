
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler

# Cargar datos
X = pd.read_csv("dataset_features.csv")
y = pd.read_csv("dataset_labels.csv")

# Variables predictoras y etiquetas
y = y["Classification"]

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

y = y.apply(map_class_robusta)
X = X[y.notnull()]
y = y[y.notnull()]

# Split y escalado
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Entrenamiento
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

# Reporte
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig("graficos/confusion_matrix_rf.png")

# ROC curves
lb = LabelBinarizer()
y_test_binarized = lb.fit_transform(y_test)
y_score = clf.predict_proba(X_test_scaled)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i, class_label in enumerate(lb.classes_):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"{class_label} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curvas ROC - Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig("graficos/roc_curves_rf.png")
plt.clf()

# Importancia de variables
importances = clf.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values("Importance", ascending=False)
sns.barplot(data=feat_imp_df, x="Importance", y="Feature")
plt.title("Importancia de Variables - Random Forest")
plt.tight_layout()
plt.savefig("graficos/feature_importance_rf.png")
