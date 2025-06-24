
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Cargar datos
X = pd.read_csv("../../preprocesamiento/dataset_features.csv").dropna()
y = pd.read_csv("../../preprocesamiento/dataset_labels.csv")["Class"]
y = y[X.index]

# Dividir y escalar
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo SVM
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)

# Guardar modelo
joblib.dump(svm, "codigo/svm_model.pkl")

# Predicción
y_pred = svm.predict(X_test_scaled)
report = classification_report(y_test, y_pred)
print(report)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=svm.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_)
disp.plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig("graficos/confusion_matrix_svm.png")
plt.close()

# Curvas ROC
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
y_score = svm.predict_proba(X_test_scaled)
fpr, tpr, roc_auc = dict(), dict(), dict()
for i, class_label in enumerate(lb.classes_):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f"{class_label} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curvas ROC - SVM")
plt.legend()
plt.tight_layout()
plt.savefig("graficos/roc_svm.png")
plt.close()
