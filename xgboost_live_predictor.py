import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
import sys

# Windows CMD Unicode fix
sys.stdout.reconfigure(encoding='utf-8')

print("="*50)
print("🤖 XGBoost Live Predictor - La Liga")
print("="*50)

print("\n1. Cargando base de datos histórica de La Liga desde la URL...")
url = "https://www.football-data.co.uk/mmz4281/2526/SP1.csv"
df = pd.read_csv(url)

print(f"✔️ Datos descargados exitosamente. ({len(df)} partidos encontrados)")

# Renombrar columnas para quitar caracteres especiales que rompen XGBoost (< y >)
df = df.rename(columns={'B365>2.5': 'B365_Over25', 'B365<2.5': 'B365_Under25'})

# Características (Features) a usar para entrenar (Información que se conoce AL MEDIOTIEMPO)
# No usamos los goles finales o tiros finales porque eso sería hacer 'trampa' (Data Leakage)
# Usamos las cuotas previas porque resumen el nivel de favorito de cada equipo.
features = ['HTHG', 'HTAG', 'AvgH', 'AvgD', 'AvgA', 'B365_Over25', 'B365_Under25']
target = 'FTR'

# Limpiar valores vacíos
df = df.dropna(subset=features + [target, 'HTR']).copy()

# Variables predictoras (X) y Etiquetas Reales al final del partido (y)
y_map = {'H': 0, 'D': 1, 'A': 2}
df['target'] = df[target].map(y_map)
y = df['target']

# Codificar a números el resultado al medio tiempo (HTR)
df['HTR_encoded'] = df['HTR'].map(y_map)
features.append('HTR_encoded')

# Agregar diferencia de goles al mediotiempo como característica inventada (Feature Engineering)
df['HT_Goal_Diff'] = df['HTHG'] - df['HTAG']
features.append('HT_Goal_Diff')

X = df[features]

print("\n2. Dividiendo datos (80% Entrenamiento, 20% Pruebas Ciega)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n3. Entrenando el modelo de Machine Learning (XGBoost Classifier)...")
model = xgb.XGBClassifier(
    objective='multi:softprob', 
    eval_metric='mlogloss',
    max_depth=4,
    learning_rate=0.05,
    n_estimators=100,
    random_state=42
)

# Entrenamiento
model.fit(X_train, y_train)

# Predicción ciega
y_pred = model.predict(X_test)

print("\n" + "="*50)
print("📊 RESULTADOS DEL MODELO")
print("="*50)

# Evaluar el Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Exactitud Global Identificando Ganadores (Accuracy): {acc * 100:.2f}%\n")

# Desglose de aciertos
target_names = ['Local (H)', 'Empate (D)', 'Visita (A)']
print("Detalle de la predicción y capacidad del AI:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\n4. Generando gráfico de Importancia de Variables...")
# Guardar gráfico de decisiones del algoritmo
fig, ax = plt.subplots(figsize=(10, 6))
xgb.plot_importance(model, importance_type='weight', ax=ax, title='Características más importantes para la IA', xlabel='Importancia (Peso)', ylabel='Dato')
plt.tight_layout()
file_path = 'xgboost_importancia.png'
plt.savefig(file_path, dpi=300)

print(f"✔️ Gráfico de jerarquía de datos creado exitosamente: {os.path.abspath(file_path)}")
print("\n[TIP] El modelo responde a la pregunta: 'Dado este resultado al descanso y estas cuotas previas al partido, ¿Quién se llevará la victoria al minuto 90?'")
