import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt # On importe KerasTuner !
import os
import glob

# --- PRÉPARATION DES DONNÉES (identique à train_general_model.py) ---
# ... (Copiez-collez toute la partie de préparation des données, jusqu'à la création de X_train, X_test, y_train, y_test)
# ... Je la remets ici pour que le script soit complet.

PATH_TO_DATA = 'donnees_finales'
all_files = glob.glob(os.path.join(PATH_TO_DATA, "*.csv"))
features = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'google_trend_score'] # Modèle V1
all_X_scaled_list = []
all_y_list = []
for file in all_files:
    df = pd.read_csv(file, index_col='Date', parse_dates=True)
    df.dropna(inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(df[features])
    all_X_scaled_list.append(X_scaled)
    all_y_list.append(df['Target'])

combined_X_scaled = np.concatenate(all_X_scaled_list, axis=0)
combined_y = pd.concat(all_y_list)

look_back = 60
X_sequences, y_sequences = [], []
for i in range(len(combined_X_scaled) - look_back):
    X_sequences.append(combined_X_scaled[i:i+look_back])
    y_sequences.append(combined_y.iloc[i + look_back - 1])
X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=True, random_state=42)
print("Données prêtes pour l'optimisation.")

# --- DÉFINITION DE L'HYPERMODÈLE ---

def build_model(hp):
    """
    Cette fonction construit le modèle et définit les hyperparamètres à chercher.
    'hp' est un objet qui nous permet de définir des plages de recherche.
    """
    model = Sequential()
    model.add(Input(shape=(look_back, len(features))))

    # Potard 1 : Nombre de neurones dans la 1ère couche LSTM
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=128, step=16)
    model.add(LSTM(units=hp_units_1, return_sequences=True))
    model.add(Dropout(0.2))

    # Potard 2 : Nombre de neurones dans la 2ème couche LSTM
    hp_units_2 = hp.Int('units_2', min_value=32, max_value=128, step=16)
    model.add(LSTM(units=hp_units_2, return_sequences=False))
    model.add(Dropout(0.2))

    # Potard 3 : Nombre de neurones dans la couche Dense
    hp_units_dense = hp.Int('units_dense', min_value=16, max_value=64, step=8)
    model.add(Dense(units=hp_units_dense, activation='relu'))
    
    model.add(Dense(1, activation='sigmoid'))

    # Potard 4 : Le taux d'apprentissage (Learning Rate)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# --- CONFIGURATION DU TUNER ---

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy', # La métrique à maximiser
    max_trials=10,             # Le nombre de combinaisons différentes à essayer
    executions_per_trial=1,    # Combien de fois entraîner chaque combinaison (pour la robustesse)
    directory='tuning_results',
    project_name='financial_model_v1'
)

# Affiche un résumé de l'espace de recherche
tuner.search_space_summary()


# --- LANCEMENT DE LA RECHERCHE ---

print("\n--- Début de la recherche des meilleurs hyperparamètres ---")
# ATTENTION : Cette étape peut être TRÈS longue !
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

print("\n--- Recherche terminée ---")


# --- RÉSULTATS ---
print("\n--- Meilleurs hyperparamètres trouvés : ---")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Unités LSTM 1: {best_hps.get('units_1')}")
print(f"Unités LSTM 2: {best_hps.get('units_2')}")
print(f"Unités Dense: {best_hps.get('units_dense')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")

# Récupérer et afficher le résumé du meilleur modèle
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Évaluer le meilleur modèle sur les données de test
print("\n--- Évaluation du meilleur modèle sur les données de test ---")
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Perte (Loss) : {loss:.4f}")
print(f"Précision (Accuracy) : {accuracy*100:.2f}%")

# Sauvegarder ce nouveau champion
best_model.save("financial_model_v1_tuned.keras")
print("\nMeilleur modèle sauvegardé avec succès !")