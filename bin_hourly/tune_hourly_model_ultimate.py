import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import os
import glob
from tqdm import tqdm

# --- 1. PRÉPARATION DES DONNÉES ULTRA-ENRICHIES ---
# (Avec un système de cache pour ne le faire qu'une fois)

INPUT_DIR = 'donnees_horaire_ULTRA_enrichies'
CACHE_DIR = 'data_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Définir les noms des fichiers cache
X_TRAIN_CACHE = os.path.join(CACHE_DIR, 'x_train.npy')
X_TEST_CACHE = os.path.join(CACHE_DIR, 'x_test.npy')
Y_TRAIN_CACHE = os.path.join(CACHE_DIR, 'y_train.npy')
Y_TEST_CACHE = os.path.join(CACHE_DIR, 'y_test.npy')
FEATURES_CACHE = os.path.join(CACHE_DIR, 'features.txt')

if all([os.path.exists(f) for f in [X_TRAIN_CACHE, X_TEST_CACHE, Y_TRAIN_CACHE, Y_TEST_CACHE]]):
    print("Chargement des données depuis le cache...")
    X_train = np.load(X_TRAIN_CACHE)
    X_test = np.load(X_TEST_CACHE)
    y_train = np.load(Y_TRAIN_CACHE)
    y_test = np.load(Y_TEST_CACHE)
    with open(FEATURES_CACHE, 'r') as f:
        features = f.read().split(',')
    look_back = X_train.shape[1]
    print(f"Données chargées depuis le cache. {len(features)} features détectées.")

else:
    print("Le cache n'a pas été trouvé. Préparation des données en cours...")
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    features = None
    all_X_scaled_list = []
    all_y_list = []
    for file in tqdm(all_files, desc="Chargement des fichiers"):
        df = pd.read_csv(file, index_col='datetime', parse_dates=True)
        df.dropna(inplace=True)
        if features is None:
            # On retire aussi 'adj close' s'il existe
            features_to_drop = ['Target', 'adj close']
            features = [col for col in df.columns if col not in features_to_drop]
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(df[features])
        all_X_scaled_list.append(X_scaled)
        all_y_list.append(df['Target'])

    combined_X_scaled = np.concatenate(all_X_scaled_list, axis=0)
    combined_y = pd.concat(all_y_list)
    look_back = 70
    X_sequences, y_sequences = [], []
    for i in tqdm(range(len(combined_X_scaled) - look_back), desc="Création Séquences"):
        X_sequences.append(combined_X_scaled[i:i+look_back])
        y_sequences.append(combined_y.iloc[i + look_back - 1])
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=True, random_state=42)
    
    # Sauvegarder les données préparées dans le cache
    print("Sauvegarde des données dans le cache pour les prochains lancements...")
    np.save(X_TRAIN_CACHE, X_train)
    np.save(X_TEST_CACHE, X_test)
    np.save(Y_TRAIN_CACHE, y_train)
    np.save(Y_TEST_CACHE, y_test)
    with open(FEATURES_CACHE, 'w') as f:
        f.write(','.join(features))

print(f"Données prêtes. Entraînement sur {len(X_train)} séquences, test sur {len(X_test)}.")


# --- 2. FONCTION DE CONSTRUCTION DU MODÈLE CORRIGÉE ---
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(look_back, len(features))))
    
    num_lstm_layers = hp.Int('num_lstm_layers', 1, 2)
    
    for i in range(num_lstm_layers):
        model.add(Bidirectional(LSTM(
            units=hp.Int(f'lstm_units_{i}', min_value=32, max_value=96, step=16),
            # ===== LA CORRECTION LOGIQUE CRUCIALE EST ICI =====
            # Mettre return_sequences=True, SAUF pour la toute dernière couche LSTM
            return_sequences=(i < num_lstm_layers - 1)
            # ===================================================
        )))
        model.add(Dropout(hp.Float(f'dropout_lstm_{i}', min_value=0.1, max_value=0.4, step=0.1)))

    for i in range(hp.Int('num_dense_layers', 0, 2)):
        model.add(Dense(
            units=hp.Int(f'dense_units_{i}', min_value=32, max_value=96, step=16),
            activation='relu'
        ))
        model.add(Dropout(hp.Float(f'dropout_dense_{i}', min_value=0.1, max_value=0.4, step=0.1)))

    model.add(Dense(1, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- 3. CONFIGURATION ET LANCEMENT DU TUNER ---
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=15,
    factor=3,
    directory='tuning_ultimate_results',
    project_name='hourly_model_corrected', # Nouveau nom de projet pour ne pas utiliser l'ancien cache
    overwrite=True # Important: recommence la recherche
)

tuner.search_space_summary()
print("\n--- Début de la recherche ULTIME (Corrigée) ---")

# On définit un batch size fixe et efficace
tuner.search(X_train, y_train, validation_data=(X_test, y_test), batch_size=128)

# --- 4. RÉSULTATS ---
print("\n--- Recherche terminée. ---")

# Utiliser un try/except car le tuner peut n'avoir trouvé aucun bon modèle
try:
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n--- Meilleurs hyperparamètres trouvés : ---")
    # Une manière propre d'afficher tous les hyperparamètres choisis
    print({hp: best_hps.get(hp) for hp in best_hps.values})

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary()

    print("\n--- Évaluation du modèle ULTIME ---")
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f"Perte (Loss) : {loss:.4f}")
    print(f"Précision (Accuracy) : {accuracy*100:.2f}%")

    best_model.save("ultimate_hourly_model.keras")
    print("\nMeilleur modèle sauvegardé avec succès !")
    
except IndexError:
    print("\nAVERTISSEMENT : La recherche n'a retourné aucun modèle performant.")
except Exception as e:
    print(f"\nUne erreur est survenue lors de l'affichage des résultats : {e}")