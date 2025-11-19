import numpy as np
import pandas as pd
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.utils import class_weight
from sklearn.metrics import matthews_corrcoef, classification_report
import joblib
from tqdm import tqdm

# --- CONFIG SOTA ---
DATA_DIR = 'data_processed'
SEQ_LEN = 60  # Fenêtre de contexte
EMBED_DIM = 64  # Dimension des vecteurs
NUM_HEADS = 4   # Têtes d'attention
FF_DIM = 128    # Feed Forward network dimension
NUM_LAYERS = 3  # Profondeur du Transformer
DROPOUT = 0.15
BATCH_SIZE = 128 # Les Transformers aiment les gros batchs
EPOCHS = 40
FUTURE_SPLIT_DATE = '2024-06-01'

# --- SETUP GPU ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

# --- 1. DATA LOADING (AVEC FEATURES SÉLECTIONNÉES) ---
def load_data():
    if not os.path.exists('selected_features.pkl'):
        print("ERREUR: Lance 'select_features.py' d'abord !")
        exit()
    
    selected_features = joblib.load('selected_features.pkl')
    print(f"Utilisation des {len(selected_features)} meilleures features : {selected_features}")
    
    files = glob.glob(os.path.join(DATA_DIR, "*_final.csv"))
    train_slices, test_slices = [], []
    
    for file in tqdm(files, desc="Loading"):
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # IMPORTANT: On ne garde que les features élues
        df_selected = df[selected_features].copy()
        df_selected['Target'] = df['Target'] # On garde la cible quand même
        
        train = df_selected[df_selected.index < FUTURE_SPLIT_DATE]
        test = df_selected[df_selected.index >= FUTURE_SPLIT_DATE]
        
        if len(train) > SEQ_LEN: train_slices.append(train)
        if len(test) > SEQ_LEN: test_slices.append(test)

    # Scaling sophistiqué (QuantileTransformer gère mieux les outliers financiers que RobustScaler)
    full_train = pd.concat(train_slices)
    scaler = QuantileTransformer(output_distribution='normal')
    scaler.fit(full_train[selected_features])
    joblib.dump(scaler, 'scaler_transformer.pkl')
    
    return train_slices, test_slices, scaler, selected_features

def make_dataset(slices, scaler, features, batch_size):
    X_list, y_list = [], []
    for df in slices:
        data = scaler.transform(df[features])
        target = df['Target'].values
        # Sliding window numpy optimisée
        # (Code simplifié pour lisibilité, en prod on utiliserait tf.data.Dataset generator)
        for i in range(len(data) - SEQ_LEN):
            X_list.append(data[i:i+SEQ_LEN])
            y_list.append(target[i+SEQ_LEN-1])
            
    X = np.array(X_list, dtype='float32')
    y = np.array(y_list, dtype='int32')
    
    return X, y

# --- 2. ARCHITECTURE TRANSFORMER ENCODER ---
# C'est ici que la magie opère 

#[Image of Transformer Encoder Architecture]

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs # Connexion résiduelle (Skip Connection)

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer_model(input_shape, n_classes):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Empilement des blocs Transformer
    for _ in range(NUM_LAYERS):
        x = transformer_encoder(x, head_size=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM, dropout=DROPOUT)

    # Pooling pour "résumer" la séquence temporelle en un vecteur
    x = layers.GlobalAveragePooling1D()(x)
    
    # Tête de classification (MLP)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Optimiseur avec Weight Decay (AdamW) pour éviter l'overfitting SOTA
    optimizer = optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# --- MAIN ---
if __name__ == "__main__":
    train_slices, test_slices, scaler, features = load_data()
    
    print("Préparation des tenseurs...")
    X_train, y_train = make_dataset(train_slices, scaler, features, BATCH_SIZE)
    X_test, y_test = make_dataset(test_slices, scaler, features, BATCH_SIZE)
    
    # Poids des classes
    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw))
    print(f"Poids: {cw_dict}")

    model = build_transformer_model((SEQ_LEN, len(features)), 3)
    model.summary()

    cbs = [
        callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor='val_loss'),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6),
        callbacks.ModelCheckpoint('transformer_model_v3.keras', save_best_only=True)
    ]

    print("\n--- Démarrage Entraînement Transformer ---")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw_dict,
        callbacks=cbs
    )

    # Évaluation SOTA
    print("\n--- RÉSULTATS TRANSFORMER V3 ---")
    model.load_weights('transformer_model_v3.keras')
    preds = model.predict(X_test, batch_size=512)
    y_pred = np.argmax(preds, axis=1)
    
    print(classification_report(y_test, y_pred, target_names=['SELL', 'WAIT', 'BUY']))
    
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"🏆 MCC Global : {mcc:.4f}")
    
    y_test_bin = (y_test == 2).astype(int)
    y_pred_bin = (y_pred == 2).astype(int)
    mcc_buy = matthews_corrcoef(y_test_bin, y_pred_bin)
    print(f"🚀 MCC Signal Achat : {mcc_buy:.4f}")