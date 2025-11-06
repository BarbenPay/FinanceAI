import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os

# --- 0. CONFIGURATION ---
TICKER_TO_FINETUNE = 'TSLA'
# On charge bien le modèle V2
GENERAL_MODEL_PATH = 'financial_model_generalist_v2.keras'
FINETUNED_MODEL_SAVE_PATH = f'finetuned_model_{TICKER_TO_FINETUNE}_v2.keras'
NEWS_SENTIMENT_PATH = 'daily_news_sentiment.csv'

# On utilise la liste de 10 features du modèle V2 !
features = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_20', 'SMA_50', 'RSI_14', 'google_trend_score', 'NewsSentiment']

print(f"--- Affinage du modèle V2 pour : {TICKER_TO_FINETUNE} ---")
model = load_model(GENERAL_MODEL_PATH)

# --- 2. PRÉPARER LES DONNÉES SPÉCIFIQUES (AVEC LE SENTIMENT) ---
df_stock = pd.read_csv(os.path.join('donnees_finales', f'{TICKER_TO_FINETUNE}_final_data.csv'), index_col='Date', parse_dates=True)
df_sentiment = pd.read_csv(NEWS_SENTIMENT_PATH, parse_dates=['Date'])
df_sentiment_ticker = df_sentiment[df_sentiment['Ticker'] == TICKER_TO_FINETUNE].set_index('Date')[['NewsSentiment']]

# Fusionner les deux sources
df = df_stock.merge(df_sentiment_ticker, left_index=True, right_index=True, how='left')
df['NewsSentiment'].ffill(inplace=True)
df['NewsSentiment'].bfill(inplace=True)
df['NewsSentiment'].fillna(0, inplace=True)

df.dropna(inplace=True) # Important de le faire après la fusion et le remplissage

X = df[features]
y = df['Target']

# Scaling, Séquençage et Division (comme avant)
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
look_back = 60
X_sequences, y_sequences = [], []
for i in range(len(X_scaled) - look_back):
    X_sequences.append(X_scaled[i:i+look_back])
    y_sequences.append(y.iloc[i + look_back - 1])
X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=False)
print(f"Données prêtes. Shape des features d'entrée : {X_train.shape}")


# --- 3. RE-COMPILATION ET AFFINAGE ---
# On peut essayer avec un learning rate un tout petit peu plus grand pour voir si ça "débloque" le modèle
low_learning_rate_optimizer = Adam(learning_rate=0.0001) 
model.compile(optimizer=low_learning_rate_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- Début de l'affinage (V2) ---")
history = model.fit(
    X_train, y_train,
    epochs=15, # 15 epochs suffiront pour voir si ça bouge
    batch_size=32,
    validation_data=(X_test, y_test)
)

# --- 4. ÉVALUATION FINALE ---
print("\n--- Évaluation du modèle V2 affiné ---")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Précision finale : {accuracy*100:.2f}%")
model.save(FINETUNED_MODEL_SAVE_PATH)