import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import glob
from tqdm import tqdm

# --- 1. CONFIGURATION ---

# Le DOSSIER contenant les fichiers CSV de Kaggle
KAGGLE_FOLDER_PATH = 'donnees_headlines' 
OUTPUT_CSV_FILE = 'daily_news_sentiment.csv'

# NOTRE BIBLIOTHÈQUE DE MOTS-CLÉS
TICKER_KEYWORDS = {
    "NVDA": ["nvidia", " nvda "],
    "AMD": ["amd", "advanced micro devices"],
    "TSLA": ["tesla", "tsla"],
    "COIN": ["coinbase", "coin"],
    "SHOP": ["shopify", "shop"],
    "SQ": ["square", "block, inc.", " sq "],
    "PLTR": ["palantir", "pltr"],
    "SNOW": ["snowflake", "snow"],
    "NET": ["cloudflare", " net "],
    "U": ["unity software", "unity", " u "],
    "RIVN": ["rivian", "rivn"],
    "LCID": ["lucid", "lcid"],
    "PLUG": ["plug power", "plug"],
    "ENPH": ["enphase", "enph"],
    "MRNA": ["moderna", "mrna"],
    "CRSP": ["crispr", "crsp"],
    "TDOC": ["teladoc", "tdoc"],
    "AMC": ["amc entertainment", " amc "],
    "GME": ["gamestop", "gme"],
    "SPCE": ["virgin galactic", "spce"]
}


# --- 2. CHARGEMENT, COMBINAISON ET HARMONISATION ---

if not os.path.exists(KAGGLE_FOLDER_PATH):
    print(f"ERREUR : Le dossier '{KAGGLE_FOLDER_PATH}' est introuvable.")
    exit()

all_news_files = glob.glob(os.path.join(KAGGLE_FOLDER_PATH, "*.csv"))
if not all_news_files:
    print(f"ERREUR : Aucun fichier CSV trouvé dans le dossier '{KAGGLE_FOLDER_PATH}'.")
    exit()

print(f"Trouvé {len(all_news_files)} fichiers de news. Chargement et combinaison...")
list_of_dfs = [pd.read_csv(file) for file in all_news_files]
df_news = pd.concat(list_of_dfs, ignore_index=True)
print(f"Combinaison réussie. {len(df_news)} titres au total. Traitement des colonnes...")

# Harmonisation des noms de colonnes possibles (minuscules/majuscules)
column_mapping = {
    'Headlines': 'headline', 'headlines': 'headline',
    'Time': 'time', 'time': 'time', 'Date': 'time',
    'Description': 'description', 'description': 'description',
}
df_news.rename(columns=lambda c: column_mapping.get(c, c), inplace=True)


# --- 3. FILTRAGE ---

# Gérer le cas où la colonne 'description' n'existerait pas dans tous les fichiers
if 'description' not in df_news.columns:
    df_news['description'] = ''
df_news['text_to_search'] = (df_news['headline'].astype(str) + ' ' + df_news['description'].fillna('').astype(str)).str.lower()
df_news.dropna(subset=['time', 'text_to_search'], inplace=True)

filtered_news = []
print("Filtrage des news pour trouver les actions pertinentes... (cela peut prendre plusieurs minutes)")

for index, row in tqdm(df_news.iterrows(), total=df_news.shape[0]):
    text = row['text_to_search']
    for ticker, keywords in TICKER_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            filtered_news.append({
                'date': pd.to_datetime(row['time'], errors='coerce').date(),
                'ticker': ticker,
                'headline': row['headline']
            })
            # On passe au suivant dès qu'un ticker est trouvé pour éviter les doublons pour une même news
            break 

# --- 4. ANALYSE ET AGRÉGATION ---
if filtered_news:
    df_filtered = pd.DataFrame(filtered_news)
    df_filtered.dropna(subset=['date'], inplace=True) # Retirer les dates invalides
    
    print(f"\nFiltrage terminé. {len(df_filtered)} titres de presse pertinents trouvés.")
    
    print("Analyse du sentiment sur les titres filtrés...")
    vader_analyzer = SentimentIntensityAnalyzer()
    
    df_filtered['sentiment_score'] = df_filtered['headline'].apply(
        lambda headline: vader_analyzer.polarity_scores(str(headline))['compound']
    )
    
    print("Agrégation des scores de sentiment par jour et par action...")
    df_daily_sentiment = df_filtered.groupby(['date', 'ticker'])['sentiment_score'].mean().reset_index()
    
    df_daily_sentiment.rename(columns={'date': 'Date', 'ticker': 'Ticker', 'sentiment_score': 'NewsSentiment'}, inplace=True)
    
    # --- 5. SAUVEGARDE ---
    df_daily_sentiment.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"\nSUCCÈS ! Les données ont été sauvegardées dans '{OUTPUT_CSV_FILE}'.")
    print("Aperçu du résultat :")
    print(df_daily_sentiment.tail())
else:
    print("\nAVERTISSEMENT : Aucune nouvelle pertinente n'a été trouvée pour les tickers spécifiés.")