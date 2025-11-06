import pandas as pd
from pytrends.request import TrendReq
import os
import time

# --- Configuration ---
tickers_list = [
    "NVDA", "AMD", "TSLA", "COIN", "SHOP", "SQ", "PLTR", "SNOW", "NET", "U",
    "RIVN", "LCID", "PLUG", "ENPH", "MRNA", "CRSP", "TDOC", "AMC", "GME", "SPCE"
]

# Initialiser l'objet pytrends
pytrends = TrendReq(hl='en-US', tz=360)

# Créer les dossiers de sortie si nécessaire
output_dir_final = 'donnees_finales'
if not os.path.exists(output_dir_final):
    os.makedirs(output_dir_final)
    
# --- Boucle sur tous les tickers ---
for ticker in tickers_list:
    print(f"--- Traitement de : {ticker} ---")
    
    # Chemin vers le fichier enrichi
    enriched_file_path = os.path.join('donnees_enrichies', f'{ticker}_enriched_data.csv')
    
    if not os.path.exists(enriched_file_path):
        print(f"Fichier non trouvé : {enriched_file_path}. Passage au suivant.")
        continue

    # --- 1. Charger les données enrichies (SMA, RSI) ---
    df = pd.read_csv(enriched_file_path, index_col='Date', parse_dates=True)

    # --- 2. Récupérer les données de Google Trends ---
    try:
        # Définir le mot-clé et la période de recherche
        # Le mot-clé est le symbole de l'action, ex: '$TSLA'
        # La timeframe doit correspondre exactement à nos données
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        timeframe = f'{start_date} {end_date}'
        
        print(f"Récupération des données Google Trends pour '{ticker}' de {start_date} à {end_date}...")

        # Construire la requête
        pytrends.build_payload(kw_list=[ticker], timeframe=timeframe, geo='US')
        
        # Obtenir les données d'intérêt
        trend_df = pytrends.interest_over_time()

        if trend_df.empty:
            print(f"Aucune donnée Google Trends trouvée pour {ticker}.")
            # On met un 0 par défaut si pas de données
            df['google_trend_score'] = 0
        else:
            # Nettoyer et fusionner les données
            trend_df = trend_df.drop(columns='isPartial')
            # Important: Google Trends renvoie souvent des données HEBDOMADAIRES sur de longues périodes.
            # On ré-échantillonne au jour le jour en propageant la dernière valeur connue (forward-fill)
            daily_trend = trend_df.reindex(df.index, method='ffill')
            
            # Ajouter la colonne au DataFrame principal
            df['google_trend_score'] = daily_trend[ticker]
            # Remplir les quelques NaN restants au début avec 0
            df['google_trend_score'].fillna(0, inplace=True) 
            print("Données de Google Trends fusionnées avec succès.")

    except Exception as e:
        print(f"Erreur lors de la récupération des données Google Trends pour {ticker}: {e}")
        # En cas d'erreur, on continue avec un score de 0 pour ne pas bloquer le script
        df['google_trend_score'] = 0

    # --- 3. Créer la colonne Cible ('Target') ---
    df['future_close'] = df['Close'].shift(-1)
    df['Target'] = (df['future_close'] > df['Close']).astype(int)
    
    # Nettoyage
    df.dropna(inplace=True)
    df = df.drop('future_close', axis=1)

    # --- 4. Sauvegarder le fichier final ---
    final_file_path = os.path.join(output_dir_final, f"{ticker}_final_data.csv")
    df.to_csv(final_file_path)
    
    print(f"Fichier final sauvegardé : {final_file_path}")
    print(f"Aperçu des dernières lignes pour {ticker}:\n{df.tail(3)}\n")

    # Ajouter une pause pour ne pas surcharger les serveurs de Google
    time.sleep(2)

print("--- Traitement de toutes les actions terminé ! ---")