import pandas as pd
import pandas_ta as ta
import os

tickers_list = [
    "NVDA", "AMD", "TSLA", "COIN", "SHOP", "SQ", "PLTR", "SNOW", "NET", "U",
    "RIVN", "LCID", "PLUG", "ENPH", "MRNA", "CRSP", "TDOC", "AMC", "GME", "SPCE"
]

# --- Boucle sur tous les tickers ---
for ticker in tickers_list:
    file_path = os.path.join('donnees_bourse', f'{ticker}_data.csv')

    if not os.path.exists(file_path):
        print(f"Le fichier {file_path} n'a pas été trouvé.")
        continue # Passe au ticker suivant

    try:
        print(f"--- Traitement de {ticker} ---")
        
        # --- Ligne Corrigée ---
        # On dit à Pandas de :
        # header=0: utiliser la première ligne comme en-tête
        # index_col=0: utiliser la première colonne (position 0) comme index
        # parse_dates=True: convertir cet index en vraies dates
        # skiprows=[1, 2]: sauter la deuxième et la troisième ligne du fichier
        df = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True, skiprows=[1, 2])
        
        # Renommer l'index pour plus de clarté
        df.index.name = 'Date'
        
        print(f"Données pour {ticker} chargées. Dimensions : {df.shape}")

        # --- Calcul des Indicateurs Techniques (aucun changement ici) ---
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        
        # Nettoyage des valeurs manquantes (NaN)
        df.dropna(inplace=True)
        
        # --- Affichage du résultat ---
        print("\nAperçu des données enrichies :")
        print(df.tail(3))
        
        # --- Sauvegarde du nouveau fichier enrichi ---
        output_dir_enriched = 'donnees_enrichies'
        if not os.path.exists(output_dir_enriched):
            os.makedirs(output_dir_enriched)
            
        enriched_file_path = os.path.join(output_dir_enriched, f"{ticker}_enriched_data.csv")
        df.to_csv(enriched_file_path)
        
        print(f"Données enrichies pour {ticker} sauvegardées dans : {enriched_file_path}\n")

    except Exception as e:
        print(f"Une erreur est survenue lors du traitement de {ticker}: {e}\n")