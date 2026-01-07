import pandas as pd
import numpy as np

def create_weekly_dataset(data):
    X = []
    y = []
    
    # Spracujeme každý cyklosčítač zvlášť
    for name, group in data.groupby('NAZOV'):
        # Nastavíme index a utriedime
        g = group.set_index('datum_a_cas').sort_index()
        
        # DÔLEŽITÉ: Resampling na hodinovú úroveň (aby sme mali fixnú mriežku)
        # Vyplníme chýbajúce hodiny nulami
        g = g.resample('h').sum(numeric_only=True).fillna(0)
        
        # Rozdelenie na týždne (začiatok v pondelok)
        weekly_groups = g.groupby(pd.Grouper(freq='W-MON'))
        
        for date, week_data in weekly_groups:
            # Berieme len kompletné týždne (168 hodín)
            if len(week_data) == 168:
                # Vytvoríme dlhý vektor príznakov:
                # Zreťazíme "počet_z" a "počet_do" za sebou
                feat_z = week_data['POCET_Z'].values
                feat_do = week_data['POCET_DO'].values
                
                # Výsledný vektor má dĺžku 336 (168+168)
                features = np.concatenate([feat_z, feat_do])
                
                X.append(features)
                y.append(name) # Cieľová premenná (label)
                
    return np.array(X), np.array(y)


def download_link(url):
  '''Method reformates the share link from Google Drive into fetchable form'''
  return 'https://drive.google.com/uc?id=' + url.split('/')[-2]

cycling_data = pd.read_csv(download_link("https://drive.google.com/file/d/15eHai6zkPwOBMq59n8uIjjohuuiaV8DF/view?usp=sharing"))
cycling_data['datum_a_cas'] = pd.to_datetime(cycling_data['DATUM_A_CAS'])
cycling_data['hodina'] = cycling_data['datum_a_cas'].dt.hour
cycling_data['den_v_tyzdni'] = cycling_data['datum_a_cas'].dt.dayofweek
cycling_data['mesiac'] = cycling_data['datum_a_cas'].dt.month
cycling_data['spolu'] = cycling_data['POCET_Z'] + cycling_data['POCET_DO']