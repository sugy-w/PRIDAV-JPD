import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight


def create_weekly_dataset(data):
    X = []
    y = []
    
    # Spracujeme každý cyklosčítač zvlášť
    for name, group in data.groupby('NAZOV'):
        # Zoradenie podľa dátumu a času
        g = group.set_index('datum_a_cas').sort_index()

        # Vyplníme chýbajúce hodiny nulami
        g = g.resample('h').sum(numeric_only=True).fillna(0)        
        # Rozdelenie na týždne (začiatok v pondelok)
        weekly_groups = g.groupby(pd.Grouper(freq='W-MON'))
        
        for date, week_data in weekly_groups:
            # Berieme len kompletné týždne (168 hodín)
            if len(week_data) == 168:
                # Zreťazíme "počet_z" a "počet_do" za sebou
                feat_z = week_data['POCET_Z'].values
                feat_do = week_data['POCET_DO'].values
                
                # Výsledný vektor má dĺžku 336 (168+168)
                features = np.concatenate([feat_z, feat_do])
                
                X.append(features)
                y.append(name) # Cieľová premenná (label)
                
    return np.array(X), np.array(y)

def train_test_split_each_route(X,y, test_size=0.25):
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    # Získame unikátne názvy trás
    unikatne_trasy = np.unique(y)

    for trasa in unikatne_trasy:
        # 1. Nájdeme indexy, kde sa nachádza konkrétna trasa
        indexy_trasy = [i for i, x in enumerate(y) if x == trasa]
        
        # 2. Vyberieme dáta len pre túto trasu
        X_trasa = X[indexy_trasy]
        y_trasa = y[indexy_trasy]
        
        # 3. Určíme bod rozdelenia (napr. prvých 75% týždňov do tréningu)
        pocet_tyzdnov = len(X_trasa)
        
        # Ochrana pre veľmi krátke dáta (ak má trasa menej ako 2 týždne, nedá sa deliť)
        if pocet_tyzdnov < 2:
            continue 
            
        split_point = int(pocet_tyzdnov * (1  - test_size))
        
        # 4. Rozdelíme chronologicky
        # Trénujeme na začiatku roka, testujeme na konci roka - PRE TÚTO TRASU
        X_train_list.append(X_trasa[:split_point])
        X_test_list.append(X_trasa[split_point:])
        
        y_train_list.append(y_trasa[:split_point])
        y_test_list.append(y_trasa[split_point:])

    # 5. Spojíme všetko dokopy do finálnych datasetov
    X_train = np.concatenate(X_train_list)
    X_test = np.concatenate(X_test_list)
    y_train = np.concatenate(y_train_list)
    y_test = np.concatenate(y_test_list)
    return X_train, X_test, y_train, y_test

def classify(y_train, y_test):
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    class_names = encoder.classes_
    print(f"Klasifikujeme tieto lokality: {class_names}", len(class_names))
    y_train_indices = np.argmax(y_train, axis=1)

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_indices),
        y=y_train_indices
    )
    class_weights_dict = dict(enumerate(class_weights))
    return y_train, y_test, class_weights_dict, class_names


def download_link(url):
  '''Method reformates the share link from Google Drive into fetchable form'''
  return 'https://drive.google.com/uc?id=' + url.split('/')[-2]

def make_cycling_data():
    cd = pd.read_csv(download_link("https://drive.google.com/file/d/15eHai6zkPwOBMq59n8uIjjohuuiaV8DF/view?usp=sharing"))
    cd['datum_a_cas'] = pd.to_datetime(cd['DATUM_A_CAS'])
    cd['hodina'] = cd['datum_a_cas'].dt.hour
    cd['den_v_tyzdni'] = cd['datum_a_cas'].dt.dayofweek
    cd['mesiac'] = cd['datum_a_cas'].dt.month
    cd['spolu'] = cd['POCET_Z'] + cd['POCET_DO']
    mapping = {
        'Cyklomost Slobody': 'Cyklomost',
        'Hradza Berg': 'Hradza',
        'Viedenska': 'Viedenska',
        'Devinska Nova Ves': 'Devinska Nova Ves',
        '#1 - Starý Most': 'Starý Most',
        '#2 - Starý most 2': 'Starý Most',
        'Starý most 2': 'Starý Most',
        '#3 - River Park': 'River Park',
        '#4 - Dolnozemská': 'Dolnozemská',
        '#5 - Devínska cesta': 'Devínska cesta',
        '#6 - Vajnorská': 'Vajnorská',
        '#7 - Vajnorská > NTC': 'Vajnorská',
        '#8 - Most SNP': 'Most SNP',
        '#9 - Páričkova': 'Páričkova',
        '#10 - Dunajská': 'Dunajská',
        '#11 - Most Apollo': 'Most Apollo',
        '#12 - Železná studnička': 'Železná studnička',
        '#13 - Vajanského 1': 'Vajanského',
        '#14 - Vajanského 2': 'Vajanského',
        '#15 - Incheba Einsteinova': 'Einsteinova',
        '#16 - Trenčianska': 'Trenčianska',
        '#17 - Dunajská/Lazaretská': 'Dunajská',
    }

    cd['NAZOV'] = cd['NAZOV'].replace(mapping)
    return cd