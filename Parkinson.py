import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import streamlit as st
df = pd.read_csv('parkinsons.data')
y=df["status"]
X=df.drop(columns=['status','name'])
from sklearn.linear_model import Ridge
matrice_correlation=X.corr()
matrice_correlation=matrice_correlation.abs()
groupes_correles = []
def into(x,list):
    for k in list:
        if x in k:
            return True
    return False
for colonne in matrice_correlation.columns:
    if not into(colonne,groupes_correles):
        groupe_correle = {colonne}
        for autre_colonne in matrice_correlation.columns:
            if colonne != autre_colonne and matrice_correlation[colonne][autre_colonne] > 0.65:
                groupe_correle.add(autre_colonne)
        if len(groupe_correle) > 1:
            groupes_correles.append(groupe_correle)

groupes_finaux = []
for groupe in groupes_correles:
    if groupe not in groupes_finaux:
        groupes_finaux.append(groupe)


model = Ridge(alpha=1)
for groupe in groupes_finaux:
    print(groupes_finaux)
    X0 = df[list(groupe)]  # Convertir l'ensemble en liste pour l'accès aux colonnes du dataframe
    y0 = df[list(groupe)[0]]
    model.fit(X0, y0)
    coefficients = model.coef_
    attribut_principal = list(groupe)[np.abs(coefficients).argmax()]
    groupe.remove(attribut_principal)
    for l in groupe:
        try:
            X = X.drop(columns=[l])
        except KeyError:
            print("Certaines colonnes n'existent pas dans le DataFrame X.")
scaler = StandardScaler()
st.title('Formulaire de données')

# Création des champs d'entrée pour chaque colonne
form_data = {}
for col in X.columns:
    if X[col].dtype == 'int64':
        form_data[col] = st.number_input(f'Entrez {col}', value=int(X[col].mean()))
    elif X[col].dtype == 'float64':
        form_data[col] = st.number_input(f'Entrez {col}', value=float(X[col].mean()))
    elif X[col].dtype == 'object':
        form_data[col] = st.text_input(f'Entrez {col}', value=str(X[col].iloc[0]))

# Bouton de soumission
if st.button('Soumettre'):
    new_data = pd.DataFrame([form_data])
    
    # Concaténer la nouvelle DataFrame avec X
    X = pd.concat([X, new_data], ignore_index=True)
    
    # Convertir X en DataFrame pour pouvoir utiliser iloc
    X_df = pd.DataFrame(X, columns=X.columns)

    # Appliquer la transformation sur X
    X_scaled = scaler.fit_transform(X_df)

    # Diviser X_scaled en X_train et X_test
    X_train = pd.DataFrame(X_scaled[:, :-1])
    X_test = pd.DataFrame(X_scaled[:, -1])
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(X_train, y)
    from sklearn.linear_model import LogisticRegression
    #la descente de gradient est utilisée automatiquement dans scikit-learn
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y)
    y_knn_pred = knn.predict(X_test)
    y_pred_log_reg = log_reg.predict(X_test)
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import log_loss

# Créer un calibrateur de probabilité
    calibrated_lr = CalibratedClassifierCV(log_reg, method='sigmoid', cv='prefit')
    calibrated_lr.fit(X_train, y)
    calibrated_probs = calibrated_lr.predict_proba(X_test)
    logloss = log_loss(y, calibrated_probs)
    y_pred_calibrated = calibrated_lr.predict(X_test)
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    model = GaussianNB(var_smoothing=1)
    model.fit(X_train, y)
    y_pred_NB = model.predict(X_test)
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.optimizers import Adam
    from tqdm import tqdm
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    model.add(Dense(2, input_dim=20, activation='selu'))
    model.add(Dense(3, activation='linear'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Entraîner le modèle
    history = model.fit(X_train, y, epochs=50, batch_size=10, validation_split=0.2)
    y_pred_NN = model.predict(X_test)
    y_pred_NN = (y_pred_NN > 0.5).astype(int).flatten()
    y_pred_NN = np.array(y_pred_NN)
    y_pred_NB = np.array(y_pred_NB)
    y_pred_calibrated = np.array(y_pred_calibrated)
    y_knn_pred = np.array(y_knn_pred)
    y_pred_log_reg = np.array(y_pred_log_reg)
    predictions = np.vstack((y_pred_NN, y_pred_NB, y_pred_calibrated, y_knn_pred,y_pred_log_reg)).T
    y_pred_ensemble = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)
    with st.expander("Afficher la DataFrame"):
        st.success(df)