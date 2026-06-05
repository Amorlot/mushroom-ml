# Mushroom ML

Progetto di classificazione funghi velenosi/commestibili sul dataset UCI Mushroom (id=73).  
Implementa un pipeline completo di preprocessing, encoding e training con Logistic Regression e XGBoost, esposto sia come script CLI che come API REST Flask.

---

## Struttura del progetto

```
mushroom-ml/
├── app.py                     # Entry point Flask
├── main.py                    # Entry point CLI (pipeline completo)
├── requirements.txt
│
├── src/
│   ├── data_loader.py         # Caricamento dataset da UCI
│   ├── data_cleaner.py        # Pulizia valori mancanti (fit/transform)
│   ├── split.py               # Train/test split stratificato 80/20
│   ├── encoder.py             # OHE feature + LabelEncoder target
│   ├── eda.py                 # Analisi esplorativa
│   ├── logreg.py              # Logistic Regression con GridSearchCV
│   └── model_xgboost.py       # XGBoost con GridSearchCV
│
├── routes/
│   ├── state.py               # Stato globale condiviso tra gli endpoint
│   ├── loader.py              # /loader/*
│   ├── split.py               # /split/*
│   ├── cleaner.py             # /cleaner/*
│   ├── encoder.py             # /encoder/*
│   ├── eda.py                 # /eda/*
│   ├── logreg.py              # /logreg/*
│   └── xgboost.py             # /xgboost/*
│
└── tests/
    ├── test_models.py         # Test DataLoader e Logreg
    └── test_routes.py         # Test endpoint Flask
```

---

## Installazione

```bash
pip install -r requirements.txt
```

---

## Utilizzo

### CLI

Esegue l'intero pipeline in sequenza e stampa i risultati a terminale:

```bash
python main.py
```

### API REST

Avvia il server Flask:

```bash
python app.py
```

Il server è disponibile su `http://localhost:5000`.  
Gli endpoint vanno chiamati **nell'ordine indicato** poiché il pipeline è stateful.

---

## Endpoint API

### DataLoader — `/loader`

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/loader/load` | Carica il dataset da UCI |
| GET | `/loader/info` | Info su righe, colonne, distribuzione target |
| GET | `/loader/missing` | Valori mancanti per colonna |

**POST /loader/load** — body JSON opzionale:
```json
{
  "dataset_id": 73,
  "target_col": "poisonous",
  "drop_cols": ["veil-type"],
  "drop_missing_thresh": null
}
```

---

### Split — `/split`

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/split/split` | Divide in train/test 80/20 stratificato |

---

### EDA — `/eda`

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| GET | `/eda/info` | Shape e lista variabili del dataset |

---

### DataCleaner — `/cleaner`

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/cleaner/configure` | Configura la strategia di pulizia |
| POST | `/cleaner/fit_transform` | Fit + transform su X_train |
| POST | `/cleaner/transform` | Transform su X_test |
| GET | `/cleaner/report` | Valori mancanti residui su X_train |

**POST /cleaner/configure** — body JSON opzionale:
```json
{
  "fix_median": false,
  "unknown_cols": ["stalk-root"]
}
```
Se `unknown_cols` è `null`, la pulizia viene applicata automaticamente a tutte le colonne categoriali con mancanti.

---

### Encoder — `/encoder`

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/encoder/fit_transform_features` | Fit OHE su X_train + transform |
| POST | `/encoder/transform_features` | Transform X_test con OHE già fittato |
| POST | `/encoder/encode_target` | Label encoding di y_train e y_test |

---

### Logistic Regression — `/logreg`

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/logreg/train` | Training con GridSearchCV |
| GET | `/logreg/evaluate` | Metriche su test set |
| GET | `/logreg/classification_report` | Report completo per classe |

**POST /logreg/train** — body JSON opzionale:
```json
{
  "cv": 5,
  "scoring": "f1_weighted"
}
```

---

### XGBoost — `/xgboost`

| Metodo | Endpoint | Descrizione |
|--------|----------|-------------|
| POST | `/xgboost/train` | Training con GridSearchCV |
| GET | `/xgboost/evaluate` | Metriche su test set |

---

## Ordine corretto delle chiamate

```
POST /loader/load
POST /split/split
POST /cleaner/configure  →  POST /cleaner/fit_transform  →  POST /cleaner/transform
POST /encoder/fit_transform_features  →  POST /encoder/transform_features  →  POST /encoder/encode_target
POST /logreg/train    e/o    POST /xgboost/train
GET  /logreg/evaluate        GET  /xgboost/evaluate
```

Ogni endpoint restituisce HTTP 400 con un messaggio d'errore se viene chiamato prima del suo prerequisito.

---

## Test

```bash
# Test veloci (senza rete, senza training)
pytest tests/test_routes.py -m "not slow"

# Test di integrazione completo (richiede rete e tempo per il training)
pytest tests/test_routes.py -m slow

# Tutti i test
pytest tests/
```

I test in `test_routes.py` sono divisi in:
- **Test di validazione** — verificano che ogni endpoint restituisca 400 se chiamato senza i prerequisiti (veloci, nessuna dipendenza esterna)
- **`test_pipeline_completo_logreg`** — esegue l'intero pipeline dalla chiamata di caricamento fino alla valutazione del modello (richiede connessione a UCI, lento per il GridSearchCV)
