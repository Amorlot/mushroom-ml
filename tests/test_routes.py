import pytest
from app import app as flask_app
from routes.state import pipeline


@pytest.fixture(autouse=True)
def reset_state():
    for key in list(pipeline.keys()):
        pipeline[key] = None
    yield


@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as c:
        yield c


# ── Errori senza prerequisiti ────────────────────────────────────────────────

def test_loader_info_senza_load(client):
    assert client.get('/loader/info').status_code == 400

def test_loader_missing_senza_load(client):
    assert client.get('/loader/missing').status_code == 400

def test_eda_info_senza_load(client):
    assert client.get('/eda/info').status_code == 400

def test_split_senza_load(client):
    assert client.post('/split/split', json={}).status_code == 400

def test_cleaner_configure_senza_split(client):
    assert client.post('/cleaner/configure', json={}).status_code == 400

def test_cleaner_fit_transform_senza_configure(client):
    assert client.post('/cleaner/fit_transform', json={}).status_code == 400

def test_cleaner_transform_senza_fit(client):
    assert client.post('/cleaner/transform', json={}).status_code == 400

def test_encoder_fit_senza_dati(client):
    assert client.post('/encoder/fit_transform_features', json={}).status_code == 400

def test_encoder_transform_senza_fit(client):
    assert client.post('/encoder/transform_features', json={}).status_code == 400

def test_encoder_encode_target_senza_fit(client):
    assert client.post('/encoder/encode_target', json={}).status_code == 400

def test_logreg_train_senza_dati(client):
    assert client.post('/logreg/train', json={}).status_code == 400

def test_logreg_evaluate_senza_train(client):
    assert client.get('/logreg/evaluate').status_code == 400

def test_logreg_report_senza_train(client):
    assert client.get('/logreg/classification_report').status_code == 400

def test_xgboost_train_senza_dati(client):
    assert client.post('/xgboost/train', json={}).status_code == 400

def test_xgboost_evaluate_senza_train(client):
    assert client.get('/xgboost/evaluate').status_code == 400


# ── Integrazione completa (richiede rete, lento per il training) ─────────────

@pytest.mark.slow
def test_pipeline_completo_logreg(client):
    # 1. Load
    r = client.post('/loader/load', json={
        'dataset_id': 73,
        'target_col': 'poisonous',
    })
    assert r.status_code == 200
    assert r.get_json()['rows'] == 8124

    # 2. Info e missing
    assert client.get('/loader/info').status_code == 200
    assert client.get('/loader/missing').status_code == 200

    # 3. EDA
    r = client.get('/eda/info')
    assert r.status_code == 200
    assert 'variabili' in r.get_json()

    # 4. Split
    r = client.post('/split/split', json={})
    assert r.status_code == 200
    data = r.get_json()
    assert data['train_rows'] == 6499
    assert data['test_rows'] == 1625

    # 5. Cleaner
    r = client.post('/cleaner/configure', json={'unknown_cols': ['stalk-root']})
    assert r.status_code == 200
    assert client.post('/cleaner/fit_transform', json={}).status_code == 200
    assert client.post('/cleaner/transform', json={}).status_code == 200
    r = client.get('/cleaner/report')
    assert r.status_code == 200

    # 6. Encoder
    assert client.post('/encoder/fit_transform_features', json={}).status_code == 200
    assert client.post('/encoder/transform_features', json={}).status_code == 200
    assert client.post('/encoder/encode_target', json={}).status_code == 200

    # 7. Logreg (cv=2 per velocizzare il test)
    r = client.post('/logreg/train', json={'cv': 2})
    assert r.status_code == 200
    assert 'best_params' in r.get_json()

    r = client.get('/logreg/evaluate')
    assert r.status_code == 200
    metrics = r.get_json()
    assert set(metrics.keys()) == {'accuracy', 'precision', 'recall', 'f1_score'}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())

    r = client.get('/logreg/classification_report')
    assert r.status_code == 200
    assert 'report' in r.get_json()