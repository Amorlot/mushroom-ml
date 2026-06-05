from flask import Blueprint, request, jsonify
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.model_xgboost import ModelXGBoost
from routes.state import pipeline

xgboost_bp = Blueprint('xgboost', __name__, url_prefix='/xgboost')


@xgboost_bp.route('/train', methods=['POST'])
def train():
    X_train = pipeline['X_train']
    y_train = pipeline['y_train']
    if X_train is None or y_train is None:
        return jsonify({'error': 'Dati non disponibili. Completare prima il pipeline di encoding'}), 400

    model = ModelXGBoost()
    model.train(X_train, y_train)
    pipeline['xgboost'] = model

    return jsonify({'status': 'ok', 'best_params': model.best_params})


@xgboost_bp.route('/evaluate', methods=['GET'])
def evaluate():
    model = pipeline['xgboost']
    X_test = pipeline['X_test']
    y_test = pipeline['y_test']
    if model is None or X_test is None or y_test is None:
        return jsonify({'error': 'Eseguire prima POST /xgboost/train'}), 400

    y_pred = model.model.predict(X_test)
    return jsonify({
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall':    recall_score(y_test, y_pred, average='weighted'),
        'f1_score':  f1_score(y_test, y_pred, average='weighted'),
    })
