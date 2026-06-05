from flask import Blueprint, request, jsonify
from src.logreg import Logreg
from routes.state import pipeline

logreg_bp = Blueprint('logreg', __name__, url_prefix='/logreg')


@logreg_bp.route('/train', methods=['POST'])
def train():
    X_train = pipeline['X_train']
    y_train = pipeline['y_train']
    if X_train is None or y_train is None:
        return jsonify({'error': 'Dati non disponibili. Completare prima il pipeline di encoding'}), 400

    body = request.get_json() or {}
    model = Logreg()
    model.train(X_train, y_train, cv=body.get('cv', 5), scoring=body.get('scoring', 'f1_weighted'))
    pipeline['logreg'] = model

    return jsonify({'status': 'ok', 'best_params': model.best_params})


@logreg_bp.route('/evaluate', methods=['GET'])
def evaluate():
    model = pipeline['logreg']
    X_test = pipeline['X_test']
    y_test = pipeline['y_test']
    if model is None or X_test is None or y_test is None:
        return jsonify({'error': 'Eseguire prima POST /logreg/train'}), 400

    return jsonify(model.evaluate(X_test, y_test))


@logreg_bp.route('/classification_report', methods=['GET'])
def classification_report():
    model = pipeline['logreg']
    X_test = pipeline['X_test']
    y_test = pipeline['y_test']
    if model is None or X_test is None or y_test is None:
        return jsonify({'error': 'Eseguire prima POST /logreg/train'}), 400

    return jsonify({'report': model.classification_report(X_test, y_test)})
