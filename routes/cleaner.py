from flask import Blueprint, request, jsonify
from src.data_cleaner import DataCleaner
from routes.state import pipeline

cleaner_bp = Blueprint('cleaner', __name__, url_prefix='/cleaner')


@cleaner_bp.route('/configure', methods=['POST'])
def configure():
    if pipeline['X_train'] is None:
        return jsonify({'error': 'Dati non splittati. Chiamare prima POST /split/split'}), 400

    body = request.get_json() or {}
    cleaner = DataCleaner()

    if body.get('fix_median', False):
        cleaner.fix_missing_numerical_median()

    unknown_cols = body.get('unknown_cols', None)
    cleaner.fix_missing_categorical_unknown(cols=unknown_cols)

    pipeline['cleaner'] = cleaner
    return jsonify({'status': 'configured'})


@cleaner_bp.route('/fit_transform', methods=['POST'])
def fit_transform():
    cleaner = pipeline['cleaner']
    X_train = pipeline['X_train']
    if cleaner is None or X_train is None:
        return jsonify({'error': 'Chiamare prima POST /cleaner/configure'}), 400

    pipeline['X_train'] = cleaner.fit_transform(X_train)
    return jsonify({'status': 'ok', 'shape': list(pipeline['X_train'].shape)})


@cleaner_bp.route('/transform', methods=['POST'])
def transform():
    cleaner = pipeline['cleaner']
    X_test = pipeline['X_test']
    if cleaner is None or X_test is None:
        return jsonify({'error': 'Chiamare prima POST /cleaner/fit_transform'}), 400

    pipeline['X_test'] = cleaner.transform(X_test)
    return jsonify({'status': 'ok', 'shape': list(pipeline['X_test'].shape)})


@cleaner_bp.route('/report', methods=['GET'])
def report():
    X_train = pipeline['X_train']
    if X_train is None:
        return jsonify({'error': 'Dati non disponibili'}), 400

    total = len(X_train)
    missing = X_train.isnull().sum()
    missing = missing[missing > 0]
    return jsonify({
        col: {'count': int(count), 'pct': round(count / total * 100, 2)}
        for col, count in missing.items()
    } if not missing.empty else {})
