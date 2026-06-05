from flask import Blueprint, jsonify
from src.encoder import Encoder
from routes.state import pipeline

encoder_bp = Blueprint('encoder', __name__, url_prefix='/encoder')


@encoder_bp.route('/fit_transform_features', methods=['POST'])
def fit_transform_features():
    X_train = pipeline['X_train']
    if X_train is None:
        return jsonify({'error': 'Dati non disponibili. Completare prima il pipeline fino a /cleaner'}), 400

    cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()
    encoder = Encoder()
    pipeline['X_train'] = encoder.fit_transform_features(X_train, cat_cols)
    pipeline['encoder'] = encoder
    return jsonify({'status': 'ok', 'shape': list(pipeline['X_train'].shape)})


@encoder_bp.route('/transform_features', methods=['POST'])
def transform_features():
    encoder = pipeline['encoder']
    X_test = pipeline['X_test']
    if encoder is None or X_test is None:
        return jsonify({'error': 'Chiamare prima POST /encoder/fit_transform_features'}), 400

    pipeline['X_test'] = encoder.transform_features(X_test)
    return jsonify({'status': 'ok', 'shape': list(pipeline['X_test'].shape)})


@encoder_bp.route('/encode_target', methods=['POST'])
def encode_target():
    encoder = pipeline['encoder']
    if encoder is None:
        return jsonify({'error': 'Chiamare prima POST /encoder/fit_transform_features'}), 400

    pipeline['y_train'] = encoder.encode_target(pipeline['y_train'], fit=True)
    pipeline['y_test'] = encoder.encode_target(pipeline['y_test'], fit=False)
    return jsonify({'status': 'ok'})
