from flask import Blueprint, jsonify
from src.encoder import Encoder
from routes.state import pipeline, pipeline_lock

encoder_bp = Blueprint('encoder', __name__, url_prefix='/encoder')


@encoder_bp.route('/fit_transform_features', methods=['POST'])
def fit_transform_features():
    with pipeline_lock:
        X_train = pipeline['X_train']
    if X_train is None:
        return jsonify({'error': 'Dati non disponibili. Completare prima il pipeline fino a /cleaner'}), 400

    cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()
    encoder = Encoder()
    X_train_encoded = encoder.fit_transform_features(X_train, cat_cols)  # computazione fuori dal lock
    with pipeline_lock:
        pipeline['X_train'] = X_train_encoded
        pipeline['encoder'] = encoder
    return jsonify({'status': 'ok', 'shape': list(X_train_encoded.shape)})


@encoder_bp.route('/transform_features', methods=['POST'])
def transform_features():
    with pipeline_lock:
        encoder = pipeline['encoder']
        X_test = pipeline['X_test']
    if encoder is None or X_test is None:
        return jsonify({'error': 'Chiamare prima POST /encoder/fit_transform_features'}), 400

    X_test_encoded = encoder.transform_features(X_test)  # computazione fuori dal lock
    with pipeline_lock:
        pipeline['X_test'] = X_test_encoded
    return jsonify({'status': 'ok', 'shape': list(X_test_encoded.shape)})


@encoder_bp.route('/encode_target', methods=['POST'])
def encode_target():
    with pipeline_lock:
        encoder = pipeline['encoder']
        y_train = pipeline['y_train']
        y_test = pipeline['y_test']
    if encoder is None:
        return jsonify({'error': 'Chiamare prima POST /encoder/fit_transform_features'}), 400

    y_train_enc = encoder.encode_target(y_train, fit=True)
    y_test_enc = encoder.encode_target(y_test, fit=False)
    with pipeline_lock:
        pipeline['y_train'] = y_train_enc
        pipeline['y_test'] = y_test_enc
    return jsonify({'status': 'ok'})
