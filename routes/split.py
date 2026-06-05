from flask import Blueprint, jsonify
from src.split import Split
from routes.state import pipeline

split_bp = Blueprint('split', __name__, url_prefix='/split')


@split_bp.route('/split', methods=['POST'])
def split():
    dl = pipeline['loader']
    if dl is None:
        return jsonify({'error': 'Dataset non caricato. Chiamare prima POST /loader/load'}), 400

    X = dl.df.drop(columns=[dl.target_col])
    y = dl.df[dl.target_col]

    splitter = Split()
    X_train, X_test, y_train, y_test = splitter.split(X, y)

    pipeline['X_train'] = X_train
    pipeline['X_test'] = X_test
    pipeline['y_train'] = y_train
    pipeline['y_test'] = y_test

    return jsonify({
        'status': 'ok',
        'train_rows': len(X_train),
        'test_rows': len(X_test),
    })
