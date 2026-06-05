from flask import Blueprint, request, jsonify
from src.data_loader import DataLoader
from routes.state import pipeline

loader_bp = Blueprint('loader', __name__, url_prefix='/loader')


@loader_bp.route('/load', methods=['POST'])
def load():
    body = request.get_json() or {}
    dl = DataLoader(
        dataset_id=body.get('dataset_id', 73),
        target_col=body.get('target_col', 'poisonous'),
        drop_cols=body.get('drop_cols', ['veil-type']),
        drop_missing_thresh=body.get('drop_missing_thresh', None),
    )
    dl.load()
    pipeline['loader'] = dl
    return jsonify({
        'status': 'ok',
        'rows': len(dl.df),
        'columns': dl.df.columns.tolist(),
    })


@loader_bp.route('/info', methods=['GET'])
def info():
    dl = pipeline['loader']
    if dl is None:
        return jsonify({'error': 'Dataset non caricato. Chiamare prima POST /loader/load'}), 400

    X = dl.df.drop(columns=[dl.target_col])
    y = dl.df[dl.target_col]
    return jsonify({
        'righe': len(dl.df),
        'colonne': len(dl.df.columns),
        'features': len(X.columns),
        'numeriche': X.select_dtypes(include='number').columns.tolist(),
        'categoriali': X.select_dtypes(exclude='number').columns.tolist(),
        'target': dl.target_col,
        'target_distribution': y.value_counts().to_dict(),
    })


@loader_bp.route('/missing', methods=['GET'])
def missing():
    dl = pipeline['loader']
    if dl is None:
        return jsonify({'error': 'Dataset non caricato. Chiamare prima POST /loader/load'}), 400

    total = len(dl.df)
    missing_counts = dl.df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    return jsonify({
        col: {'count': int(count), 'pct': round(count / total * 100, 2)}
        for col, count in missing_counts.items()
    })
