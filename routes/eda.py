from flask import Blueprint, jsonify
from src.eda import Eda
from routes.state import pipeline

eda_bp = Blueprint('eda', __name__, url_prefix='/eda')


@eda_bp.route('/info', methods=['GET'])
def info():
    dl = pipeline['loader']
    if dl is None:
        return jsonify({'error': 'Dataset non caricato. Chiamare prima POST /loader/load'}), 400

    eda = Eda(dl.df)
    result = eda.info()
    return jsonify({
        'n_righe': result['n_righe, n_colonne'][0],
        'n_colonne': result['n_righe, n_colonne'][1],
        'variabili': result['variabili'],
    })
