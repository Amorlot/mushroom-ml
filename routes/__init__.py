from routes.loader import loader_bp
from routes.split import split_bp
from routes.cleaner import cleaner_bp
from routes.encoder import encoder_bp
from routes.eda import eda_bp
from routes.logreg import logreg_bp
from routes.xgboost import xgboost_bp


def register_blueprints(app):
    app.register_blueprint(loader_bp)
    app.register_blueprint(split_bp)
    app.register_blueprint(cleaner_bp)
    app.register_blueprint(encoder_bp)
    app.register_blueprint(eda_bp)
    app.register_blueprint(logreg_bp)
    app.register_blueprint(xgboost_bp)
