from celery import Celery
from agents.stablebaselines3.models import DRLAgent
from app import app
from meta.preprocessor.preprocessors import data_split
from services import trainAI, load_data_predict
from meta.env_trade.portfolio.env_portfolio import StockPortfolioEnv
import pandas as pd
import config

def make_celery(flask_app):
    celery = Celery(
        'statistic_task',
        backend=flask_app.config['CELERY_RESULT_BACKEND'],
        broker=flask_app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(flask_app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with flask_app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


celery = make_celery(app)

@celery.task()
def process(filename, model):
    trainAI(filename=filename, method=model)
    return True

@celery.task()
def predict(env_trade, trained_ppo):
    predictData(env_trade, trained_ppo)
    return True

def predictData(model, env):
    # episode_total_assets = agent.DRL_prediction_load_from_file("ppo", e_trade_gym, "trained_models/trained_ppo.zip")
    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model,
                            environment = env)
    df_daily_return.to_csv('results/1.csv')
    df_actions.to_csv('results/2.csv')
