from flask import Flask, request, jsonify
from agents.stablebaselines3.models import DRLAgent
from reports.overview import overview_stock
from chart.chart import get_chart_day
import json
import numpy as np
from services import get_data_for_train, predictData
from datetime import datetime

app = Flask(__name__)
drl_agent = DRLAgent
trained_ppo = drl_agent.load_model("ppo", "trained_models/trained_ppo.zip")

app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


@app.route("/overview", methods=["GET"])
def overview():
    symbol = request.args.get('symbol')
    data = overview_stock(symbol)
    data = json.dumps(data, cls=NpEncoder)

    return data

@app.route("/chart", methods=["GET"])
def chart():
    symbol = request.args.get("symbol")
    data = get_chart_day(symbol)
    return data

@app.route("/train-ai", methods=["POST"])
def chose_method():
    model= request.json.get("model")
    file_data = request.json.get("filename")   
    from worker import celery

    celery.send_task('worker.process', kwargs={'filename': file_data, 
                                                'model': model})
    return 'Task added to the queue!'
    
@app.route("/crawl-data", methods=["POST"])
def crawldata():
    assets = request.json["assets"]
    start_date = request.json["start_date"]
    sdate = datetime.strptime(start_date, "%m/%d/%Y")
    formatted_sdate = sdate.strftime("%Y-%m-%d")

    end_date = request.json["end_date"]
    edate = datetime.strptime(end_date, "%m/%d/%Y")
    formatted_edate = edate.strftime("%Y-%m-%d")

    print(formatted_edate)
    filename = get_data_for_train(assets, formatted_sdate, formatted_edate)
    return filename
 
@app.route("/predict-data", methods=["POST"])
def checkdata():
    # model_trained = request.json['model_trained']
    model_type = request.json['model_type']
    end_trader = request.json["end_trader"]
    start_trader = request.json["start_trader"]
    assets = request.json['assets']
    filedata = request.json['filedata']
    data = predictData(filedata, assets, end_trader, start_trader, method=model_type)
    return data

@app.route("/training", methods=["POST"])
def trainning():
    pass
# def convert_data(df):
#     data_lookback = df.loc[i-lookback:i,:]
#     price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
#     return_lookback = price_lookback.pct_change().dropna()
#     return_list.append(return_lookback)

#     covs = return_lookback.cov().values 

if __name__ == '__main__':
    app.run(debug=True)