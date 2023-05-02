from flask import Flask, request, jsonify
from agents.stablebaselines3.models import DRLAgent

app = Flask(__name__)
drl_agent = DRLAgent
trained_ppo = drl_agent.load_model("ppo", "trained_models/trained_ppo.zip")

@app.route("/predict", methods=["POST"])
def predict():
    model = request.json['model']
    data_trade = request.json['data_trade']
    
    #Predict
    action, _ = drl_agent.predict(model, data_trade)
    
    # Trả về kết quả dưới dạng JSON
    return jsonify({'action': action})

@app.route("/get-data-stock", methods=["GET"])
    

def convert_data(df):
    data_lookback = df.loc[i-lookback:i,:]
    price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
    return_lookback = price_lookback.pct_change().dropna()
    return_list.append(return_lookback)

    covs = return_lookback.cov().values 

if __name__ == '__main__':
    app.run()