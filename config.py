# directory
from __future__ import annotations

DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

TRAIN_START_DATE = "2016-01-06"
TRAIN_END_DATE = "2020-02-1"

TEST_START_DATE = "2020-08-01"
TEST_END_DATE = "2020-10-01"

TRADE_START_DATE = "2020-11-01"
TRADE_END_DATE = "2021-01-01"
INDICATORS = [
  "macd",
  "boll_ub",
  "boll_lb",
  "rsi_30",
  "cci_30",
  "dx_30",
  "close_30_sma",
  "close_60_sma"
]

# Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 64,  # bug fix:KeyError: 'eval_times' line 68, in get_model model.eval_times = model_kwargs["eval_times"]
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}

DOW_30_TICKER = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WBA",
    "WMT",
    "DIS",
    "DOW",
]

VN_30_TICKER = [
    "HPG",
    # "SSI",
    "VND",
    "STB",
    "VPB",
    "NVL",
    "POW",
    "MBB",
    "PDR",
    "TCH",
    "TPB",
    "VIC",
    "TCB",
    "SBT",
    "CTG",
    "PLX",
    "MWG",
    "VRE",
    "VHM",
    "HDB",
    "VNM",
    "FPT",
    "BID",
    "MSN",
    "VCB",
    "PNJ",
    "REE",
    "KDH",
    "VJC",
    "GAS",
    "BVH"
]


FUEVFVND_PER = {
    "ACB": 8.3,
    "CTG": 0.7,
    "DHC": 0.2,
    "EIB": 1.1,
    "FPT": 16,
    "GMD": 2.5,
    "KDH": 1.5,
    "MBB": 5.8,
    "MSB": 2.7,
    "MWG": 14.5,
    "NLG": 0.6,
    "OCB": 0.6,
    "PNJ": 15.6,
    "REE": 10.6,
    "TCB": 7.2,
    "TPB": 2.2,
    "VIB": 2.2,
    "TCM": 0.2,
    "VPB": 7.5,
}

FUEVFVND = [
    "ACB",
    "CTG",
    "DHC",
    "EIB",
    "FPT",
    "GMD",
    "KDH",
    "MBB",
    "MSB",
    "MWG",
    "NLG",
    "OCB",
    "PNJ",
    "REE",
    "TCB",
    "TPB",
    "VIB",
    "TCM",
    "VPB"
    ]
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.51 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.41 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_3_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.41 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.41 Safari/537.36"
]

# Config for DataLoader
URL_VND = 'https://www.vndirect.com.vn/portal/thong-ke-thi-truong-chung-khoan/lich-su-gia.shtml'
URL_CAFE = "http://s.cafef.vn/Lich-su-giao-dich-"
HEADERS = {'content-type': 'application/x-www-form-urlencoded', 'User-Agent': 'Mozilla'}

__all__ = [
    'USER_AGENTS', 'URL_VND', 'URL_CAFE', 'HEADERS'
]