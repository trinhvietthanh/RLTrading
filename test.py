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

a= 0
for i in FUEVFVND_PER:
  a += FUEVFVND_PER[i]

print(a)