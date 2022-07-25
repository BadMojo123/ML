EURUSDMIN = 1.00600
EURUSDMAX = 1.5
EURUSDVOLMIN = 0
EURUSDVOLMAX = 500
EURUSDMAXSIZE_1M = 0.0003

AdvisorDNN_M1_Name = "klony/EURUSD1M/Mieszko1530527759.020745.h5"
AdvisorDNN_M1_ClassifierName = "klony/EURUSD1M/MieszkoGraph1530527759.020745.h5"
AdvisorDNN_M5_Name = "klony/EURUSD5M/Mieszko1530527617.9131126.h5"
AdvisorDNN_M5_ClassifierName = "klony/EURUSD5M/MieszkoGraph1530527617.9131126.h5"
AdvisorDNN_M15_Name = "klony/EURUSD15M/Mieszko1530527820.415477.h5"
AdvisorDNN_M15_ClassifierName = "klony/EURUSD15M/MieszkoGraph1530527820.415477.h5"
reqHist = 65
# Commands for Zawisza
# eurusd_buy_order = "TRADE|OPEN|0|EURUSD|0|50|50|Python-to-MT4|price"
eurusd_buy_order = "TRADE|OPEN|0|EURUSD|0|0|0|Python-to-MT4"  # |price
eurusd_sell_order = "TRADE|OPEN|1|EURUSD|0|0|0|Python-to-MT4"
eurusd_close_orders = "TRADE|CLOSE|0|EURUSD|0|50|50|Python-to-MT4"
get_rates = "RATES|EURUSD"
BuyOrder = 'BUY'
SellOrder = 'SELL'
CloseOrder = 'CLOSE'
# DATA|SYMBOL|TIMEFRAME|start_pos|data count to copy
get_hist_1m = "DATA|EURUSD|1|0|" + str(reqHist)
get_hist_5m = "DATA|EURUSD|5|0|" + str(reqHist)
get_hist_15m = "DATA|EURUSD|15|0|" + str(reqHist)
get_hist_30m = "DATA|EURUSD|30|0|" + str(reqHist)
get_hist_60m = "DATA|EURUSD|60|0|" + str(reqHist)
