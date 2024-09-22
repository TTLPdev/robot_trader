from pandas_ta import volume
from binance_f import RequestClient
from binance_f.constant.test import *
from binance_f.base.printobject import *
from binance_f.model.constant import *
import pandas as pd
import numpy as np
import time
import sys, os
import config as cfg
import pandas_ta as pta
import decimal

# global var (it will be after initialisation a dataframe) for some candlestick data
OHLCV = 0

def getStdOut():
    return sys.stdout

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint(std):
    sys.stdout = std

def singlePrint(string, std):
    enablePrint(std)
    print(string)
    blockPrint()

#create a binance request client
def init_client():
    client = RequestClient(api_key=cfg.getPublicKey(), secret_key=cfg.getPrivateKey(), url=cfg.getBotSettings().api_url)
    return client

#Get futures balances. We are interested in USDT by default as this is what we use as margin.
def get_futures_balance(client, _asset = "USDT"):
    balances = client.get_balance()
    asset_balance = 0
    for balance in balances:
        if balance.asset == _asset:
            asset_balance = balance.balance
            break

    return asset_balance

#Init the market we want to trade. First we change leverage type
#then we change margin type
def initialise_futures(client, _market="BTCUSDT", _leverage=1, _margin_type="CROSSED"):
    try:
        client.change_initial_leverage(_market, _leverage)
    except Exception as e:
        print(e)

    try:
        client.change_margin_type(_market, _margin_type)
    except Exception as e:
        print(e)

#get all of our open orders in a market
def get_orders(client, _market="BTCUSDT"):
    orders = client.get_open_orders(_market)
    return orders, len(orders)

#get all of our open trades
def get_positions(client):
    positions = client.get_position_v2()
    return positions

#get trades we opened in the market the bot is trading in
def get_specific_positon(client, _market="BTCUSDT"):
    positions = get_positions(client)

    for position in positions:
        if position.symbol == _market:
            break

    return position

#close opened position
def close_position(client, _market="BTCUSDT"):
    position = get_specific_positon(client, _market)
    qty = float(position.positionAmt)

    _side = "BUY"
    if qty > 0.0:
        _side = "SELL"

    if qty < 0.0:
        qty = qty * -1

    qty = str(qty)

    execute_order(client, _market=_market,
                  _qty = qty,
                  _side = _side)

#get the liquidation price of the position we are in. - We don't use this - be careful!
def get_liquidation(client, _market="BTCUSDT"):
    position = get_specific_positon(client, _market)
    price = position.liquidationPrice
    return price

#Get the entry price of the position the bot is in
def get_entry(client, _market="BTCUSDT"):
    position = get_specific_positon(client, _market)
    price = position.entryPrice
    return price

#Execute an order, this can open and close a trade
def execute_order(client, _market="BTCUSDT", _type = "MARKET", _side="BUY", _position_side="BOTH", _qty=1):
    _qty = str(_qty)
    client.post_order(symbol=_market,
                      ordertype=_type,
                      side=_side,
                      positionSide=_position_side,
                      quantity = _qty)

#calculate how big a position we can open with the amount we can loss on a trade
def calculate_position_size(entry, stop_loss, max_loss, client, market):
    size = max_loss/abs(entry-stop_loss)
    precision = get_market_precision(client, _market=market)
    qty = round_to_precision(size, precision)

    return qty

def calculate_tp_1_size(qty_position, client, market):
    tp_1_size = qty_position/2
    precision = get_market_precision(client, _market=market)
    tp_1_size = round_to_precision(tp_1_size, precision)

    return tp_1_size

#check if the position is still active, or if the trailing stop was hit.
def check_in_position(client, _market="BTCUSDT"):
    position = get_specific_positon(client, _market)

    in_position = False

    if float(position.positionAmt) != 0.0:
        in_position = True

    return in_position


#Create a stop order to close our order with losses
def submit_sl_order(client, _stopPrice, _market="BTCUSDT", _type="STOP_MARKET", _side="BUY"):
    _stopPrice = str(_stopPrice)
    client.post_order(symbol=_market,
                      side=_side,
                      ordertype=_type,
                      stopPrice=_stopPrice,
                      workingType="MARK_PRICE",
                      closePosition=True)

#Create a take profit order to close our order with profits
def submit_final_tp_order(client, _stopPrice, _market="BTCUSDT", _type="TAKE_PROFIT_MARKET", _side="BUY"):
    _stopPrice = str(_stopPrice)
    client.post_order(symbol=_market,
                      side=_side,
                      ordertype=_type,
                      stopPrice=_stopPrice,
                      workingType="CONTRACT_PRICE",
                      closePosition=True)

def submit_tp_1_order(client, _price, qty, _market="BTCUSDT", _type="LIMIT", _side="BUY"):
    _price = str(_price)
    qty = str(qty)
    client.post_order(symbol=_market,
                      side=_side,
                      ordertype=_type,
                      timeInForce="GTC",
                      quantity=qty,
                      reduceOnly=True,
                      price=_price,)

# get the current market price
def get_market_price(client, _market="BTCUSDT"):
    price = client.get_symbol_price_ticker(_market)
    price = price[0].price
    return price

# get the precision of the market, this is needed to avoid errors when creating orders
def get_market_precision(client, _market="BTCUSDT"):
    market_data = client.get_exchange_information()
    precision = 3
    for market in market_data.symbols:
        if market.symbol == _market:
            precision = market.quantityPrecision
            break
        
    return precision

# round the position size we can open to the precision of the market
def round_to_precision(_qty, _precision):
    new_qty = "{:0.0{}f}".format(_qty , _precision)
    return float(new_qty)

# convert from client candle data into a set of lists
def convert_candles(candles):
    o = []
    h = []
    l = []
    c = []
    v = []

    for candle in candles:
        o.append(float(candle.open))
        h.append(float(candle.high))
        l.append(float(candle.low))
        c.append(float(candle.close))
        v.append(float(candle.volume))

    return o, h, l, c, v

#convert list candle data into list of heikin ashi candles
def construct_heikin_ashi(o, h, l, c):
    h_o = []
    h_h = []
    h_l = []
    h_c = []

    for i, v in enumerate(o):

        close_price = (o[i] + h[i] + l[i] + c[i]) / 4

        if i == 0:
            open_price = close_price
        else:
            open_price = (h_o[-1] + h_c[-1]) / 2

        high_price = max([h[i], close_price, open_price])
        low_price = min([l[i], close_price, open_price])

        h_o.append(open_price)
        h_h.append(high_price)
        h_l.append(low_price)
        h_c.append(close_price)

    return h_o, h_h, h_l, h_c

def get_supertrend(ohlcv):
    supertrend = pta.supertrend(ohlcv["high"], ohlcv["low"], ohlcv["close"], length=10)
    return supertrend

def get_ichimoku(ohlcv):
    ichimoku = pta.ichimoku(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    return ichimoku

def get_macd(ohlcv):
    macd = pta.macd(ohlcv["close"])
    return macd

def get_kc(ohlcv, length=20, scalar=1.5):
    kc = pta.kc(ohlcv["high"], ohlcv["low"], ohlcv["close"], length, scalar)
    return kc

def get_ema(ohlcv, length):
    ema = pta.ema(ohlcv["close"], length)
    return ema

def get_rsi(ohlcv, scalar=50):
    rsi = pta.rsi(ohlcv["close"], scalar)
    return rsi

def get_atr(ohlcv, length=14):
    atr = pta.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], length)
    return atr

def get_tp(entry, stop_loss, rr):
    # entry: entry price
    # stop_loss: stop loss
    # rr: risk ration (eg: 1 (one time the risk))
    stop_level = (entry - stop_loss)*rr
    tp = entry + stop_level

    return tp

def get_price_precision(_close_price):
    close_price = str(_close_price)
    d = decimal.Decimal(close_price)
    precision = abs(d.as_tuple().exponent)
    
    return precision

def handle_signal(client, std, max_loss, market="BTCUSDT", leverage=3, order_side="BUY"):
    global OHLCV
    initialise_futures(client, _market=market, _leverage=leverage)
    infos_debug = [order_side] #debugging infos

    # SL calculation
    ichimoku = get_ichimoku(OHLCV)
    ichimoku_b = ichimoku[1]
    ssa = float(ichimoku_b.loc[499, 'ISA_9'])
    ssb = float(ichimoku_b.loc[499, 'ISB_26'])
    list_indic_sl = [ssa, ssb]
    close_price = float(OHLCV.loc[498, 'close'])


    # SL calculation
    if order_side == "BUY":
        # find which is lower in this list [ssa, ssb]
        sl = min(list_indic_sl)
        sl_tp_side = "SELL"
        infos_debug.append(sl)

    if order_side == "SELL":
        # find which is higher in this list [ssa, ssb]
        sl = max(list_indic_sl)
        sl_tp_side = "BUY"
        infos_debug.append(sl)
        
    price_precision = get_price_precision(close_price)
    infos_debug.append(price_precision)
    sl = round_to_precision(sl, price_precision)
    infos_debug.append(sl)
    #verify if sl is not too close enty price
    if (abs(sl-close_price)/close_price)*100 >= 0.12:
        # TP calculation
        tp = get_tp(close_price, sl, 1.5)
        tp = round_to_precision(tp, price_precision)
        infos_debug.append(tp)
        # position size calculation
        qty = calculate_position_size(close_price, sl, max_loss, client, market)
        infos_debug.append(qty)
        enablePrint(std)
        execute_order(client, _market=market, _side=order_side, _qty=qty)
        submit_sl_order(client, sl, _market=market, _side=sl_tp_side)
        submit_final_tp_order(client, tp, _market=market, _side=sl_tp_side)
        blockPrint()

        side = -1
        if order_side == "BUY":
            side = 1
        else:
            side = -1

        in_position = True

        singlePrint(f"{order_side}: {qty} ${close_price} using x{leverage} leverage", std)
        log_trade(_qty=qty, _market=market, _leverage=leverage, _side=side,
        _cause="Signal", _market_price=close_price, _type=order_side)

        infos_debug.append(in_position)
        return infos_debug
    else:
        in_position=False
        infos_debug.append(in_position)
        return infos_debug


#create a dataframe for our candles
def to_dataframe(o, h, l, c, v):
    df = pd.DataFrame()

    df['open'] = o
    df['high'] = h
    df['low'] = l
    df['close'] = c
    df['volume'] = v

    return df

"""
#Exponential moving avg - unused
def ema(s, n):
    s = np.array(s)
    out = []
    j = 1

    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    out.append(sma)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    out.append(( (s[n] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ( (i - out[j]) * multiplier) + out[j]
        j = j + 1
        out.append(tmp)

    return np.array(out)
"""

#Avarage true range function used by our trading strat
def avarage_true_range(high, low, close):

    atr = []

    for i, v in enumerate(high):
        if i!= 0:
            value = np.max([high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1])])
            atr.append(value)
    return np.array(atr)

# Trading strategy: Long: MEGA COMPLIQUÉ ATT UN PEU FREROT
#                 : Short: inverse of long
#signals are -1 for short, 1 for long, 0 for do nothing, 'in' to know that price is in kumo, false_signal' for an ichimoku signal not confirmed
def trading_signal(client, market):
    global OHLCV

    kc = get_kc(OHLCV, 20, 1.5)
    ema_ = get_ema(OHLCV, 200)
    ema_ = get_ema(OHLCV, 200)
    ema_ = get_ema(OHLCV, 200)
    rsi = get_rsi(OHLCV, 50)
    atr = get_atr(OHLCV, 14)

    
    close_price = float(OHLCV.loc[498, 'close'])
    #last_ema = float(ema[498])

    entry = 0

    # Long
    if truc:
        entry = 1
    # Short
    if truc:
        entry = -1

    return [entry]

#get the data from the market, create heikin ashi candles and then generate signals
#return the signals to the bot
def get_signal(candle_w, client, market):
    add_candle_to_ohlcv_df(candle_w)
    entry = trading_signal(client, market)
    return entry

#get signal that is confirmed across multiple time scales
def get_multi_scale_signal(client, _market="BTCUSDT", _periods=["5m", "15m"]):

    signals = np.zeros(499)
    use_last = True

    for i, v in enumerate(_periods):
  
        _signal = get_signal(client, _market, _period= v, use_last=use_last)
        signals = signals + np.array(_signal)

    signals = signals / len(_periods)

    trade_signal = []

    for i, v in enumerate(list(signals)):

        if v == -1:
            trade_signal.append(-1)
        elif v == 1:
            trade_signal.append(1)
        else:
            trade_signal.append(0)

    return trade_signal

#calculate a rounded position size for the bot, based on current USDT holding, leverage and market
def calculate_position(client, _market="BTCUSDT", _leverage=1):
    usdt = get_futures_balance(client, _asset = "USDT")
    qty = calculate_position_size(client, usdt_balance=usdt, _market=_market, _leverage=_leverage)
    precision = get_market_precision(client, _market=_market)
    qty = round_to_precision(qty, precision)
    return qty

#function for logging trades to csv for later analysis
def log_trade(_qty=0, _market="BTCUSDT", _leverage=1, _side="long", _cause="signal", _market_price=0, _type="exit"):
    df = pd.read_csv("trade_log.csv")
    df2 = pd.DataFrame()
    df2['time'] = [time.time()]
    df2['market'] = [_market]
    df2['qty'] = [_qty]
    df2['leverage'] = [_leverage]
    df2['cause'] = [_cause]
    df2['side'] = [_side]
    df2['market_price'] = [_market_price]
    df2['type'] = [_type]

    df = df.append(df2, ignore_index=True)
    df.to_csv("trade_log.csv", index=False)

def initialise_ohlcv(candle_w, client, market, period):
    global OHLCV

    # some websocket data
    close_w = float(candle_w['c'])
    volume_w = float(candle_w['v'])

    # get_candlestick_data data
    candles = client.get_candlestick_data(market, interval=period)
    o, h, l, c, v = convert_candles(candles)
    del o[-1]
    del h[-1]
    del l[-1]
    del c[-1]
    del v[-1]

    ohlcv_df = to_dataframe(o, h, l, c, v)
    close_price_df = float(ohlcv_df.loc[498, 'close'])
    volume_df = float(ohlcv_df.loc[498, 'volume'])

    if close_w==close_price_df and volume_w==volume_df:
        OHLCV = ohlcv_df
        return True
    else:
        return False

def add_candle_to_ohlcv_df(candle_w):
    global OHLCV

    # data of candle
    open_w = float(candle_w['o'])
    high_w = float(candle_w['h'])
    low_w = float(candle_w['l'])
    close_w = float(candle_w['c'])
    volume_w = float(candle_w['v'])

    # deleting the first row of df
    OHLCV.drop(index=0, axis=0, inplace=True)

    # adding new row
    new_row = {'open':open_w, 'high':high_w, 'low': low_w, 'close':close_w, 'volume':volume_w}
    OHLCV = OHLCV.append(new_row, ignore_index=True)
