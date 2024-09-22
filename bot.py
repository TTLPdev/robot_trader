import json
import time
import websocket
import bot_functions as bf
import config as cfg
from playsound import playsound


#Connect to the binance api and produce a client
client = bf.init_client()

#Load settings from settings.json
settings = cfg.getBotSettings()
market = settings.market
ticker_socket = market.lower()
leverage = int(settings.leverage)
margin_type = settings.margin_type
max_loss_per_trade = float(settings.max_loss_per_trade)

#turn off print unless we really need to print something
std = bf.getStdOut()
bf.blockPrint()
bf.singlePrint("Bot Starting", std)

#global values used by bot to keep track of state
entry_price = 0
in_position = False
side = 0
price_was_in_kumo = False
ohlcv_df_initialised = False

#Initialise the market leverage and margin type.
bf.initialise_futures(client, _market=market, _leverage=leverage)

#print some initialisation data
balance = bf.get_futures_balance(client)
bf.singlePrint(f"your futures balance: {balance}\n", std)
bf.singlePrint(f"contract that bot will trade: {market}", std)
bf.singlePrint(f"max loss per trade: {max_loss_per_trade}", std)

def on_message(ws, message):
    global in_position
    global price_was_in_kumo
    global client
    global std
    global ohlcv_df_initialised
    global market
    global leverage
    global side
    global entry_price

    json_message = json.loads(message)
    candle = json_message['k']
    candle_is_closed = bool(candle['x'])
    open_w = float(candle['o'])
    high_w = float(candle['h'])
    low_w = float(candle['l'])
    close_w = float(candle['c'])
    volume_w = float(candle['v'])


    if candle_is_closed:
        bf.singlePrint(f"open :{open_w}\nhigh: {high_w}\nlow: {low_w}\nclose: {close_w}\nvolume: {volume_w}\nprice was in kumo: {price_was_in_kumo}", std)

        try:
            #if not currently in a position then execute this set of logic
            if in_position == False:
                if ohlcv_df_initialised:
                    data_get_signal = bf.get_signal(candle, client, market)
                    entry = data_get_signal[0]
                    bf.singlePrint(f"trading_signal: {data_get_signal}\n", std)
                    
                    if price_was_in_kumo:
                        if entry == 'false_signal':
                            bf.singlePrint("\n\nFalse singal\n\n", std)
                            price_was_in_kumo = False
                        
                        if entry == 1:
                            bf.singlePrint("\n\nLONG\n\n", std)
                            data_trade = bf.handle_signal(client, std, max_loss_per_trade, market, leverage, "BUY")
                            bf.singlePrint(f'data_trade: {data_trade}', std)
                            in_position = data_trade[-1]
                            side = 1
                            playsound("notif.mp3", block=False)

                        if entry == -1:
                            bf.singlePrint("\n\nSHORT\n\n", std)
                            data_trade = bf.handle_signal(client, std, max_loss_per_trade, market, leverage, "SELL")
                            bf.singlePrint(f'data_trade: {data_trade}', std)
                            in_position = data_trade[-1]
                            side = -1
                            playsound("notif.mp3", block=False)
                    else:
                        if entry == 'in':
                            price_was_in_kumo = True
                else:
                    time.sleep(2)
                    ohlcv_df_initialised = bf.initialise_ohlcv(candle, client, market, "5m")
                    if ohlcv_df_initialised:
                        bf.singlePrint(f"ohlcv of bot_functions\n{bf.OHLCV}\nOHLCV initialised\n\n\n", std)

            
            #If already in a position wait for the trade to close
            elif in_position == True:
                bf.singlePrint("\nalready in position! waiting for sl or tp to be touched", std)

                position_active = bf.check_in_position(client, market)
                bf.singlePrint(f"position active: {position_active}", std)
                if position_active == False:
                    in_position == False
                    price_was_in_kumo = False
                    client.cancel_all_orders(market)
   
        except Exception as e:

            bf.singlePrint(f"Encountered Exception {e}", std)
            time.sleep(100)

def on_open(ws):
    bf.singlePrint("\nwebsocket opened", std)

def on_close(ws):
    bf.singlePrint("websocket closed", std)

def on_error(ws):
    bf.singlePrint("websocket error", std)

# we use websocket to know when candles close
socket = f'wss://fstream.binance.com/ws/{ticker_socket}@kline_1m'
ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close, on_error=on_error)
ws.run_forever()
