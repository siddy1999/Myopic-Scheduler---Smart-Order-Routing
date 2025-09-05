import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta,timezone
import dask.dataframe as dd

mode = "look_up"
if mode == "look_up":
    from gen_lookup_table import preprocess_data,find_nearest_timestamp,calculate_queue_depth,execute_optimization,generate_lookup_table
elif mode == "ofi":
    from gen_lookup_table_ofi import preprocess_data,find_nearest_timestamp,calculate_queue_depth,execute_optimization,generate_lookup_table

def round_to_target(a, b, target_sum):
    a_floor = int(np.floor(a))
    b_floor = int(np.floor(b))
    floor_sum = a_floor + b_floor
    to_add = target_sum - floor_sum
    fracs = [(a - a_floor, 'a'), (b - b_floor, 'b')]
    fracs.sort(reverse=True)
    a_rounded = a_floor
    b_rounded = b_floor
    for i in range(to_add):
        if fracs[i][1] == 'a':
            a_rounded += 1
        else:
            b_rounded += 1
    return a_rounded, b_rounded

def depth_loc(row,price):
    OB = row.to_dict()
    for i in OB.keys():
        if ("px" in i) and OB[i] == price:
            depth = i[-1:]
            return int(depth)

def price_loc(row,depth):
    OB = row.to_dict()
    for i in OB.keys():
        if i == ("bid_px_0" + str(depth)):
            return OB[i]
    print(OB.keys())
    print("bid_px_0" + str(depth))

def backtest(stock,days,strategy_params,data_path,frequency,start_time,end_time,lookup_duration):

    for day in days:

        date = day.replace("-", "")
        order_log = []
        result_log = []
        id_counter = 0
        path = data_path + stock + "/" + "xnas-itch-" + date + ".mbp-10.parquet"
        df = dd.read_parquet(path)
        df['ts_event'] = dd.to_datetime(df['ts_event'],utc=True)
        start = day + " " + start_time[0] + ":" + start_time[1]
        end = day + " " + end_time[0] + ":" + end_time[1]
        clock = datetime.strptime(start, '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
        last_order_time = datetime.strptime(end, '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
        last_order_time = last_order_time - timedelta(minutes=strategy_params["T"])
        filtered_df = df[(df['ts_event'] >= start) & (df['ts_event'] <= end)]
        df_trade = filtered_df.compute()
        df_trade = preprocess_data(df_trade)
        dfs = {
            "NDAQ" : df_trade
        }

        while clock <= last_order_time:
            target_time = clock + pd.Timedelta(minutes=strategy_params["T"])
            lookup_start = clock - timedelta(hours=lookup_duration[0], minutes=lookup_duration[1])

            filtered_df = df[(df['ts_event'] >= lookup_start) & (df['ts_event'] < clock)]
            df_lookup = filtered_df.compute()
            df_lookup = preprocess_data(df_lookup)
            dfs_lookup = {
                "NDAQ": df_lookup
            }

            generate_lookup_table(
                dfs_lookup,
                pd.Timestamp(target_time),
                S=strategy_params["S"],
                T=strategy_params["T"],
                f=strategy_params["f"],
                r=strategy_params["r"],
                lambda_u=strategy_params["lambda_u"],
                lambda_o=strategy_params["lambda_o"],
                N=strategy_params["N"],
                method="exp",
                stock=stock)

            result = execute_optimization(
                dfs,
                pd.Timestamp(target_time),
                S=strategy_params["S"],
                T=strategy_params["T"],
                f=strategy_params["f"],
                r=strategy_params["r"],
                lambda_u=strategy_params["lambda_u"],
                lambda_o=strategy_params["lambda_o"],
                N=strategy_params["N"],
                method="lookup",
                stock=stock)
            X_star = result[0]
            market_v,limit_v = round_to_target(X_star[0],X_star[1],strategy_params["S"])
            target_time = pd.Timestamp(target_time)
            nearest_ts = find_nearest_timestamp(dfs["NDAQ"], target_time)
            market_p = df_trade["best_ask"][df_trade["timestamp"] == nearest_ts].values[0]
            limit_p = df_trade["best_bid"][df_trade["timestamp"] == nearest_ts].values[0]
            queue_size = calculate_queue_depth(dfs,target_time)
            log = {
                "id": id_counter,
                "time" : clock,
                "market_v" : market_v,
                "market_p" : market_p,
                "limit_v" : limit_v,
                "limit_p" : limit_p,
                "left_market_v": 0,
                "left_market_p": 0,
                "limit_fill" : 0,
                "mid_price": [],
                "queue" : queue_size["NDAQ"],
                "fill_time": [],
                "filled" : False
            }
            mid_price = (df_trade["best_ask"][df_trade["timestamp"] == nearest_ts].values[0] + df_trade["best_bid"][df_trade["timestamp"] == nearest_ts].values[0])/2
            log["mid_price"].append((float(mid_price),int(market_v)))
            id_counter += 1
            order_log.append(log)
            clock = clock + pd.Timedelta(seconds=frequency)


        df_trade.reset_index(inplace=True)
        df_trade = df_trade.rename(columns={'best_bid': 'bid_px_00', 'best_ask': 'ask_px_00'})
        for row in tqdm(df_trade.iterrows(), total=len(df_trade)):
            current_row = row[1]
            if len(order_log) == 0:
                break
            time = pd.Timestamp(current_row["ts_event"])
            if order_log[0]["time"] < time:
                depth = depth_loc(current_row,order_log[0]["limit_p"])
                price = price_loc(current_row,current_row["depth"])
                if  (current_row["depth"] == depth or price >= order_log[0]["limit_p"]) and (current_row["side"] == "B") and (current_row["action"] == "T"):
                    order_log[0]["queue"] -= current_row["size"]
                    left_fill = order_log[0]["limit_v"] - order_log[0]["limit_fill"]
                    fill_amount = min(current_row["size"],left_fill)
                else:
                    fill_amount = 0
                if (current_row["depth"] == depth) and (current_row["side"] == "B") and (current_row["action"] == "C"):
                    if order_log[0]["queue"] >= current_row["size"]:
                        order_log[0]["queue"] -= current_row["size"]
                if order_log[0]["queue"] < 0:
                    order_log[0]["limit_fill"]  = -1 * order_log[0]["queue"]
                    if fill_amount != 0:
                        order_log[0]["fill_time"].append(str(current_row["ts_event"]))
                        mid_price = (current_row["ask_px_00"] + current_row["bid_px_00"])/2
                        order_log[0]["mid_price"].append((float(mid_price),int(fill_amount)))
                if (order_log[0]["limit_fill"] >= order_log[0]["limit_v"]):
                    order_log[0]["limit_fill"] = order_log[0]["limit_v"]
                    order_log[0]["filled"] = True
                    result_log.append(order_log[0])
                    order_log.pop(0)
                if (len(order_log) != 0):
                    if (time - order_log[0]["time"]).total_seconds() > (strategy_params["T"]*60):
                        nearest_ts = find_nearest_timestamp(dfs["NDAQ"], target_time)
                        market_p = df_trade["ask_px_00"][df_trade["timestamp"] == nearest_ts].values[0]
                        order_log[0]["left_market_v"] += order_log[0]["limit_v"] - order_log[0]["limit_fill"]
                        order_log[0]["left_market_p"] = market_p
                        mid_price = (df_trade["ask_px_00"][df_trade["timestamp"] == nearest_ts].values[0] + df_trade["bid_px_00"][df_trade["timestamp"] == nearest_ts].values[0]) /2
                        order_log[0]["mid_price"].append((float(mid_price),int(order_log[0]["left_market_v"])))
                        result_log.append(order_log[0])
                        order_log.pop(0)

        result = pd.DataFrame.from_dict(result_log)
        result.to_csv("Result/" + stock + "/" + day +"_RESULT.csv")

strategy_params = {
        'S': 100,
        'T': 5,
        'f': 0.003,
        'r': [0.002],
        'theta': 0.0005,
        'lambda_u': 0.05,
        'lambda_o': 0.05,
        'N': 1000
    }
order_freq = 120 # unit sec
start_time = ("15","45") #At least from 9:45 each day (hour,minute)
end_time = ("16","00") #End before 16:00 (hour,minute)
stock = "AAPL"
data_path = "Data/"
lookup_duration = (0,15) #(hour,minute)
# days = ["20250407","20250408","20250409","20250410","20250411"]
days = ["2025-04-11"]
backtest(stock,days,strategy_params,data_path,order_freq,start_time,end_time,lookup_duration)