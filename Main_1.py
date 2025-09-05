import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta,timezone
import os,time,pickle

mode = "ofi"
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

def backtest(stock,day,strategy_params,data_path,frequency,start_time,end_time,lookup_duration,df):
    total_cost = 0
    total_size = 0


    # date = day.replace("-", "")
    order_log = []
    result_log = []
    id_counter = 0
    
    # local_path = f"{data_path}{stock}/xnas-itch-{date}.mbp-10.csv"

    if df is None:
        logging.warning(f"No data for {stock} on {day}")
        return None    
    # df = pd.read_csv(local_path)
    
    df['ts_event'] = pd.to_datetime(df['ts_event'],utc=True)
    start = day + " " + start_time[0] + ":" + start_time[1]
    end = day + " " + end_time[0] + ":" + end_time[1]
    last_order_time = datetime.strptime(end, '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
    end_extention = last_order_time + timedelta(minutes=5)
    clock = datetime.strptime(start, '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
    last_order_time = last_order_time - timedelta(minutes=strategy_params["T"])
    filtered_df = df[(df['ts_event'] >= start) & (df['ts_event'] <= end_extention)]
    #df_trade = filtered_df.compute()
    df_trade = preprocess_data(filtered_df)
    dfs = {
        "v1" : df_trade
    }
    ##############################
    pkl_path = f"ratio_table_{stock}.pkl"
    
    while clock <= last_order_time:
        target_time = clock + pd.Timedelta(minutes=strategy_params["T"])
        lookup_start = clock - timedelta(hours=lookup_duration[0], minutes=lookup_duration[1])
        
        
        #############################
        if os.path.exists(pkl_path):
            with open(pkl_path,"rb") as f:
                master_lookup = pickle.load(f)
        else:
            master_lookup = {}
        
        
                
        filtered_df = df[(df['ts_event'] >= lookup_start) & (df['ts_event'] < clock)]#
        #df_lookup = filtered_df.compute()
        df_lookup = preprocess_data(filtered_df)
        dfs_lookup = {
            "v1": df_lookup
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

        with open(pkl_path, "rb") as f:
            fresh_lookup = pickle.load(f)

        # Patch fresh_lookup to wrap entries with venue key if not already
        patched_lookup = {}
        for k, v in fresh_lookup.items():
            if isinstance(v, dict) and "avg_ratio" in v and "thresholds" in v:
                patched_lookup[k] = v  # 
            else:
                print(f"[WARNING] Skipping malformed key: {k}")


        # Merge into master
        master_lookup.update(patched_lookup)

        # Save patched version
        with open(pkl_path, "wb") as f:
            pickle.dump(master_lookup, f)


        
        
        
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
        if result != None:
            X_star = result[0]
            market_v,limit_v = round_to_target(X_star[0],X_star[1],strategy_params["S"])
            target_time = pd.Timestamp(target_time)
            nearest_ts = find_nearest_timestamp(dfs["v1"], target_time)
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
                "queue" : queue_size["v1"],
                "fill_time": [],
                "filled" : False,
                "Time Horizon": strategy_params["T"]
            }
            total_size += strategy_params["S"]
            total_cost += (market_p * market_v)
            mid_price = (df_trade["best_ask"][df_trade["timestamp"] == nearest_ts].values[0] + df_trade["best_bid"][df_trade["timestamp"] == nearest_ts].values[0])/2
            log["mid_price"].append((float(mid_price),int(market_v)))
            id_counter += 1
            order_log.append(log)
        else:
            print("Invaild Lookup Table")
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
                total_cost += order_log[0]["limit_fill"] * order_log[0]["limit_p"]
                order_log.pop(0)
            if (len(order_log) != 0):
                if (time - order_log[0]["time"]).total_seconds() > (strategy_params["T"]*60):
                    nearest_ts = find_nearest_timestamp(dfs["v1"], target_time)
                    market_p = df_trade["ask_px_00"][df_trade["timestamp"] == nearest_ts].values[0]
                    order_log[0]["left_market_v"] += order_log[0]["limit_v"] - order_log[0]["limit_fill"]
                    order_log[0]["left_market_p"] = market_p
                    mid_price = (df_trade["ask_px_00"][df_trade["timestamp"] == nearest_ts].values[0] + df_trade["bid_px_00"][df_trade["timestamp"] == nearest_ts].values[0]) /2
                    order_log[0]["mid_price"].append((float(mid_price),int(order_log[0]["left_market_v"])))
                    result_log.append(order_log[0])
                    total_cost += order_log[0]["limit_fill"] * order_log[0]["limit_p"]
                    total_cost += market_p * order_log[0]["left_market_v"]
                    order_log.pop(0)
    result = pd.DataFrame.from_dict(result_log)
    result.to_csv( data_path + stock + "/" + day +"_result.csv")
    return total_cost/total_size



##############--------------MAIN-----------###########

# backtest_type("ofi") # ofi or look_up

# strategy_params = { #define the parameters
#         'S': 100,
#         'T': 5,
#         'f': 0.003,
#         'r': [0.003],
#         'theta': 0.0005,
#         'lambda_u': 0.05,
#         'lambda_o': 0.05,
#         'N': 1000
#     }

# order_freq = 120 # unit sec
# start_time = ("15","45") #At least from 9:45 each day (hour,minute) string
# end_time = ("16","00") #End before 16:00 (hour,minute) string
# stocks = ["AAPL"]

# data_path = f"s3://blockhouse-databento-mbp10/lookup-table/"
# lookup_duration = (0,15) #(hour,minute) int
# days = ["2025-04-01"]
# average_cost = {}
# for stock in stocks:
#     average_cost[stock] = backtest(stock,days,strategy_params,data_path,order_freq,start_time,end_time,lookup_duration)