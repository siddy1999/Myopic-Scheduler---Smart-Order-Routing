import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
import dask.dataframe as dd
import logging

# Import existing modules
mode = "ofi"  # OFI mode use karenge for better results
if mode == "look_up":
    from gen_lookup_table import preprocess_data, find_nearest_timestamp, calculate_queue_depth, execute_optimization, generate_lookup_table
elif mode == "ofi":
    from gen_lookup_table_ofi import preprocess_data, find_nearest_timestamp, calculate_queue_depth, execute_optimization, generate_lookup_table

# Import myopic modules
from myopic_sor_scheduler import MyopicScheduler, MyopicParameters

def round_to_target(a, b, target_sum):
    """Existing function - no changes needed"""
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

def depth_loc(row, price):
    """Existing function - no changes needed"""
    OB = row.to_dict()
    for i in OB.keys():
        if ("px" in i) and OB[i] == price:
            depth = i[-1:]
            return int(depth)

def price_loc(row, depth):
    """Existing function - no changes needed"""
    OB = row.to_dict()
    for i in OB.keys():
        if i == ("bid_px_0" + str(depth)):
            return OB[i]
    print(OB.keys())
    print("bid_px_0" + str(depth))

def enhanced_preprocess_data(df):
    """Enhanced preprocessing with myopic requirements"""
    # Existing preprocessing
    df = preprocess_data(df)
    
    # Add myopic-specific columns
    if 'signed_volume' not in df.columns:
        df['signed_volume'] = df.get('bid_fill', 0) - df.get('ask_fill', 0)
    
    if 'mid_price' not in df.columns:
        df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2
    
    # Calculate volatility from price changes (replace yfinance dependency)
    if 'Volatility' not in df.columns:
        df['Volatility'] = df['mid_price'].pct_change().rolling(100).std().fillna(0.01)
    
    # Calculate ADV from trade data (replace yfinance dependency)
    if 'ADV' not in df.columns:
        daily_volume = df[df['action'] == 'T']['size'].sum() if 'action' in df.columns else 1000000
        df['ADV'] = daily_volume
    
    return df

def enhanced_backtest(stock, days, strategy_params, data_path, frequency, 
                     start_time, end_time, lookup_duration, use_myopic=True):
    """Enhanced backtest with myopic scheduling option"""
    
    # If myopic disabled, use original backtest
    if not use_myopic:
        return original_backtest(stock, days, strategy_params, data_path, 
                               frequency, start_time, end_time, lookup_duration)
    
    print(f"🚀 Starting MYOPIC ENHANCED backtest for {stock}")
    
    # Initialize myopic scheduler
    myopic_params = MyopicParameters(
        lambda_value=25000.0,  # Will be auto-estimated
        beta=0.693,           # 1-hour half-life
        volatility=0.01,      # Default, will be updated
        adv=1000000.0        # Default, will be updated
    )
    scheduler = MyopicScheduler(myopic_params)
    
    all_results = []
    
    for day in days:
        print(f"📅 Processing day: {day}")
        
        # Date and time setup (same as original)
        date = day.replace("-", "")
        order_log = []
        result_log = []
        id_counter = 0
        
        # Load data
        path = data_path + stock + "/" + "xnas-itch-" + date + ".mbp-10.parquet" 
        try:
            df = dd.read_parquet(path)
        except:
            # Fallback to CSV if parquet not available
            path = data_path + stock + "/" + "xnas-itch-" + date + ".mbp-10.csv"
            df = dd.read_csv(path)
        
        df['ts_event'] = dd.to_datetime(df['ts_event'], utc=True)
        
        # Time window setup
        start = day + " " + start_time[0] + ":" + start_time[1]
        end = day + " " + end_time[0] + ":" + end_time[1]
        clock = datetime.strptime(start, '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
        last_order_time = datetime.strptime(end, '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)
        last_order_time = last_order_time - timedelta(minutes=strategy_params["T"])
        
        # Filter and preprocess data
        filtered_df = df[(df['ts_event'] >= start) & (df['ts_event'] <= end)]
        df_trade = filtered_df.compute()
        df_trade = enhanced_preprocess_data(df_trade)  # Enhanced preprocessing
        
        print(f"📊 Data loaded: {len(df_trade)} rows")
        
        # MYOPIC ENHANCEMENT: Estimate lambda from data
        try:
            print("🧠 Estimating lambda parameter...")
            lambda_values = scheduler.estimate_lambda(df_trade)
            if lambda_values:
                best_lambda = lambda_values.get('60s', lambda_values.get('30s', 25000.0))
                scheduler.params.lambda_value = best_lambda
                print(f"✅ Lambda estimated: {best_lambda:.2f}")
            else:
                print("⚠️  Lambda estimation failed, using default")
        except Exception as e:
            print(f"⚠️  Lambda estimation error: {e}")
        
        # Update volatility and ADV from actual data
        if len(df_trade) > 0:
            scheduler.params.volatility = df_trade['Volatility'].mean()
            scheduler.params.adv = df_trade['ADV'].iloc[0]
            print(f"📈 Updated volatility: {scheduler.params.volatility:.6f}")
            print(f"📊 Updated ADV: {scheduler.params.adv:.0f}")
        
        dfs = {"NDAQ": df_trade}
        
        # MYOPIC ENHANCEMENT: Generate optimal trading schedule
        print("🎯 Generating myopic trading schedule...")
        session_length = (last_order_time - clock).total_seconds() / 60  # in minutes
        
        try:
            myopic_schedule = scheduler.generate_trading_schedule(
                df=df_trade,
                total_quantity=strategy_params["S"], 
                time_horizon=int(session_length)
            )
            print(f"📋 Generated {len(myopic_schedule)} myopic decisions")
        except Exception as e:
            print(f"❌ Myopic schedule generation failed: {e}")
            print("🔄 Falling back to traditional approach")
            return original_backtest(stock, [day], strategy_params, data_path, 
                                   frequency, start_time, end_time, lookup_duration)
        
        # MAIN TRADING LOOP - MYOPIC VERSION
        for decision in myopic_schedule:
            myopic_time = decision['timestamp']
            optimal_quantity = abs(decision['optimal_quantity'])
            
            # Skip very small trades
            if optimal_quantity < 1:
                continue
            
            # Convert to pandas timestamp if needed
            if not isinstance(myopic_time, pd.Timestamp):
                myopic_time = pd.Timestamp(myopic_time)
            
            # Check if within trading window
            if myopic_time > last_order_time:
                break
                
            print(f"⏰ Myopic decision at {myopic_time}: {optimal_quantity:.0f} shares")
            
            clock = myopic_time
            target_time = clock + pd.Timedelta(minutes=strategy_params["T"])
            lookup_start = clock - timedelta(hours=lookup_duration[0], 
                                           minutes=lookup_duration[1])
            
            # Generate lookup table (same as original)
            try:
                filtered_df_lookup = df[(df['ts_event'] >= lookup_start) & 
                                      (df['ts_event'] < clock)]
                df_lookup = filtered_df_lookup.compute()
                df_lookup = enhanced_preprocess_data(df_lookup)
                dfs_lookup = {"NDAQ": df_lookup}
                
                generate_lookup_table(
                    dfs_lookup,
                    pd.Timestamp(target_time),
                    S=optimal_quantity,  # ⭐ MYOPIC QUANTITY
                    T=strategy_params["T"],
                    f=strategy_params["f"],
                    r=strategy_params["r"],
                    lambda_u=strategy_params["lambda_u"],
                    lambda_o=strategy_params["lambda_o"],
                    N=strategy_params["N"],
                    method="exp",
                    stock=stock
                )
                
                # SOR optimization (same as original)
                result = execute_optimization(
                    dfs,
                    pd.Timestamp(target_time),
                    S=optimal_quantity,  # ⭐ MYOPIC QUANTITY
                    T=strategy_params["T"],
                    f=strategy_params["f"],
                    r=strategy_params["r"],
                    lambda_u=strategy_params["lambda_u"],
                    lambda_o=strategy_params["lambda_o"],
                    N=strategy_params["N"],
                    method="lookup",
                    stock=stock
                )
                
                if result is not None and result[0] is not None:
                    X_star = result[0]
                    market_v, limit_v = round_to_target(X_star[0], X_star[1], int(optimal_quantity))
                    
                    # Get current market prices (same as original)
                    nearest_ts = find_nearest_timestamp(dfs["NDAQ"], pd.Timestamp(target_time))
                    if nearest_ts is not None:
                        market_p = df_trade["best_ask"][df_trade["timestamp"] == nearest_ts].values[0]
                        limit_p = df_trade["best_bid"][df_trade["timestamp"] == nearest_ts].values[0]
                        queue_size = calculate_queue_depth(dfs, pd.Timestamp(target_time))
                        
                        # Create order log (enhanced with myopic info)
                        log = {
                            "id": id_counter,
                            "time": clock,
                            "market_v": market_v,
                            "market_p": market_p,
                            "limit_v": limit_v,
                            "limit_p": limit_p,
                            "left_market_v": 0,
                            "left_market_p": 0,
                            "limit_fill": 0,
                            "mid_price": [],
                            "queue": queue_size.get("NDAQ", 0),
                            "fill_time": [],
                            "filled": False,
                            "Time Horizon": strategy_params["T"],
                            # ⭐ MYOPIC SPECIFIC FIELDS
                            "myopic_quantity": optimal_quantity,
                            "myopic_alpha": decision.get('alpha', 0),
                            "myopic_impact": decision.get('price_impact', 0),
                            "lambda_used": scheduler.params.lambda_value
                        }
                        
                        mid_price = (market_p + limit_p) / 2
                        log["mid_price"].append((float(mid_price), int(market_v)))
                        
                        order_log.append(log)
                        id_counter += 1
                        
                        print(f"✅ Order placed: {market_v}M + {limit_v}L at {market_p:.2f}/{limit_p:.2f}")
                    
            except Exception as e:
                print(f"❌ Error in SOR execution: {e}")
                continue
        
        print(f"📝 Total orders placed: {len(order_log)}")
        
        # ORDER EXECUTION SIMULATION (Same as original)
        print("🎮 Starting order execution simulation...")
        df_trade.reset_index(inplace=True)
        df_trade = df_trade.rename(columns={'best_bid': 'bid_px_00', 'best_ask': 'ask_px_00'})
        
        for row in tqdm(df_trade.iterrows(), total=len(df_trade), desc="Processing ticks"):
            current_row = row[1]
            if len(order_log) == 0:
                break
                
            time = pd.Timestamp(current_row["ts_event"])
            
            if order_log[0]["time"] < time:
                # [Same fill logic as original - no changes needed]
                depth = depth_loc(current_row, order_log[0]["limit_p"])
                price = price_loc(current_row, current_row["depth"]) if 'depth' in current_row else None
                
                if ((current_row.get("depth") == depth or (price and price >= order_log[0]["limit_p"])) and 
                    (current_row.get("side") == "B") and 
                    (current_row.get("action") == "T")):
                    
                    order_log[0]["queue"] -= current_row.get("size", 0)
                    left_fill = order_log[0]["limit_v"] - order_log[0]["limit_fill"]
                    fill_amount = min(current_row.get("size", 0), left_fill)
                else:
                    fill_amount = 0
                
                if ((current_row.get("depth") == depth) and 
                    (current_row.get("side") == "B") and 
                    (current_row.get("action") == "C")):
                    if order_log[0]["queue"] >= current_row.get("size", 0):
                        order_log[0]["queue"] -= current_row.get("size", 0)
                
                if order_log[0]["queue"] < 0:
                    order_log[0]["limit_fill"] = -1 * order_log[0]["queue"]
                    if fill_amount != 0:
                        order_log[0]["fill_time"].append(str(current_row["ts_event"]))
                        mid_price = (current_row.get("ask_px_00", 0) + current_row.get("bid_px_00", 0)) / 2
                        order_log[0]["mid_price"].append((float(mid_price), int(fill_amount)))
                
                if order_log[0]["limit_fill"] >= order_log[0]["limit_v"]:
                    order_log[0]["limit_fill"] = order_log[0]["limit_v"]
                    order_log[0]["filled"] = True
                    result_log.append(order_log[0])
                    order_log.pop(0)
                
                if len(order_log) != 0:
                    if (time - order_log[0]["time"]).total_seconds() > (strategy_params["T"] * 60):
                        nearest_ts = find_nearest_timestamp(dfs["NDAQ"], pd.Timestamp(target_time))
                        if nearest_ts is not None:
                            market_p = df_trade["ask_px_00"][df_trade["timestamp"] == nearest_ts].values[0]
                            order_log[0]["left_market_v"] += order_log[0]["limit_v"] - order_log[0]["limit_fill"]
                            order_log[0]["left_market_p"] = market_p
                            mid_price = ((df_trade["ask_px_00"][df_trade["timestamp"] == nearest_ts].values[0] + 
                                        df_trade["bid_px_00"][df_trade["timestamp"] == nearest_ts].values[0]) / 2)
                            order_log[0]["mid_price"].append((float(mid_price), int(order_log[0]["left_market_v"])))
                            result_log.append(order_log[0])
                            order_log.pop(0)
        
        # Save results
        if result_log:
            result_df = pd.DataFrame.from_dict(result_log)
            output_path = f"Result/{stock}/{day}_MYOPIC_RESULT.csv"
            result_df.to_csv(output_path, index=False)
            print(f"💾 Results saved to {output_path}")
            
            all_results.extend(result_log)
        else:
            print("⚠️  No completed orders to save")
    
    print(f"🏁 Myopic backtest completed! Total results: {len(all_results)}")
    return all_results

def original_backtest(stock, days, strategy_params, data_path, frequency, 
                     start_time, end_time, lookup_duration):
    """Original backtest for fallback"""
    print("🔄 Running original backtest...")
    # [Copy the original backtest function from Main_1.py here]
    # This is your fallback when myopic fails
    pass  # Implement this by copying existing backtest function