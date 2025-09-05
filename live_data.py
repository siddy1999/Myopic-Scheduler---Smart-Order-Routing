from __future__ import annotations

import json
import time
from typing import Dict, Any, Optional

import pandas as pd
from kafka import KafkaConsumer, TopicPartition

SYMBOLS = ["AAPL", "MSFT", "AMZN", "NVDA", "META"]
_PARTITION_MAPPING: Dict[str, int] = {
    "AAPL": 0,
    "MSFT": 1,
    "AMZN": 2,
    "NVDA": 3,
    "META": 4,
}
_TOPIC = "market_data"

# Row flattener (extract level 0 bid/ask)
def flatten_row(record: Dict[str, Any]) -> pd.Series:
    levels = record.get("levels", [])
    lvl0 = levels[0] if isinstance(levels, list) and levels else {}
    bid = lvl0.get("bid", {})
    ask = lvl0.get("ask", {})
    return pd.Series({
        "sequence": record.get("sequence"),
        "timestamp": pd.to_datetime(record.get("timestamp"), unit="ns", utc=True),
        "action": record.get("action"),
        "side": record.get("side"),
        "bid_price": bid.get("price"),
        "bid_size": bid.get("size"),
        "ask_price": ask.get("price"),
        "ask_size": ask.get("size"),
        "price": record.get("price"),
        "size": record.get("size"),
    })

# One-shot fetch helper

def fetch_live_market_data(
    *,
    n_ticks: Optional[int] = None,
    buffer_time_sec: int = 0,
    flatten: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Fetch live data either by count (last n_ticks) or time buffer.
    - n_ticks: number of most recent ticks per symbol.
    - buffer_time_sec: fallback for time-based window if n_ticks is None.
    - flatten: apply flatten_row to each record.
    """
    # Standalone consumer
    consumer = KafkaConsumer(
        bootstrap_servers=["kafka:9092"],
        enable_auto_commit=False,
        group_id=None,
        auto_offset_reset="earliest",
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        key_deserializer=lambda b: b.decode("utf-8"),
    )

    result: Dict[str, pd.DataFrame] = {}
    now_ms = int(time.time() * 1000)

    for symbol, partition in _PARTITION_MAPPING.items():
        tp = TopicPartition(_TOPIC, partition)
        consumer.assign([tp])

        if n_ticks is not None:
            # Count-based lookback
            beginning = consumer.beginning_offsets([tp])[tp]
            end = consumer.end_offsets([tp])[tp]
            start = max(beginning, end - n_ticks)
            consumer.seek(tp, start)
        else:
            # Time-based lookback
            lookback_ms = int(buffer_time_sec * 1000)
            timestamps = {tp: now_ms - lookback_ms}
            offsets = consumer.offsets_for_times(timestamps)
            off = offsets.get(tp)
            if off and off.offset is not None:
                consumer.seek(tp, off.offset)
            else:
                consumer.seek_to_beginning(tp)

        # Poll once
        raw = consumer.poll(timeout_ms=200).get(tp, [])
        rows = []
        for msg in raw:
            rec = msg.value
            ts = pd.to_datetime(rec["timestamp"], unit="ns", utc=True)
            rec["timestamp"] = ts
            rec["minute"] = ts.floor("min")
            if flatten:
                rows.append(flatten_row(rec))
            else:
                rows.append(rec)

        # Build DataFrame
        if rows:
            df = pd.DataFrame(rows)
        else:
            # Empty DataFrame with correct columns
            if flatten:
                df = pd.DataFrame(columns=flatten_row({}).index)
            else:
                df = pd.DataFrame(columns=[
                    "symbol","timestamp","human_readable_time",
                    "sequence","action","side","price","size","levels","minute"
                ])
        result[symbol] = df
        consumer.unsubscribe()

    consumer.close()
    return result