import databento as db
import json
from kafka import KafkaProducer
from datetime import datetime
import threading

# Define symbols that you want to subscribe to.
SYMBOLS = ["AAPL", "MSFT", "AMZN", "NVDA", "META"]

# Custom mapping for each symbol to a specific partition (assumes topic 'market_data' has 5 partitions).
PARTITION_MAPPING = {
    "AAPL": 0,
    "MSFT": 1,
    "AMZN": 2,
    "NVDA": 3,
    "META": 4
}

def custom_partitioner(key_bytes, all_partitions, available):
    """Custom partitioner that assigns a partition based on a pre-defined mapping.
    
    Arguments:
    - key_bytes: The message key as bytes.
    - all_partitions: All partition ids for the topic.
    - available: Currently available partition ids.
    
    Returns:
    - The partition id based on the mapping.
    """
    # Decode the key to string.
    key = key_bytes.decode('utf-8')
    # Look up the partition number for this key.
    partition_id = PARTITION_MAPPING.get(key)
    # Ensure the partition_id is in the available partitions; otherwise, fallback to the first available.
    if partition_id is None or partition_id not in available:
        partition_id = available[0]
    return partition_id

# Kafka Producer configuration with custom partitioner.
producer = KafkaProducer(
    bootstrap_servers=['localhost:29092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    key_serializer=lambda x: x.encode('utf-8'),
    partitioner=custom_partitioner
)

def format_mbp_message(msg, symbol):
    """Format MBP-1 message into a clean dictionary structure and add the symbol."""
    return {
        'symbol': symbol,  # Use the provided symbol
        'timestamp': msg.hd.ts_event,  # Nanosecond timestamp
        'human_readable_time': datetime.fromtimestamp(msg.hd.ts_event / 1e9).isoformat(),
        'sequence': msg.sequence,
        'action': msg.action,
        'side': msg.side,
        'price': float(msg.price),
        'size': msg.size,
        'levels': [{
            'bid': {
                'price': float(level.bid_px),
                'size': level.bid_sz,
                'count': level.bid_ct
            },
            'ask': {
                'price': float(level.ask_px),
                'size': level.ask_sz,
                'count': level.ask_ct
            }
        } for level in msg.levels]
    }

def create_process_message(symbol):
    """Create a callback function that attaches the given symbol to each message."""
    def process_message(msg):
        try:
            formatted_msg = format_mbp_message(msg, symbol)
            # Send to the single Kafka topic with symbol as key.
            producer.send(
                'market_data',
                key=symbol,  # Using the symbol as the key ensures the custom partitioner assigns the correct partition.
                value=formatted_msg
            )
            print(json.dumps(formatted_msg, indent=2))
        except Exception as e:
            print(f"Error processing message for {symbol}: {e}")
    return process_message

def run_client(client, symbol):
    """Run the Databento client and block until complete (or a timeout)."""
    try:
        client.start()
        client.block_for_close(timeout=950)  # Run for 5 minutes.
    except KeyboardInterrupt:
        print(f"\nGracefully shutting down client for {symbol}...")
    finally:
        pass

# Create separate Live client instances (and corresponding threads) per symbol.
client_threads = []
for symbol in SYMBOLS:
    # Create a new client for the symbol.
    client = db.Live("db-UJqP5caEMG3VW4XhXpgbMaaUHGm4U")
    # Subscribe only to this symbol.
    client.subscribe(
        dataset="EQUS.MINI",
        schema="mbp-1",
        symbols=[symbol]  # Single symbol subscription.
    )
    # Add the callback that knows the symbol.
    client.add_callback(create_process_message(symbol))
    # Run each client in a separate thread.
    t = threading.Thread(target=run_client, args=(client, symbol))
    client_threads.append(t)
    t.start()

# Wait for all client threads to finish.
for t in client_threads:
    t.join()

# Clean up the Kafka producer once all client threads have completed.
producer.close()
