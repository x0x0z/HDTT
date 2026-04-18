import requests
import pandas as pd
import random
import time
from datetime import datetime

# --- CONFIGURATION ---
# We rotate these to prevent "Node Busy" errors
NODES = [
    #"https://rpc.ankr.com/eth",
    "https://eth.llamarpc.com",
    #"https://cloudflare-eth.com"
]

TARGET_TX_COUNT = 2000  # Set to your desired target (2000-5000)
OUTPUT_FILE = "dataset3_real_benign.csv"
START_BLOCK = 16000000  # Approx Jan 2023
END_BLOCK = 21000000    # Approx Oct 2024

def get_block(block_num):
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getBlockByNumber",
        "params": [hex(block_num), True],
        "id": 1
    }
    
    # Try nodes in order until one works
    for node in NODES:
        try:
            r = requests.post(node, json=payload, timeout=3)
            if r.status_code == 200:
                data = r.json()
                if "result" in data and data["result"]:
                    return data["result"]
        except:
            continue
    return None

def parse(block):
    txs = []
    if not block or 'transactions' not in block: return []
    
    for tx in block['transactions']:
        try:
            val = int(tx['value'], 16) / 1e18
            
            gas_hex = tx.get('gasPrice', tx.get('maxFeePerGas', '0x0'))
            gas = int(gas_hex, 16) / 1e9
            
            inp = (len(tx.get('input', '0x')) - 2) // 2
            is_contract = 1 if (inp > 0 or tx.get('to') is None) else 0
            
            txs.append({
                "value_eth": val,
                "gas_price_gwei": gas,
                "is_contract": is_contract,
                "input_len": max(0, inp),
                "label": 0  # Benign
            })
        except: continue
    return txs

def main():
    print(f"--- Fast Scraper: Target {TARGET_TX_COUNT} Benign Txs ---")
    data = []
    
    while len(data) < TARGET_TX_COUNT:
        # Grab batches of 20 blocks
        blocks = random.sample(range(START_BLOCK, END_BLOCK), 20)
        
        for b in blocks:
            b_data = get_block(b)
            if b_data:
                new_txs = parse(b_data)
                # Take max 50 per block to ensure diversity
                random.shuffle(new_txs)
                data.extend(new_txs[:50])
                
                print(f"Collected: {len(data)} / {TARGET_TX_COUNT}", end="\r")
            
            if len(data) >= TARGET_TX_COUNT: break
            
    # Save
    df = pd.DataFrame(data[:TARGET_TX_COUNT])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ SUCCESS! Saved {len(df)} transactions to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()