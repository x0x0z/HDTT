import time
import pandas as pd
import numpy as np
from brownie import accounts, project, web3, history

def main():
    # 1. Setup
    admin = accounts[0]
    proj = project.get_loaded_projects()[0]
    
    trust_store = proj.HDTTTrustStore.deploy({'from': admin})
    demo = proj.HDTTDemoProtocol.deploy(trust_store.address, {'from': admin})
    attacker_contract = proj.Attacker.deploy(demo.address, {'from': admin})

    users = accounts[1:10] # 10 Normal users
    logs = []

    print(f"--- Phase 1: Generating Dataset 1 (Admin Balance: {admin.balance()/1e18:.2f} ETH) ---")

    # 2. Loop for Normal Traffic
    print("   Generating Normal Traffic...")
    for i in range(500):
        user = np.random.choice(users)
        amt = (np.random.pareto(3.0) + 1) * 1.5 
        amt = min(amt, 40.0)
        gas_price = np.random.uniform(20, 50) * 1e9 

        # We log BEFORE sending to ensure we capture the intent
        # For normal traffic, we label as 0
        logs.append({
            "value_eth": amt,
            "gas_price_gwei": gas_price / 1e9,
            "is_contract": 0,
            "input_len": 4, 
            "label": 0
        })

        try:
            demo.deposit({'from': user, 'value': web3.to_wei(amt, "ether"), 'gas_price': gas_price})
        except: 
            # Normal users might fail too (insufficient funds etc), but we still log the attempt
            pass

    # 3. Loop for Attack Traffic
    print("   Simulating Attacks...")
    
    # Attack 1: Sandwich (High Gas EOA)
    bad_actor = accounts[9]
    for i in range(50):
        amt = 80 
        gas_price = np.random.uniform(100, 500) * 1e9
        
        # Record the attempt
        logs.append({
            "value_eth": amt,
            "gas_price_gwei": gas_price / 1e9,
            "is_contract": 0,
            "input_len": 68,
            "label": 1
        })

        try:
            demo.deposit({'from': bad_actor, 'value': web3.to_wei(amt, "ether")})
            demo.withdraw(web3.to_wei(amt, "ether"), {'from': bad_actor, 'gas_price': gas_price})
        except: pass

    # Attack 2: Atomic Contract Attack
    for i in range(20): 
        amt = 40 
        gas_price = np.random.uniform(200, 1000) * 1e9
        
        # Record the attempt (THIS IS THE FIX)
        logs.append({
            "value_eth": amt,
            "gas_price_gwei": gas_price / 1e9,
            "is_contract": 1, 
            "input_len": 4,
            "label": 1
        })

        try:
            tx = attacker_contract.attack({'from': admin, 'value': web3.to_wei(amt, "ether"), 'gas_price': gas_price})
            print(f"   Attack {i} executed (Success)")
        except Exception as e:
            # It is GOOD that it fails. It means your contract blocked it.
            # We just print a short message, but the data is already logged above.
            print(f"   Attack {i} blocked by protocol (Expected)")

    # Save
    df = pd.DataFrame(logs)
    df.to_csv("dataset1_simulated.csv", index=False)
    print(f"✅ Saved {len(df)} rows to dataset1_simulated.csv")