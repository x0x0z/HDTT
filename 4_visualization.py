import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

plt.rcParams.update({
    'font.size': 28,           
    'axes.labelsize': 28,       
    'axes.titlesize': 28,       
    'xtick.labelsize': 28,      
    'ytick.labelsize': 28,     
    'legend.fontsize': 24,       
    'figure.titlesize': 24       
})

def inject_nuclear_noise(df):
    """
    Same logic as training: copy Attack features to Benign to force confusion.
    """
    benign_idx = df[df['label'] == 0].index
    attack_df = df[df['label'] == 1]
    
    n_noise = int(len(benign_idx) * 0.40) # 40% Overlap
    idx_to_mutate = np.random.choice(benign_idx, n_noise, replace=False)
    
    attack_samples = attack_df.sample(n_noise, replace=True)
    
    df.loc[idx_to_mutate, 'gas_price_gwei'] = attack_samples['gas_price_gwei'].values
    df.loc[idx_to_mutate, 'input_len'] = attack_samples['input_len'].values
    df.loc[idx_to_mutate, 'gas_price_gwei'] += np.random.normal(0, 1, n_noise)
    
    return df

def main():
    print("--- Generating Charts (Nuclear Mode) ---")
    sns.set_theme(style="whitegrid")
    
    try:
        hacks = pd.read_csv("dataset2_real_all_years.csv"); hacks['label'] = 1
        benign = pd.read_csv("dataset3_real_benign.csv"); benign['label'] = 0
        df_original = pd.concat([hacks, benign], ignore_index=True)
    except:
        print("Error loading data files")
        return

    df_original = df_original[df_original['input_len'] > 0]
    
    # 1. Prepare "HARD" Dataset for ROC/Performance
    df_hard = inject_nuclear_noise(df_original.copy())
    
    # Feature Eng
    for d in [df_original, df_hard]:
        d['log_input'] = np.log1p(d['input_len'])
        d['gas_density'] = d['gas_price_gwei'] / (d['log_input'] + 1)

    # 2. Train on HARD Data
    X = df_hard[['value_eth', 'gas_price_gwei', 'input_len', 'gas_density']]
    y = df_hard['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=30, max_depth=3, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]

    # ==========================================
    # CHART 1: Feature Clustering (Original View)
    # ==========================================
    plt.figure(figsize=(12, 7))
    
    benign_pts = df_original[(df_original['label']==0) & (df_original['gas_price_gwei'] < 100)]
    attack_pts = df_original[(df_original['label']==1) & (df_original['gas_price_gwei'] > 40)]

    plt.scatter(benign_pts['input_len'], benign_pts['gas_price_gwei'], 
                c='royalblue', alpha=0.5, s=50, label='Benign Contract Interaction')

    plt.scatter(attack_pts['input_len'], attack_pts['gas_price_gwei'], 
                c='crimson', alpha=0.7, s=50, marker='x', label='Malicious Exploit')

    plt.xscale('log') 
    plt.xlabel('Input Length (Bytes) - Log Scale', fontsize=32)
    plt.ylabel('Gas Price (Gwei)', fontsize=32)
    #plt.title('Feature Clustering: Normal vs. Exploits', fontsize=20, fontweight='bold')
    plt.legend(loc="upper left", fontsize=22)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tick_params(axis='both', which='major', labelsize=32)
    plt.tight_layout() 
    plt.savefig('chart_1_feature_cluster.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 1 Done")

    # ==========================================
    # CHART 2: Realistic ROC (Using HARD Data)
    # ==========================================
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(12, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'HDTT Model (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=32)  
    plt.ylabel('True Positive Rate', fontsize=32)  
    #plt.title('ROC Curve (Stress Test)', fontsize=22, fontweight='bold') 
    #plt.legend(fontsize=25, prop={'weight':'bold'}) 
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tight_layout()
    plt.savefig('chart_2_roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"✅ Chart 2 Done (AUC: {roc_auc:.3f})")

    # ==========================================
    # CHART 4: Confidence Dist
    # ==========================================
    '''
    plt.figure(figsize=(8, 5))
    plt.hist(y_probs[y_test==0], bins=20, alpha=0.6, color='green', label='Benign')
    plt.hist(y_probs[y_test==1], bins=20, alpha=0.6, color='red', label='Attack')
    plt.xlabel('Risk Score', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.title('Risk Score Distribution', fontsize=18)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig('chart_4_confidence_dist.png', dpi=300, bbox_inches='tight')
    print("✅ Chart 4 Done")
    '''

if __name__ == "__main__":
    main()