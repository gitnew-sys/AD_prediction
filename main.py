import os
import trimesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules.features import extract_all_descriptors
from modules.classifier import fisher_score, HierarchicalClassifier

# input preprocessed data
def load_real_data(data_dir):
    results = []
    if not os.path.exists(data_dir): return []
    subjects = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for sub in subjects:
        sub_path = os.path.join(data_dir, sub)
        h_p, a_p = os.path.join(sub_path, "hippo.stl"), os.path.join(sub_path, "amygdala.stl")
        if os.path.exists(h_p) and os.path.exists(a_p):
            print(f"Handle subjects: {sub}")
            m_h, m_a = trimesh.load(h_p), trimesh.load(a_p)
            f_h, f_a = extract_all_descriptors(m_h), extract_all_descriptors(m_a)
            
            # 自动识别标签: 文件夹名含 AD 为 2, 含 MCI 为 1, NC为 0
            label = 0
            if "AD" in sub.upper(): label = 2
            elif "MCI" in sub.upper(): label = 1
            
            results.append({
                'Subject': sub,
                'Label': label,
                'Feature_ShapeOp': np.concatenate([f_h['shape_operator'], f_a['shape_operator']])
            })
    return results

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    real_data_dir = os.path.join(base_dir, "data", "real_samples")
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    print("AD hierarchical based classification")
    print("====================================================")
    
    real_data = load_real_data(real_data_dir)
    dataset = real_data if real_data else generate_mock_data()
    df = pd.DataFrame(dataset)
    
    # 第一阶段: NC (0) vs Abnormal (1,2) ---
    print("\n[Phase 1] NC vs. Abnormal ...")
    X_all = np.stack(df['Feature_ShapeOp'].values)
    y_stage1 = (df['Label'] > 0).astype(int)
    
    scores1 = fisher_score(X_all, y_stage1)
    top_idx1 = np.argsort(scores1)[-30:] 
    X_s1 = X_all[:, top_idx1]
    
    clf1 = HierarchicalClassifier()
    res1 = clf1.evaluate(X_s1, y_stage1, "Stage 1")
    
    # 第二阶段: AD (2) vs MCI (1) ---
    print("[Phase 2] AD vs. MCI ...")
    df_abnormal = df[df['Label'] > 0]
    
    if len(df_abnormal) < 5:
        print("Warning：Not enough abnormal samples to proceed to the second stage of classification.")
        res2 = {"acc": 0, "auc": 0, "sens": 0, "spec": 0}
    else:
        X_ab = np.stack(df_abnormal['Feature_ShapeOp'].values)
        y_stage2 = (df_abnormal['Label'] == 2).astype(int) # AD 为 1, MCI 为 0
        
        scores2 = fisher_score(X_ab, y_stage2)
        top_idx2 = np.argsort(scores2)[-30:]
        X_s2 = X_ab[:, top_idx2]
        
        clf2 = HierarchicalClassifier()
        res2 = clf2.evaluate(X_s2, y_stage2, "Stage 2")

    # --- Result + plot ---
    print("\n" + "="*30)
    print(f"Phase 1 (NC vs Abnormal) Accuracy: {res1['acc']:.2%}")
    print(f"Phase 2 (AD vs MCI) Accuracy: {res2['acc']:.2%}")
    print("="*30)

    labels = ["Accuracy", "Sensitivity", "Specificity", "AUC"]
    s1_vals = [res1['acc'], res1['sens'], res1['spec'], res1['auc']]
    s2_vals = [res2['acc'], res2['sens'], res2['spec'], res2['auc']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, s1_vals, width, label='Phase 1 (NC vs Abn)', color='teal')
    ax.bar(x + width/2, s2_vals, width, label='Phase 2 (AD vs MCI)', color='orange')
    
    ax.set_ylabel('Score')
    ax.set_title('Hierarchical Classification Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    save_path = os.path.join(results_dir, "hierarchical_performance.png")
    plt.savefig(save_path)
    plt.close()
    #print(f"\n完整性能对比图已保存至: {save_path}")

if __name__ == "__main__":
    main()
