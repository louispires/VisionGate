import os
from collections import defaultdict

def analyze_dataset():
    dataset_stats = defaultdict(int)
    
    print("📊 Dataset Analysis")
    print("=" * 50)
    
    for split in ['train', 'val']:
        print(f"\n{split.upper()} SET:")
        for class_name in ['closed', 'open']:
            path = f"dataset/{split}/{class_name}"  # Go up one level
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                dataset_stats[f"{split}_{class_name}"] = count
                print(f"  {class_name}: {count:,} images")
            else:
                print(f"  {class_name}: ❌ Directory not found")
    
    total = sum(dataset_stats.values())
    print(f"\n🎯 SUMMARY:")
    print(f"Total images: {total:,}")
    
    # Calculate balance
    train_closed = dataset_stats.get('train_closed', 0)
    train_open = dataset_stats.get('train_open', 0)
    val_closed = dataset_stats.get('val_closed', 0)
    val_open = dataset_stats.get('val_open', 0)
    
    if train_closed > 0 and train_open > 0:
        train_balance = min(train_closed, train_open) / max(train_closed, train_open)
        print(f"Train class balance: {train_balance:.3f} (closer to 1.0 = better)")
    
    if val_closed > 0 and val_open > 0:
        val_balance = min(val_closed, val_open) / max(val_closed, val_open)
        print(f"Val class balance: {val_balance:.3f} (closer to 1.0 = better)")
    
    # Training recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    if total < 2000:
        print("⚠️  Dataset is small - aim for 2,000+ images for good results")
    elif total < 5000:
        print("✅ Good dataset size - can achieve 93-95% accuracy")
    else:
        print("🚀 Excellent dataset size - can achieve 95-97% accuracy")
    
    if train_balance < 0.8:
        print("⚠️  Classes are imbalanced - collect more of the minority class")
    else:
        print("✅ Classes are well balanced")
    
    # Split ratio analysis
    train_total = train_closed + train_open
    val_total = val_closed + val_open
    if train_total > 0 and val_total > 0:
        split_ratio = val_total / (train_total + val_total)
        print(f"Train/Val split: {train_total}/{val_total} ({split_ratio:.1%} validation)")
        if split_ratio < 0.15 or split_ratio > 0.25:
            print("⚠️  Consider 15-20% validation split for optimal results")
    
    return dataset_stats

if __name__ == "__main__":
    stats = analyze_dataset()