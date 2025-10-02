import os
import shutil
import random
from collections import defaultdict

def split_dataset(source_dir="../dataset", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train/val/test splits
    """
    # Convert to absolute path to avoid path issues
    source_dir = os.path.abspath(source_dir)
    
    # Verify ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Ratios must sum to 1.0")
    
    random.seed(seed)  # For reproducible splits
    
    print("🔄 Splitting dataset...")
    print(f"Split ratios: {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test")
    
    # Create new directory structure
    splits = ['train', 'val', 'test']
    classes = ['closed', 'open']
    
    # Collect all images from current train and val BEFORE creating backup
    all_images = defaultdict(list)
    
    for current_split in ['train', 'val']:
        for class_name in classes:
            path = os.path.join(source_dir, current_split, class_name)
            if os.path.exists(path):
                images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for img in images:
                    # Store just the filename, we'll copy from backup
                    all_images[class_name].append(img)
                    
    print(f"\nFound images:")
    for class_name in classes:
        print(f"  {class_name}: {len(all_images[class_name])} images")
    
    # Create backup of current structure
    backup_path = f"{source_dir}_backup"
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    shutil.copytree(source_dir, backup_path)
    print(f"✅ Backup created at {backup_path}")
    
    # Remove existing split directories to avoid duplicates
    for split in splits:
        split_path = os.path.join(source_dir, split)
        if os.path.exists(split_path):
            shutil.rmtree(split_path)
    
    # Create new directory structure
    for split in splits:
        for class_name in classes:
            os.makedirs(os.path.join(source_dir, split, class_name), exist_ok=True)
    
    # Split each class independently
    stats = defaultdict(lambda: defaultdict(int))
    
    for class_name in classes:
        images = all_images[class_name].copy()  # Copy the list
        random.shuffle(images)  # Randomize order
        
        total = len(images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        test_count = total - train_count - val_count  # Remaining go to test
        
        print(f"\n{class_name.upper()} split:")
        print(f"  Train: {train_count}")
        print(f"  Val: {val_count}")
        print(f"  Test: {test_count}")
        
        # Move images to new splits
        splits_data = [
            ('train', images[:train_count]),
            ('val', images[train_count:train_count + val_count]),
            ('test', images[train_count + val_count:])
        ]
        
        for split_name, split_images in splits_data:
            dest_dir = os.path.join(source_dir, split_name, class_name)
            
            for img_filename in split_images:
                # Find the source file in backup (try both train and val)
                src_path = None
                for backup_split in ['train', 'val']:
                    potential_src = os.path.join(backup_path, backup_split, class_name, img_filename)
                    if os.path.exists(potential_src):
                        src_path = potential_src
                        break
                
                if src_path:
                    dest_path = os.path.join(dest_dir, img_filename)
                    
                    # Handle filename conflicts by adding index
                    counter = 1
                    while os.path.exists(dest_path):
                        name, ext = os.path.splitext(img_filename)
                        dest_path = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                        counter += 1
                    
                    shutil.copy2(src_path, dest_path)
                    stats[split_name][class_name] += 1
                else:
                    print(f"⚠️ Could not find source for {img_filename}")
    
    # Clean up old structure (remove old train/val from backup source)
    # (We keep the new structure)
    
    print(f"\n✅ Dataset split completed!")
    print(f"\nFinal distribution:")
    total_images = 0
    for split in splits:
        split_total = sum(stats[split].values())
        total_images += split_total
        print(f"{split.upper()}: {split_total} images")
        for class_name in classes:
            print(f"  {class_name}: {stats[split][class_name]}")
    
    print(f"\nTotal: {total_images} images")
    return stats

def verify_split(source_dir="../dataset"):
    """Verify the split was successful"""
    # Convert to absolute path
    source_dir = os.path.abspath(source_dir)
    
    print("\n🔍 Verifying split...")
    
    splits = ['train', 'val', 'test']
    classes = ['closed', 'open']
    
    for split in splits:
        print(f"\n{split.upper()}:")
        for class_name in classes:
            path = os.path.join(source_dir, split, class_name)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {class_name}: {count} images")

if __name__ == "__main__":
    # Run the split
    stats = split_dataset()
    
    # Verify results
    verify_split()
    
    print(f"\n🎯 Next steps:")
    print("1. Run: python analyze_dataset.py")
    print("2. Update train_gate.py to use new splits")
    print("3. Train your model: python train_gate.py")
    print("4. Test final model on test set")