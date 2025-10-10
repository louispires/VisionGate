#!/usr/bin/env python3
"""
Super simple script to test deployed VisionGate model
Just modify the URL and FOLDER variables below and run!
"""

import os
import requests
from collections import Counter
import concurrent.futures
import time
from threading import Lock

# ========================================
# MODIFY THESE VARIABLES:
# ========================================
DEPLOYED_URL = "http://10.10.10.72:8000/classify"  # Your deployed model URL
SOURCE_FOLDER = r"C:\Working\source\VisionGate\dataset_backup\train"        # Base folder containing closed/open subfolders
MAX_IMAGES = 5000  # Test 5000 images for comprehensive testing
MAX_WORKERS = 20   # Number of concurrent threads (adjust based on server capacity)

# Global counters with thread safety
results_lock = Lock()
progress_lock = Lock()
open_files = []  # Track files classified as open
open_files_lock = Lock()

def process_single_image(args):
    """Process a single image - designed for concurrent execution"""
    image_file, image_path, index, total = args
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_file, f, 'image/jpeg')}
            response = requests.post(DEPLOYED_URL, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get('status', result.get('prediction', result.get('class', str(result).strip('"{}'))))
            confidence = result.get('confidence', 0.0)

            # Track files classified as open
            if prediction.lower() == 'closed':
                with open_files_lock:
                    open_files.append({
                        'file': image_file,
                        'confidence': confidence,
                        'full_path': image_path
                    })
            
            # Thread-safe progress reporting
            with progress_lock:
                if index % 100 == 0 or index == total:
                    print(f"📸 {index}/{total} processed... Latest: {prediction} ({confidence:.3f})")
            
            return {'status': 'success', 'prediction': prediction, 'confidence': confidence, 'file': image_file}
        else:
            return {'status': 'http_error', 'code': response.status_code, 'file': image_file}
            
    except requests.exceptions.Timeout:
        return {'status': 'timeout', 'file': image_file}
    except requests.exceptions.ConnectionError:
        return {'status': 'connection_error', 'file': image_file}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'file': image_file}

def test_model():
    """Test deployed model with concurrent processing for both closed and open folders"""
    
    print(f"🔍 Testing images from: {SOURCE_FOLDER}")
    print(f"🌐 Against URL: {DEPLOYED_URL}")
    print(f"⚡ Using {MAX_WORKERS} concurrent workers")
    print("=" * 50)
    
    # Look for closed and open subfolders
    subfolders = []
    for subfolder_name in ['closed', 'open']:
        subfolder_path = os.path.join(SOURCE_FOLDER, subfolder_name)
        if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
            subfolders.append((subfolder_name, subfolder_path))
    
    if not subfolders:
        print(f"❌ No 'closed' or 'open' subfolders found in {SOURCE_FOLDER}")
        return
    
    # Process each subfolder
    all_low_confidence = []
    all_misclassified = []
    
    for expected_class, folder_path in subfolders:
        print(f"\n📁 Processing {expected_class.upper()} images from: {folder_path}")
        print("-" * 50)
        
        # Reset global variables for each folder
        global open_files
        open_files = []
        
        # Get all image files
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit for testing
        image_files = image_files[:MAX_IMAGES]
        total_files = len(image_files)
        
        if not image_files:
            print(f"❌ No images found in {folder_path}")
            continue
        
        print(f"📊 Processing {total_files} images...")
        
        # Prepare arguments for concurrent processing
        args_list = [
            (image_file, os.path.join(folder_path, image_file), i+1, total_files) 
            for i, image_file in enumerate(image_files)
        ]
        
        # Track results
        results = Counter()
        errors = Counter()
        confidences = []
        all_results = []  # Store all individual results
        start_time = time.time()
        
        # Process images concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(process_single_image, args): args for args in args_list}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_args):
                result = future.result()
                
                if result['status'] == 'success':
                    results[result['prediction']] += 1
                    confidences.append(result['confidence'])
                    all_results.append({
                        'file': result['file'],
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'full_path': os.path.join(folder_path, result['file']),
                        'expected_class': expected_class
                    })
                else:
                    errors[result['status']] += 1
                    if len(errors) <= 5:  # Show first few errors
                        print(f"❌ {result['status']}: {result['file']}")
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        total_processed = sum(results.values())
        total_errors = sum(errors.values())
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        images_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0
        
        # Show results for this folder
        print(f"\n📊 RESULTS FOR {expected_class.upper()} FOLDER:")
        print("=" * 50)
        
        for prediction, count in results.most_common():
            percentage = (count / total_processed) * 100 if total_processed > 0 else 0
            print(f"{prediction.upper()}: {count:,} ({percentage:.1f}%)")
        
        if errors:
            print(f"\n❌ ERRORS:")
            for error_type, count in errors.items():
                print(f"  {error_type}: {count}")
        
        print(f"\n📈 STATISTICS:")
        print(f"✅ Successfully processed: {total_processed:,}/{total_files:,}")
        print(f"⚡ Processing speed: {images_per_second:.1f} images/second")
        print(f"🕐 Total time: {elapsed_time:.1f} seconds")
        print(f"🎯 Average confidence: {avg_confidence:.3f}")
        if confidences:
            print(f"📊 Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
        
        success_rate = (total_processed / total_files) * 100
        print(f"📋 Success rate: {success_rate:.1f}%")
        
        # Collect low confidence files from this folder
        low_confidence_files = [item for item in all_results if item['confidence'] < 1.00]
        all_low_confidence.extend(low_confidence_files)
        
        # Collect misclassified files
        misclassified_files = [item for item in all_results if item['prediction'].lower() != expected_class.lower()]
        all_misclassified.extend(misclassified_files)
        
        if low_confidence_files:
            print(f"\n🔍 FILES WITH CONFIDENCE < 1.00 IN {expected_class.upper()} FOLDER ({len(low_confidence_files)} files):")
            print("=" * 60)
            
            # Sort by confidence (lowest first)
            low_confidence_sorted = sorted(low_confidence_files, key=lambda x: x['confidence'])
            
            for i, item in enumerate(low_confidence_sorted, 1):
                print(f"{i:2d}. {item['file']} -> {item['prediction']} (confidence: {item['confidence']:.3f})")
        else:
            print(f"\n✅ All {expected_class} files have perfect 1.000 confidence!")
    
    # Show overall summary
    print(f"\n🎯 OVERALL SUMMARY:")
    print("=" * 60)
    
    if all_low_confidence:
        print(f"\n🔍 ALL FILES WITH CONFIDENCE < 1.00 ({len(all_low_confidence)} files total):")
        print("=" * 60)
        print("(These have lower confidence and should be reviewed)")
        print()
        
        # Sort by confidence (lowest first - most suspicious)
        all_low_confidence_sorted = sorted(all_low_confidence, key=lambda x: x['confidence'])

        for i, item in enumerate(all_low_confidence_sorted, 1):
            print(f"{i:2d}. {item['file']} ({item['expected_class']}) -> {item['prediction']} (confidence: {item['confidence']:.3f})")
        
        # Save to file for easy access
        with open('low_confidence_results.txt', 'w') as f:
            f.write(f"Files with confidence < 1.00 from {SOURCE_FOLDER}\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Files with confidence < 1.00: {len(all_low_confidence)}\n")
            f.write("=" * 60 + "\n\n")

            for i, item in enumerate(all_low_confidence_sorted, 1):
                f.write(f"{i:2d}. {item['file']} ({item['expected_class']}) -> {item['prediction']} (confidence: {item['confidence']:.3f})\n")
                f.write(f"    Full path: {item['full_path']}\n\n")

        print(f"\n💾 Detailed list saved to: low_confidence_results.txt")
        print(f"📂 You can inspect these images to verify classifications")
    else:
        print(f"\n✅ All files have perfect 1.000 confidence!")
    
    if all_misclassified:
        print(f"\n❌ MISCLASSIFIED FILES ({len(all_misclassified)} files total):")
        print("=" * 60)
        print("(These were classified incorrectly)")
        print()
        
        # Sort by confidence (lowest first - most suspicious)
        all_misclassified_sorted = sorted(all_misclassified, key=lambda x: x['confidence'])

        for i, item in enumerate(all_misclassified_sorted, 1):
            print(f"{i:2d}. {item['file']} (expected: {item['expected_class']}, got: {item['prediction']}, confidence: {item['confidence']:.3f})")
        
        # Save to file for easy access
        with open('misclassified_results.txt', 'w') as f:
            f.write(f"Misclassified files from {SOURCE_FOLDER}\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Misclassified files: {len(all_misclassified)}\n")
            f.write("=" * 60 + "\n\n")

            for i, item in enumerate(all_misclassified_sorted, 1):
                f.write(f"{i:2d}. {item['file']} (expected: {item['expected_class']}, got: {item['prediction']}, confidence: {item['confidence']:.3f})\n")
                f.write(f"    Full path: {item['full_path']}\n\n")

        print(f"\n💾 Misclassified list saved to: misclassified_results.txt")
    else:
        print(f"\n✅ No misclassified files - perfect accuracy!")

if __name__ == "__main__":
    test_model()