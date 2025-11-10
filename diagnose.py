"""
Diagnostic script to check if everything is set up correctly.
Run this before training to catch issues early.
"""

import os
import sys

def check_python_version():
    """Check Python version."""
    print("=" * 80)
    print("CHECKING PYTHON VERSION")
    print("=" * 80)
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and 9 <= version.minor <= 11:
        print("✓ Python version is good!")
    else:
        print("⚠ Warning: Python 3.9-3.11 recommended")
    print()

def check_imports():
    """Check if all required packages are installed."""
    print("=" * 80)
    print("CHECKING DEPENDENCIES")
    print("=" * 80)
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'datasets': 'Datasets',
        'accelerate': 'Accelerate',
        'bitsandbytes': 'BitsAndBytes'
    }
    
    all_good = True
    for module, name in required.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {name}: {version}")
        except ImportError as e:
            print(f"✗ {name}: NOT INSTALLED")
            all_good = False
    
    if all_good:
        print("\n✓ All dependencies installed!")
    else:
        print("\n✗ Some dependencies missing. Run: pip install -r requirements.txt")
    print()
    return all_good

def check_cuda():
    """Check CUDA availability."""
    print("=" * 80)
    print("CHECKING CUDA/GPU")
    print("=" * 80)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print("✓ GPU detected and ready!")
        else:
            print("⚠ Warning: No GPU detected. Training will be VERY slow on CPU.")
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
    print()

def check_model_path(model_path):
    """Check if model path exists and has required files."""
    print("=" * 80)
    print("CHECKING MODEL PATH")
    print("=" * 80)
    print(f"Model path: {model_path}")
    
    # Convert to absolute path
    abs_path = os.path.abspath(model_path)
    print(f"Absolute path: {abs_path}")
    
    if not os.path.exists(abs_path):
        print(f"✗ Path does not exist!")
        print(f"  Please check: {abs_path}")
        return False
    
    print("✓ Path exists!")
    
    # Check for required files
    required_files = ['config.json']
    optional_files = ['pytorch_model.bin', 'model.safetensors', 'tokenizer.model', 'tokenizer.json']
    
    print("\nChecking for model files:")
    for fname in required_files:
        fpath = os.path.join(abs_path, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / 1024
            print(f"  ✓ {fname} ({size:.1f} KB)")
        else:
            print(f"  ✗ {fname} MISSING (REQUIRED)")
            return False
    
    print("\nOptional files:")
    for fname in optional_files:
        fpath = os.path.join(abs_path, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath) / (1024**3)  # GB
            print(f"  ✓ {fname} ({size:.2f} GB)")
    
    # List all files in directory
    print("\nAll files in model directory:")
    try:
        files = os.listdir(abs_path)
        for f in sorted(files)[:10]:  # Show first 10
            print(f"  - {f}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
    except Exception as e:
        print(f"  Error listing files: {e}")
    
    print("\n✓ Model path looks good!")
    print()
    return True

def check_data_path(data_path):
    """Check if processed data exists."""
    print("=" * 80)
    print("CHECKING DATA PATH")
    print("=" * 80)
    print(f"Data path: {data_path}")
    
    abs_path = os.path.abspath(data_path)
    print(f"Absolute path: {abs_path}")
    
    if not os.path.exists(abs_path):
        print(f"✗ Path does not exist!")
        print(f"  You need to run: python src/data_preprocessing.py")
        return False
    
    print("✓ Path exists!")
    
    # Check for dataset files
    required = ['dataset_info.json']
    for fname in required:
        fpath = os.path.join(abs_path, fname)
        if os.path.exists(fpath):
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname} MISSING")
            print(f"  You need to run: python src/data_preprocessing.py")
            return False
    
    # Check splits
    splits = ['train', 'validation', 'test']
    for split in splits:
        split_path = os.path.join(abs_path, split)
        if os.path.exists(split_path):
            print(f"  ✓ {split}/ directory exists")
        else:
            print(f"  ✗ {split}/ directory missing")
    
    print("\n✓ Data path looks good!")
    print()
    return True

def test_model_loading(model_path):
    """Try to actually load the model."""
    print("=" * 80)
    print("TESTING MODEL LOADING")
    print("=" * 80)
    print("This will take 30-60 seconds...")
    print("If it hangs here, press Ctrl+C and check the issues below.")
    print()
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Test tokenizer first (fast)
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✓ Tokenizer loaded! Vocab size: {len(tokenizer)}")
        
        # Test model loading (slow)
        print("\nLoading model (this takes time)...")
        print("Progress: ", end="", flush=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print(" Done!")
        
        print(f"✓ Model loaded successfully!")
        print(f"  Parameters: {model.num_parameters():,}")
        print(f"  Device: {model.device}")
        
        # Try a quick generation
        print("\nTesting generation...")
        inputs = tokenizer("SELECT * FROM", return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Generation works! Output: {result[:50]}...")
        
        print("\n✓ Model is fully functional!")
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠ Loading interrupted by user")
        print("\nPossible issues:")
        print("1. Model files are on slow storage (HDD) - loading takes time")
        print("2. Insufficient RAM (need ~16GB)")
        print("3. Model files are corrupted")
        return False
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        print("\nPossible fixes:")
        print("1. Check if model files are complete")
        print("2. Try re-downloading the model")
        print("3. Check you have enough RAM/VRAM")
        return False
    print()

def main():
    print("\n")
    print("=" * 80)
    print(" SQL LORA PROJECT - DIAGNOSTIC TOOL")
    print("=" * 80)
    print()
    
    # Get paths from command line or use defaults
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./Models/llama-2-7b"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "./data/processed"
    
    # Run checks
    check_python_version()
    deps_ok = check_imports()
    
    if not deps_ok:
        print("Please install dependencies first!")
        return
    
    check_cuda()
    model_ok = check_model_path(model_path)
    data_ok = check_data_path(data_path)
    
    if not model_ok:
        print("\n" + "=" * 80)
        print("⚠ MODEL PATH ISSUE DETECTED")
        print("=" * 80)
        print("Please fix the model path before running training.")
        return
    
    if not data_ok:
        print("\n" + "=" * 80)
        print("⚠ DATA PATH ISSUE DETECTED")
        print("=" * 80)
        print("Please run data preprocessing first:")
        print("  python src/data_preprocessing.py")
        return
    
    # Test actual model loading
    print("\nDo you want to test loading the model? (This takes 30-60 seconds)")
    print("Press Enter to test, or Ctrl+C to skip")
    try:
        input()
        test_model_loading(model_path)
    except KeyboardInterrupt:
        print("\nSkipped model loading test.")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print("\nIf all checks passed, you can run:")
    print(f'  python src/train.py --model_name "{model_path}" --data_dir "{data_path}" --output_dir ./models/sql-lora')
    print()

if __name__ == "__main__":
    main()