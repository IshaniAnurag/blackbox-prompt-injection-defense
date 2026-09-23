import sys

def main():
    print("Python Executable:", sys.executable)
    print("Python Version:", sys.version)

    modules = ['torch', 'transformers', 'sklearn', 'pandas', 'flask', 'requests', 'yaml']
    all_ok = True
    for mod_name in modules:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, '__version__', 'ok')
            print(f"  [OK] {mod_name} ({ver})")
        except Exception as e:
            print(f"  [FAIL] {mod_name}: {e}")
            all_ok = False

    if all_ok:
        import torch
        print("PyTorch CUDA available:", torch.cuda.is_available())
        print("Environment verification PASSED!")
    else:
        print("Environment verification FAILED!")

if __name__ == "__main__":
    main()
