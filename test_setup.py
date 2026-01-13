#!/usr/bin/env python3
import os
import sys

print("Testing setup...")
print(f"Python: {sys.version}")
print(f"Current dir: {os.getcwd()}")

# Check required files
files_needed = ['run.py', 'requirements.txt', 'config.py', 'src/__init__.py']
for file in files_needed:
    exists = os.path.exists(file)
    print(f"{'✅' if exists else '❌'} {file}: {exists}")

# Check data directories
dirs_needed = ['data/raw_pdfs', 'src']
for dir_path in dirs_needed:
    exists = os.path.isdir(dir_path)
    print(f"{'✅' if exists else '❌'} {dir_path}/: {exists}")

# Check for PDFs
if os.path.exists('data/raw_pdfs'):
    pdfs = [f for f in os.listdir('data/raw_pdfs') if f.endswith('.pdf')]
    print(f"\nFound {len(pdfs)} PDF files:")
    for pdf in pdfs[:5]:
        print(f"  • {pdf}")
    if len(pdfs) > 5:
        print(f"  • ... and {len(pdfs) - 5} more")
else:
    print("\n❌ data/raw_pdfs/ directory not found!")