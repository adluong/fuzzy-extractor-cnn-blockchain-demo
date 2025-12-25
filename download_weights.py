#!/usr/bin/env python3
"""
FaceNet Weights Download & SSL Fix
===================================

Run this script if you get SSL certificate errors when loading FaceNet.

Usage:
    python download_weights.py

This script will:
1. Fix SSL certificate issues
2. Download FaceNet weights to the correct cache location
3. Verify the download
"""

import os
import sys
import ssl
import urllib.request
import hashlib
from pathlib import Path

# FaceNet weight URLs
WEIGHTS = {
    'vggface2': {
        'url': 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt',
        'filename': '20180402-114759-vggface2.pt',
        'md5': None  # Optional verification
    },
    'casia-webface': {
        'url': 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt',
        'filename': '20180408-102900-casia-webface.pt',
        'md5': None
    }
}

def get_cache_dir():
    """Get the facenet-pytorch cache directory."""
    # facenet-pytorch uses torch.hub cache
    torch_home = os.environ.get('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
    cache_dir = os.path.join(torch_home, 'checkpoints')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def download_with_ssl_fix(url, dest_path):
    """Download file with SSL certificate verification disabled."""
    print(f"Downloading from {url}")
    print(f"Destination: {dest_path}")
    
    # Create SSL context that doesn't verify certificates
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    try:
        # Download with progress
        with urllib.request.urlopen(url, context=ctx) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192
            
            with open(dest_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  Progress: {pct:.1f}% ({mb:.1f} MB)", end='', flush=True)
            
            print()  # New line after progress
        
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def verify_download(path):
    """Verify the downloaded file is a valid PyTorch model."""
    try:
        import torch
        state_dict = torch.load(path, map_location='cpu')
        print(f"  ✓ Valid PyTorch model with {len(state_dict)} parameters")
        return True
    except Exception as e:
        print(f"  ✗ Invalid file: {e}")
        return False

def main():
    print("=" * 60)
    print("FaceNet Weights Downloader (SSL Fix)")
    print("=" * 60)
    
    cache_dir = get_cache_dir()
    print(f"\nCache directory: {cache_dir}")
    
    for name, info in WEIGHTS.items():
        print(f"\n[{name}]")
        
        dest_path = os.path.join(cache_dir, info['filename'])
        
        # Check if already exists
        if os.path.exists(dest_path):
            print(f"  File exists: {dest_path}")
            if verify_download(dest_path):
                print(f"  ✓ {name} weights already downloaded and valid")
                continue
            else:
                print(f"  Removing invalid file...")
                os.remove(dest_path)
        
        # Download
        print(f"  Downloading {name} weights...")
        if download_with_ssl_fix(info['url'], dest_path):
            if verify_download(dest_path):
                print(f"  ✓ {name} weights downloaded successfully!")
            else:
                print(f"  ✗ Download verification failed")
        else:
            print(f"  ✗ Download failed")
    
    print("\n" + "=" * 60)
    print("Done! Try running main.py again.")
    print("=" * 60)

def fix_ssl_globally():
    """Apply global SSL fix (alternative method)."""
    print("\nApplying global SSL fix...")
    
    # Method 1: Set environment variable
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        print("  ✓ Set SSL_CERT_FILE from certifi")
    except ImportError:
        print("  - certifi not installed")
    
    # Method 2: Disable verification (less secure but works)
    ssl._create_default_https_context = ssl._create_unverified_context
    print("  ✓ Disabled SSL verification globally")

if __name__ == "__main__":
    # Apply global fix first
    fix_ssl_globally()
    
    # Then download
    main()
