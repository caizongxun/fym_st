#!/usr/bin/env python3
"""Script to remove all emojis from Python files"""

import os
import re
from pathlib import Path

def remove_emojis(text):
    """Remove emojis from text"""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # supplemental symbols
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def process_file(filepath):
    """Process a single file to remove emojis"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        new_content = remove_emojis(content)
        
        if new_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Processed: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Main function"""
    # Define directories to process
    base_dir = Path("trading_system")
    dirs_to_process = [
        base_dir / "gui" / "pages",
        base_dir / "core",
        base_dir / "workflows",
        base_dir / "config"
    ]
    
    processed_count = 0
    total_count = 0
    
    for directory in dirs_to_process:
        if not directory.exists():
            print(f"Directory not found: {directory}")
            continue
        
        print(f"\nProcessing directory: {directory}")
        
        for filepath in directory.rglob("*.py"):
            total_count += 1
            if process_file(filepath):
                processed_count += 1
    
    print(f"\n=== Summary ===")
    print(f"Total files scanned: {total_count}")
    print(f"Files with emojis removed: {processed_count}")
    print(f"Files unchanged: {total_count - processed_count}")

if __name__ == "__main__":
    main()