import os
import re

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in lines:
                end = False
                if '%' in line:
                    end = True
                cleaned_line = re.split(r'%', line, maxsplit=1)[0]  # Remove everything after and including '%'
                f.write(cleaned_line)  # Write only the cleaned line without adding extra new lines
                if end:
                    break
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            process_file(file_path)

if __name__ == "__main__":
    folder_path = '3sat/test'
    if os.path.isdir(folder_path):
        process_folder(folder_path)
        print("Processing complete.")
    else:
        print("Invalid folder path.")
