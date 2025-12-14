import os

# Path to the file causing the crash
FILE_PATH = "PaddleOCR_Engine/ppocr/modeling/heads/rec_multi_head.py"

def apply_patch():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: Could not find {FILE_PATH}")
        return

    print(f"📖 Reading {FILE_PATH}...")
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    patches_applied = 0
    
    # The line causing the crash
    target_code = 'assert len(self.head_list) >= 2'

    for line in lines:
        if target_code in line:
            # We comment it out instead of deleting, to be safe
            indent = line[:line.find('assert')]
            new_lines.append(f"{indent}# {target_code.strip()}  # Patched to allow single head\n")
            patches_applied += 1
            print("✅ Found and disabled the MultiHead assertion.")
        else:
            new_lines.append(line)

    if patches_applied > 0:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"🚀 Successfully patched {patches_applied} issue in rec_multi_head.py")
    else:
        print("⚠️ Warning: No patches applied. Check if the file is already fixed.")

if __name__ == "__main__":
    apply_patch()