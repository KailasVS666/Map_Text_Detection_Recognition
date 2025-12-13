import os

# Path to the file causing the crash
FILE_PATH = "PaddleOCR_Engine/tools/train.py"

def apply_patch():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: Could not find {FILE_PATH}")
        return

    print(f"📖 Reading {FILE_PATH}...")
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    patches_applied = 0
    
    # We will replace these specific unsafe patterns with safe ones
    replacements = {
        # Fix 1: SARLoss Check
        'if list(config["Loss"]["loss_config_list"][1].keys())[0] == "SARLoss":': 
        'if "loss_config_list" in config["Loss"] and len(config["Loss"]["loss_config_list"]) > 1 and list(config["Loss"]["loss_config_list"][1].keys())[0] == "SARLoss":',
        
        # Fix 2: NRTRLoss Check
        'elif list(config["Loss"]["loss_config_list"][1].keys())[0] == "NRTRLoss":':
        'elif "loss_config_list" in config["Loss"] and len(config["Loss"]["loss_config_list"]) > 1 and list(config["Loss"]["loss_config_list"][1].keys())[0] == "NRTRLoss":'
    }

    for line in lines:
        patched_line = line
        for unsafe, safe in replacements.items():
            if unsafe in line:
                # Preserve indentation
                indent = line[:line.find(unsafe)]
                patched_line = f"{indent}{safe}\n"
                patches_applied += 1
                print(f"✅ Patched unsafe check: {unsafe.split('==')[1].strip()}")
                break # Only patch once per line
        new_lines.append(patched_line)

    if patches_applied > 0:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"🚀 Successfully patched {patches_applied} bugs in train.py")
    else:
        print("⚠️ Warning: No patches applied. Either the file is already fixed or the code looks different.")

if __name__ == "__main__":
    apply_patch()