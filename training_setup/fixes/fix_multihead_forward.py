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
    
    # We want to make the 'sar_head' call conditional
    # Target: sar_out = self.sar_head(x, targets[1:])
    # Replacement: sar_out = self.sar_head(x, targets[1:]) if hasattr(self, 'sar_head') else {}
    
    for line in lines:
        if "sar_out = self.sar_head(x, targets[1:])" in line:
            # Preserve indentation
            indent = line[:line.find("sar_out")]
            new_line = f"{indent}sar_out = self.sar_head(x, targets[1:]) if hasattr(self, 'sar_head') else {{}}\n"
            new_lines.append(new_line)
            patches_applied += 1
            print("✅ Patched 'sar_head' call.")
            
        elif "nrtr_out = self.gtc_head(x, targets[1:])" in line:
             # Just in case NRTR is called this way
            indent = line[:line.find("nrtr_out")]
            new_line = f"{indent}nrtr_out = self.gtc_head(x, targets[1:]) if hasattr(self, 'gtc_head') else {{}}\n"
            new_lines.append(new_line)
            patches_applied += 1
            print("✅ Patched 'gtc_head' call.")
            
        # Catch generic variable assignments from missing heads
        elif "= self.sar_head(" in line and "if hasattr" not in line:
             indent = line[:line.find(line.strip())]
             parts = line.strip().split('=')
             var_name = parts[0].strip()
             new_line = f"{indent}{var_name} = self.sar_head({parts[1].split('(')[1]} if hasattr(self, 'sar_head') else {{}}\n"
             # This generic catch is risky, stick to exact match above first
             new_lines.append(line)
             
        else:
            new_lines.append(line)

    if patches_applied > 0:
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"🚀 Successfully patched {patches_applied} issues in rec_multi_head.py")
    else:
        print("⚠️ Warning: No patches applied. The code might look different than expected.")
        # Debug: Print lines around 148 to see what it looks like
        print("Context around line 148:")
        start = max(0, 145)
        end = min(len(lines), 155)
        for i in range(start, end):
            print(f"{i+1}: {lines[i].strip()}")

if __name__ == "__main__":
    apply_patch()