import os
import re

RESULTS_DIR = "/home/sungj4/projects/digital-pathology/code/Results_figure_ext"

# Define thresholds
MIN_LOG_RANK = 0     # Skip if test log-rank <= this value
MAX_LOG_RANK = 1     # Skip if test log-rank > this value
MIN_C_INDEX  = 0     # Skip if test c-index < this value

# Regex pattern to parse the file name structure:
#   {model}_{loss}__Epc{epoch}_[{lr_train}-{lr_valid}-{lr_test}]_[{c_train}-{c_valid}-{c_test}]_Ext.png
pattern = re.compile(
    r'^(?P<model>\w+)_'
    r'(?P<loss>\w+)__Epc'
    r'(?P<epoch>\d+)_\['
    r'(?P<lr_train>[-\d.]+)-'
    r'(?P<lr_valid>[-\d.]+)-'
    r'(?P<lr_test>[-\d.]+)\]_'
    r'\['
    r'(?P<c_train>[-\d.]+)-'
    r'(?P<c_valid>[-\d.]+)-'
    r'(?P<c_test>[-\d.]+)\]_Ext.*\.png$'
)

results_list = []  # List to store all epoch results

for filename in os.listdir(RESULTS_DIR):
    match = pattern.match(filename)
    if not match:
        continue
    
    model = match.group('model')
    loss = match.group('loss')
    epoch = int(match.group('epoch'))
    lr_test = float(match.group('lr_test'))
    c_test = float(match.group('c_test'))
    
    # Skip if log-rank is out of range (<= MIN_LOG_RANK or > MAX_LOG_RANK)
    if lr_test <= MIN_LOG_RANK or lr_test > MAX_LOG_RANK:
        continue
    
    # Skip if test c-index is below our threshold
    if c_test < MIN_C_INDEX:
        continue
    
    results_list.append({
        'model': model,
        'loss': loss,
        'epoch': epoch,
        'lr_test': lr_test,
        'c_test': c_test,
        'filename': filename
    })

# Sort the list by test c-index in descending order
sorted_by_c_index = sorted(results_list, key=lambda x: x['c_test'], reverse=True)

# Print out all epoch results sorted by test c-index
print("=== Sorted by Test C-Index (Descending) ===")
for info in sorted_by_c_index:
    print(f"[{info['model']} - {info['loss']}] Epoch {info['epoch']} | "
          f"Test C-Index = {info['c_test']:.3f} | Test Log-Rank = {info['lr_test']:.3f} | "
          f"File: {info['filename']}")
