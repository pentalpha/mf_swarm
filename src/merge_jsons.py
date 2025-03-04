import sys
import json
from os import path

json_paths = []
output_path = sys.argv[-1]

for n, p in enumerate(sys.argv):
    if n != 0 and p != output_path:
        json_paths.append(p)

data = {path.basename(p).replace('.json', ''): json.load(open(p, 'r')) for p in json_paths}

json.dump(data, open(output_path, 'w'))
