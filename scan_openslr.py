import requests
import re

requests.packages.urllib3.disable_warnings()

slrs = [63, 64, 65, 66, 67, 68, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 103]

for slr in slrs:
    url = f"https://www.openslr.org/resources/{slr}/"
    try:
        r = requests.get(url, verify=False, timeout=5)
        if r.status_code == 200:
            lines = [line for line in r.text.splitlines() if any(ext in line for ext in ['.zip', '.tar.gz', '.tgz'])]
            print(f"SLR {slr}:")
            for line in lines[:3]:
                print("  ", line.strip())
    except Exception as e:
        print(f"SLR {slr} error: {e}")
