import json
import sys
from collections import Counter

def extract_frequencylist(infile, language):
    c = Counter()
    with open(infile, 'r') as f:
        for line in f:
            c.update(line.split())

    with open('../data/frequencies_' + language + '.json', 'r') as f:
        json.dump(c, f)

if __name__ == "__main__":
    extract_frequencylist(sys.argv[1], sys.argv[2])