from tokenise import tokenize
import re
import sys
import json
from collections import Counter

valid_checker = re.compile(r'(^[^\d\W])[^\d\W]*(-[^\d\W]*)*([^\d\W]$)')

def prepreproc(infile, outfile):

    with open(infile, 'r') as f:
        text = f.read()
        with open(outfile, 'w') as g:
            text = re.sub(r'(?<!\n)\n(?!\n)', "\t", text)
            text = re.sub("\n\n", "\n", text)
            g.write(text)

def preproc(infile, outfile):

    with open(infile, 'r') as f:
        with open(outfile, 'w') as g:
            for i, line in enumerate(f):
                if line:
                    sentences = tokenize(line)
                    preproc_lines = []
                    for sentence in sentences:
                        preproc_line = " ".join([t for t in sentence.lower().split() if valid_checker.match(t)])
                        preproc_lines.append(preproc_line)
                        preproc_lines.append("\t")
                    g.write(" ".join(preproc_lines) + "\n")

def extract_frequencylist(infile, language):

    c = Counter()
    with open(infile, 'r') as f:
        for line in f:
            c.update(line.split())

    with open('frequencies_' + language + '.json', 'r') as f:
        json.dump(c, f)

if __name__ == "__main__":
    prepreproc(sys.argv[1], sys.argv[2])
    preproc(sys.argv[2], sys.argv[3])