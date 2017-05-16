from tokenise import tokenize
import re
import sys

valid_checker = re.compile(r'(^[^\d\W])[^\d\W]*(-[^\d\W]*)*([^\d\W]$)')

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

if __name__ == "__main__":
    print('Started preprocessing')
    preproc(sys.argv[1], sys.argv[2])
    print('Finished')