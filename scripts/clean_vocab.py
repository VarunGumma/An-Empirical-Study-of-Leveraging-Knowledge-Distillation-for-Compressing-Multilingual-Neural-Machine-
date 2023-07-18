import sys
import codecs

def clean_vocab():
    for line in sys.stdin:
        line = codecs.decode(line, 'utf-8')
        fields = line.strip("\r\n ").split(" ")
        if len(fields) == 2:
            print(line)

if __name__ == "__main__":
    clean_vocab()