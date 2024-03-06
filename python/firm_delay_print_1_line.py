#!/usr/bin/python3

import glob
import argparse
import re

parser = argparse.ArgumentParser(description='Print first line that is not a comment')
parser.add_argument("--glob", type=str)
args=parser.parse_args()

files = [a for a in glob.glob(args.glob)]
files.sort(key=lambda n: float(re.findall("\d+", n)[-1]))
p = re.compile('\d*')
print("#chain_len CritBuffer formula")


def sum(n):
	s=0.0
	for i in range(1,n+1):
		s+=1/i
	return s

for file in files:
	f = open(file, "r")
	for line in f.readlines():
		if not line.startswith("#"):
			fl = re.findall("\d+\.\d+", line)
			chain_len = re.findall("\d+", file)
			c=int(chain_len[-1])
			s=sum(c)
			print(chain_len[-1], fl[-1], s)
			break
