# it seems to be overly complicated, so I decide to use simplier and more straigtforward way to do subparsing
import argparse

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='types of A')
parser.add_argument("-v")

a_parser = subparsers.add_parser("A")
b_parser = subparsers.add_parser("B")

a_parser.add_argument("something", choices=['a1', 'a2'])