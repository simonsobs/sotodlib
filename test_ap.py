import argparse as ap

parser = ap.ArgumentParser()

parser.add_argument("--los", type=str, nargs="+")

args = parser.parse_args()

print(args.los)
