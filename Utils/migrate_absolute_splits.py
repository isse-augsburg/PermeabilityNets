from argparse import ArgumentParser
from pathlib import Path
import sys
import pickle
import os


def migrate_paths(paths, root):
    return [str(Path(p).relative_to(Path(root))).strip('/') for p in paths]


def migrate_split(ifile, ofile, root):
    with open(ifile, "rb") as f:
        filenames_str = pickle.load(f)
    unix_abs = all(fn.startswith('/') for fn in filenames_str)
    if os.name == 'nt' and unix_abs:  # Simon workaround
        # If the paths were already saved as Windows paths, as in the tests, do nothing
        # Explicitly not using type() and WindowsPath here, since this Class is not implemented on Linux
        # -> Check would not work
        if filenames_str[0][0] != 'Y' and filenames_str[0][0] != 'X':
            filenames_str = [Path('Y:/') / '/'.join(x.parts[3:]) for x in filenames_str]
    filenames = [Path(fn) for fn in filenames_str]
    all_abs = all(p.is_absolute() for p in filenames)
    if not all_abs:
        print("Error: Not all paths in the input file are absolute!", file=sys.stderr)
        exit(1)

    migrated = migrate_paths(filenames, root)
    with open(ofile, "wb") as f:
        pickle.dump(migrated, f)


def main():
    parser = ArgumentParser(description="Migrates an absolute split pickle to relative paths")
    parser.add_argument("-i", "--input", help="The input file", required=True)
    parser.add_argument("-o", "--output", help="The output file", required=True)
    parser.add_argument("-r", "--root", help="The data root, must be a prefix of every loaded path", required=True)

    args = parser.parse_args()
    migrate_split(args.input, args.output, args.root)


if __name__ == '__main__':
    main()
