#!/usr/bin/env python3

import sys
import oarg

TO_EXCLUDE = [
    #"fmm:splash2x",
    #"volrend:splash2x",
    #"radiosity:splash2x"
]

def transform(line, rev=False, new_fmt=False):
    if new_fmt:
        return line

    if rev:
        name, suite, *stats = line.split(",")
    else:
        suite, name, *stats = line.split(",")
    return "%s,%s" % (name + ":" + suite, ",".join(stats))

def main():
    filename = oarg.Oarg("-f --filename", "", "input filepath", 0)
    new_fmt = oarg.Oarg("-n --new-fmt", False, "format name:suite")
    rev = oarg.Oarg("-r --rev", False,
        "reverse order from suite,name to name,suite")
    out = oarg.Oarg("-o --out", "", "output file")
    hlp = oarg.Oarg("-h --help", False, "this help message")
    oarg.parse()

    if hlp.val:
        oarg.describe_args("options:")
        exit()

    if not filename.found:
        print("error: must pass filename (use -h for help)")
        exit()

    with open(filename.val, "r") as f:
        lines = [l.strip().lower() for l in f]
        header, *lines = lines

    lines = list(map(lambda l: transform(l, rev.val, new_fmt.val), lines))
    lines = list(filter(lambda l: l.split(",")[0] not in TO_EXCLUDE, lines))
    lines = sorted(lines, key=lambda l: l.split(",")[0])

    out_file = sys.stdout if not out.val else open(out.val, "w")
    print(transform(header, rev.val, new_fmt.val), file=out_file)
    print("\n".join(lines), file=out_file)
    out_file.close()

if __name__ == "__main__":
    main()
