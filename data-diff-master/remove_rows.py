import csv
import sys

"""
python remove_rows.py crimes2015.csv crimes2015-mod2.csv Arrest false
"""

"""
This is mostly a quick script to perform a selection query on a dataset
and then output a new csv.
See usage notes.
"""

def remove(column_name, value, incsv, outcsv):
    headers = next(incsv)
    outcsv.writerow(headers)
    for i in range(len(headers)):
        if column_name == headers[i]:
            break
    column_num = i
    for each in incsv:
        if value.lower() != each[column_num].lower():
            outcsv.writerow(each)


def setup(infname, outfname):
    infile = open(infname, 'r')
    outfile = open(outfname, 'w')
    incsv = csv.reader(infile)
    outcsv = csv.writer(outfile)
    return (incsv, outcsv)

if __name__ == "__main__":
    if len(sys.argv) == 5:
        (incsv, outcsv) = setup(sys.argv[1], sys.argv[2])
        remove(sys.argv[3], sys.argv[4], incsv, outcsv)
    else:
        print('usage -- python infilename outfilename columnname columnval')
        print('infilename -- file to read')
        print('outfilename -- file to output new dataset to')
        print('columnname -- the column on which you will predicate')
        print('columnval -- the column value you want to remove')
