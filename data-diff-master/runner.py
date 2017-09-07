import detect_columns
import detect_rows
import modification_searching
import fileio
import find_differing_rows as fdr

import sys


def main():
    """
    Main function for Data-Diff
    This mostly opens up the datasets, and calls necessary scripts
    """
    (origin, dest) = fileio.open_datasets(sys.argv[1], sys.argv[2])

    # if a schema file is provided, it uses it, else it prompts the user
    if len(sys.argv) >= 4:
        (mapping, pk) = detect_columns.column_diff_detector(origin, dest, schema_file=sys.argv[3])
    else:
        (mapping, pk) = detect_columns.column_diff_detector(origin, dest)

    # find and print removals
    row_removed = detect_rows.row_removal_detection(origin, dest, mapping, pk)
    if row_removed:
        detect_rows.print_mods(row_removed)

    if len(sys.argv) == 5:
        search_op = sys.argv[4]
    else:
        search_op = "random"
    # find and print modifications
    mods = modification_searching.single_modification_per_column(origin, dest, mapping, pk, search_op)
    if mods:
        modification_searching.print_mods(mods)

    # find and print new rows
    added_rows = fdr.find_added(origin, dest, mapping, pk)
    if added_rows:
        print('Added rows with PKs {}'.format(added_rows))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('USAGE: python runner.py <origin_dataset> <destination_dataset> <optional_schema_file>')
        sys.exit(1)
    main()
