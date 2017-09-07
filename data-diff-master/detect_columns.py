import json
import sys

from fileio import open_datasets


def differ(origin, dest):
    """
    Used to determine the differences
    between the column labels of two
    datasets
    """
    ret = []
    for i in range(len(origin)):
        if origin[i] not in dest:
            ret.append((origin[i], i))
    return ret


def set_differ(origin, dest):
    """
    Takes the simple difference of two sets
    """
    return origin - dest


def mapping_preprocessing(origin_set, map_info, add_del_info, add_del_str):
    """
    Used when a schema file is not provided,
    so that you can find the columns that were
    added, deleted, or renamed
    """
    for each in origin_set:
        x = int(input('{}: 0 -- column was {}, 1 -- column is a renaming of one of the previous columns: '.format(each, add_del_str)))
        if x is 1:
            map_info.append(each)
        else:
            add_del_info.append(each[0])


def pk_detection(dest_headers):
    """
    Used when a schema file is not provided,
    so that you can find the primary key
    """
    print()
    print(dest_headers)
    return int(input('Enter the index of the primary key from the columns of the destination, as listed above: '))


def print_headers(origin_headers, dest_headers):
    """
    Prints the headers of both datasets
    """
    print()
    print('ORIGIN COLUMNS')
    print(origin_headers)
    print()
    print('DESTINATION COLUMNS')
    print(dest_headers)
    print()


def schema_file_check(origin_h, dest_h, schema_file):
    """
    Process the schema file and return the mapping
    and index of the primary key in the destination.

    Look at the sample schema file provided in
    the data/ folder of this repository for how
    to model the schema files

    """
    with open(schema_file) as js_file:
        schema_mapping = json.load(js_file)
        dest_h_list = dest_h.tolist()
        pk = dest_h_list.index(schema_mapping['pk'])
        mapping = {}
        for i in range(len(origin_h)):
            mapped_cur = schema_mapping['schema'][origin_h[i]]
            if mapped_cur != '':
                mapping[dest_h_list.index(mapped_cur)] = i
        if 'added' in schema_mapping:
            for each in schema_mapping['added']:
                mapping[dest_h_list.index(each)] = -1
        return(mapping, pk)


def column_diff_detector(origin, dest, schema_file=None):
    """
    Take in the datasets and then find the column changes
    This can take a schema file, or it can prompt the user
    for input. It is usually easiest to just provide a
    schema file rather than rely on command line input.

    The mapping this function returns is a dictionary
    where the keys are column numbers in the destination
    and the values are column numbers in the origin

    NOTE that values of -1 in the mapping dict indicate
    that the column is new in the destination.
    """
    origin_headers = origin.columns.values
    dest_headers = dest.columns.values
    if schema_file is not None:
        return schema_file_check(origin_headers, dest_headers, schema_file)

    print_headers(origin_headers, dest_headers)

    added = differ(dest_headers, origin_headers)
    deleted = differ(origin_headers, dest_headers)

    # currently -- name_changes_dest -- appears only in destination ds
    # name_changes_origin -- appears only in origin ds
    name_changes_dest = []
    name_changes_origin = []
    true_added = []
    true_deleted = []
    print('Columns from Destination')
    mapping_preprocessing(added, name_changes_dest, true_added, 'added')
    print('\nColumns from Origin')
    mapping_preprocessing(deleted, name_changes_origin, true_deleted, 'deleted')

    results = []
    results_map = dict()
    print('Mapping from Destination Columns to Origin Columns')
    print_headers(origin_headers, dest_headers)
    for each in name_changes_dest:
        old = int(input('{}: enter entry index this maps to in original dataset: '.format(each)))
        results.append((each[0], each[1], old))
        results_map[each[1]] = old
    for i in range(len(dest_headers)):
        if i not in results_map:
            if dest_headers[i] in true_added:
                # -1 is the code for columns that were added in dest dataset
                results_map[i] = -1
            else:
                index = list(origin_headers).index(dest_headers[i])
                results_map[i] = index
    print('Column Mapping, from origin --> destination')
    for k, v in results_map.items():
        if v != -1:
            print(str(origin_headers[v]) + "--->" + str(dest_headers[k]))
        else:
            print("NEW --->" + str(dest_headers[k]))
    pk = pk_detection(dest_headers)

    print('\nPrimary Key -- Column Number: {} Column Name: {}\n'.format(pk, dest_headers[pk]))
    return (results_map, pk)


if __name__ == "__main__":
    (origin, dest) = open_datasets(sys.argv[1], sys.argv[2])
    column_diff_detector(origin, dest)
