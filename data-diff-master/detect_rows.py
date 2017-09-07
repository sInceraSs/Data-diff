from find_differing_rows import find_added


def row_removal_detection(origin, dest, column_map, pk_dest):
    """
    params

    origin: dataframe for origin
    dest: dataframe for destination
    column_map: integer index mapping from labels of destination to origin
    pk_dest: integer index of the primary key of the destination
    """
    to_drop_dest = list(find_added(origin, dest, column_map, pk_dest))
    if to_drop_dest:
        dest = dest[~(dest.iloc[:, pk_dest].isin(to_drop_dest))]
    origin_headers = origin.columns.values
    dest_headers = dest.columns.values
    set_difference_map = {}
    for i in range(len(dest_headers)):
        # Note that entries in the map are -1
        # if the column is new in dest
        if column_map[i] != -1:
            cur_set_origin = set(origin.iloc[:, column_map[i]])
            cur_set_dest = set(dest.iloc[:, i])
            diff = cur_set_origin - cur_set_dest
            if diff and len(diff) < 20:
                set_difference_map[column_map[i]] = diff

    last_good_assignment = origin
    dest_pks = set(dest.iloc[:, pk_dest])
    found_perfect_match = False
    applied_modifications = {}

    while set_difference_map:
        cur = min(set_difference_map,
                  key=lambda key: len(set_difference_map[key]))
        cur_assignment = remove_rows(last_good_assignment,
                                     cur,
                                     set_difference_map[cur])
        cur_assignment_quality = assignment_quality(set(cur_assignment.iloc[:, column_map[pk_dest]]),
                                                    dest_pks)

        if cur_assignment_quality >= 1:
            last_good_assignment = cur_assignment
            applied_modifications[origin_headers[cur]] = set_difference_map[cur]
            if cur_assignment_quality >= 2:
                found_perfect_match = True
                break

        set_difference_map.pop(cur, None)

    if found_perfect_match:
        print('Found a perfect matching')
    else:
        print('Found a close matching')
    return applied_modifications


def print_mods(applied_modifications):
    for key, value in applied_modifications.items():
        print('Removed values {} from origin column: {}'.format(value, key))


def remove_rows(origin, removal_col, removal_values):
    """
        origin -- the dataframe to remove values from
        removal_col -- column number in origin array to check values against
        removal_values -- remove rows where the value at removal_col is in this set
    """
    # rows_to_remove = []
    # for i in range(len(origin)):
    #     if origin.iloc[i,removal_col] in removal_values:
    #         rows_to_remove.append(i)

    # x = origin.drop(rows_to_remove)
    x = origin[~(origin[origin.columns[removal_col]].isin(removal_values))]
    return x


def assignment_quality(modified_pks, dest_pks):
    """
        Returns 0 if the assignment is bad
        Returns 1 if the assignment is good
        Returns 2 if the assignment is perfect
    """
    if dest_pks - modified_pks:
        return 0
    if modified_pks - dest_pks:
        return 1
    return 2
