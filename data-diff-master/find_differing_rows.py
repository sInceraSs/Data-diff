import pandas as pd


def df_same_cols(origin, dest, column_map, pk_dest):
    """
    Given the origin, destination, column mapping,
    and destination primary key index, create two dataframes
    that have the same column labels. Essentially, make the
    columns of the origin and destination the same.
    """
    # find set up dataframes for origin and dest that only
    # have the columns of interest
    # need to iterate over the column mapping for this
    # ignore the columns that don't exist in the origin
    dest_cols = []
    origin_cols = []
    for dest_col_num, origin_col_num in column_map.items():
        if origin_col_num != -1:
            dest_cols.append(dest_col_num)
            origin_cols.append(origin_col_num)

    origin_selected_cols = origin.iloc[:, origin_cols]
    dest_selected_cols = dest.iloc[:, dest_cols]
    origin_selected_cols.columns = dest_selected_cols.columns.values

    return (origin_selected_cols, dest_selected_cols)


def find_removed(origin, dest, column_map, pk_dest):
    """
    This function returns the primary keys of the rows
    that have been removed in the destination dataset
    """
    origin_pks = set(origin.iloc[:, column_map[pk_dest]])
    dest_pks = set(dest.iloc[:, pk_dest])
    return origin_pks - dest_pks


def find_added(origin, dest, column_map, pk_dest):
    """
    This function returns the primary keys of the rows
    that have been added in the destination dataset
    """
    origin_pks = set(origin.iloc[:, column_map[pk_dest]])
    dest_pks = set(dest.iloc[:, pk_dest])
    return dest_pks - origin_pks


def find_modified(origin, dest, column_map, pk_dest):
    """
    This function returns the primary keys of the rows
    that have been modified in the destination dataset
    """
    dest_pks = set(dest.iloc[:, pk_dest])

    origin_selected_cols, dest_selected_cols = df_same_cols(origin, dest, column_map, pk_dest)
    pk_label_dest = dest.columns.values[pk_dest]
    origin_selected_cols = origin_selected_cols.set_index(pk_label_dest)
    dest_selected_cols = dest_selected_cols.set_index(pk_label_dest)

    to_drop_origin = list(find_removed(origin, dest, column_map, pk_dest))
    to_drop_dest = list(find_added(origin, dest, column_map, pk_dest))
    origin_selected_cols.drop(to_drop_origin, inplace=True)
    dest_selected_cols.drop(to_drop_dest, inplace=True)

    in_both = pd.merge(origin_selected_cols.reset_index(), dest_selected_cols.reset_index(), how='inner')
    in_both_set = set(in_both.loc[:, pk_label_dest])

    """
    modified = find_modified(origin, dest, column_map, pk_dest)
    o_in = origin.set_index(origin.columns.values[column_map[pk_dest]])
    d_in = dest.set_index(dest.columns.values[pk_dest])
    print(o_in.loc[list(modified)])
    print(d_in.loc[list(modified)])

    this is how you could easily use indexing to print out all the modified values
    """
    return(dest_pks - in_both_set - set(to_drop_dest))
