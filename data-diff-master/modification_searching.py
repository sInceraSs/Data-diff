import fileio
import sys

import numpy as np
from time import clock
from column_regression import regression
import detect_columns
import find_differing_rows as fdr


def single_modification_per_column(origin, dest, column_map, pk_dest, search_op):
    """
    Assumes that a single modification was made to the columns that have changed,
    and performs a regression (or whatever the Modification object from
    column_regression.py chooses to perform).

    If the quality of that assignment is bad, it tries to search for a
    split point, and looks for if two modifications are sufficient.

    It returns a dictionary of modifications, where the keys are
    the column labels, and the values are Modification objects, or
    are dictionaries themselves. See documentation below for
    split_modifications to see what the contents of that dictionary
    will be.
    """
    modified_pks = fdr.find_modified(origin, dest, column_map, pk_dest)
    if modified_pks:
        origin_cols, dest_cols = \
            fdr.df_same_cols(origin, dest, column_map, pk_dest)

        pk_label_dest = dest.columns.values[pk_dest]
        origin_cols = origin_cols.set_index(pk_label_dest).loc[modified_pks]
        dest_cols = dest_cols.set_index(pk_label_dest).loc[modified_pks]

        mods = {}
        for each_col in origin_cols.columns.values:
            keep = origin_cols[each_col] != dest_cols[each_col]
            origin_col = origin_cols.loc[keep, each_col]
            dest_col = dest_cols.loc[keep, each_col]
            if not origin_col.equals(dest_col):
                cur_mod = regression(origin_col, dest_col)
                if cur_mod.quality():
                    mods[each_col] = cur_mod
                else:
                    start = clock()
                    if search_op == "naive":
                        mods[each_col] = split_modifications(each_col, origin_col, dest_col)
                    elif search_op == "random":
                        mods[each_col] = random_linear_modifications(each_col,origin_col,dest_col, False)
                    elif search_op == "greedy":          
                        mods[each_col] = greedy_linear_modifications(each_col,origin_col,dest_col)
                    elif search_op == "no_greedy":
                        mods[each_col] = nonoverlap_greedy_linear_modifications(each_col,origin_col,dest_col)
                    end = clock()
                    print "split", end - start
        return mods


def split_modifications(colname, origin_col, dest_col):
    """
    If you find that a single modification to a column does
    not prove to be effective, failover to this mechanism,
    which searches for a split point in the data.

    It then finds one modification for values <= than the split
    point and another modification for values > than the split
    point

    Returns a dictionary, where the keys are a string
    representing which side of the split point, and the
    values are Modification objects. (See column_regression.py)
    """
    # could do a binary search rather than a linear one
    for split_point in sorted(origin_col.unique()):
        cond = origin_col <= split_point
        oppcond = ~cond
        o_first = origin_col[cond]
        o_second = origin_col[oppcond]
        d_first = dest_col[cond]
        d_second = dest_col[oppcond]
        out_dct = {}
        f_reg = regression(o_first, d_first)
        s_reg = regression(o_second, d_second)
        if f_reg.quality() and s_reg.quality():
            out_dct['<={}'.format(split_point)] = f_reg
            out_dct['>{}'.format(split_point)] = s_reg

            return out_dct
    return 'could not find modifications for the following PKs: {}'.format(set(origin_col.index.values))


def print_mods(mod_dict):
    """
    Function that can print the modifications,
    when given the dictionary of the modification
    """
    for k, v in mod_dict.items():
        if isinstance(v, dict):
            for cond, mod in v.items():
                print('Modified {} with condition {} using {}'
                      .format(k, cond, mod))
        else:
            print('Modified {} using {}'.format(k, v))

def random_linear_modifications(colname, origin_col, dest_col, ran=False):
    """
    Implement a random search algorithm to match the change of each coloum
    Under the assumption that the selection can select any tuples even their values overlap
    """
    o_tuples, d_tuples = sorted(origin_col.unique()), sorted(dest_col.unique())
    cur_tuples = sorted(origin_col.unique())
    res_set = list()
    while len(cur_tuples):
        pk_tuple = cur_tuples[ np.random.randint(0, len(cur_tuples)) ]
        pt_idx = o_tuples.index(pk_tuple)
        pk_set, cur_set = [ pk_tuple ], dict()
        max_coef, max_flag = 1, [0, d_tuples[pt_idx]]
        for o_tuple in cur_tuples:
            ot_idx = o_tuples.index(o_tuple)
            if pk_tuple != o_tuple:
                coef = float( d_tuples[ot_idx] - d_tuples[pt_idx]) / (o_tuple - pk_tuple)
                intercept = float(d_tuples[ot_idx]) - o_tuple*coef
                if coef not in cur_set:
                    cur_set[ coef ] = [ intercept ]
                cur_set[ coef ].append( o_tuple )
        for coef in cur_set:
            if len(cur_set[ coef ]) > max_coef:
                max_coef =  len(cur_set[ coef ])
                max_flag = [ coef, cur_set[ coef ][0] ]
        #choose to random select or first meet
        if not ran:
            if max_flag[0] in cur_set:
                for o_tuple in cur_set[ max_flag[0] ][1:]:
                    pk_set.append(o_tuple)
        else:
            max_choice = []
            if max_coef == 2:
                max_choice.append( [pk_tuple] )
            for coef in cur_set:
                if len(cur_set[coef]) == max_coef:
                    max_choice.append( [coef] + cur_set[coef] )
            if max_choice:
                pk = np.random.randint( len(max_choice) )
                max_flag = max_choice[pk][:2]
                for o_tuple in max_choice[pk][2:]:
                    pk_set.append( o_tuple )
        for pk_tuple in pk_set:
            cur_tuples.remove(pk_tuple)
        res_set.append( max_flag + [min(pk_set), max(pk_set), len(pk_set)] )
    mods = dict()
    for coef, intercept, min_, max_, range_ in res_set:
        mods['{} tuples from [{}, {}]'.format(range_, min_, max_)] = 'linear modification with coef {:.2f} and intercept {:.2f}'.\
            format(coef, intercept)
    return mods

def greedy_linear_modifications(colname, origin_col, dest_col):
    """
    Implement a greedy search algorithm to match the change of each coloum
    Under the assumption that the selection can select any tuples even their values overlap
    """
    o_tuples, d_tuples = sorted(origin_col.unique()), sorted(dest_col.unique())
    cur_tuples = sorted(origin_col.unique())
    res_set = list()
    while len(cur_tuples):
        cur_set = dict()
        max_flag, max_coef = list(), list()
        for o_tuple in cur_tuples:
            max_o_coef = list()
            ot_idx = o_tuples.index(o_tuple)
            ot_coef = { (float('nan'), d_tuples[ot_idx]):[ o_tuple ] }
            for pk_tuple in cur_tuples:
                if pk_tuple != o_tuple:
                    pt_idx = o_tuples.index(pk_tuple)
                    coef = float( d_tuples[ot_idx] - d_tuples[pt_idx]) / (o_tuple - pk_tuple)
                    intercept = float(d_tuples[ot_idx]) - o_tuple*coef
                    if (coef, intercept) not in ot_coef:
                        ot_coef[ (coef, intercept) ] = [ o_tuple ]
                    ot_coef[ (coef, intercept) ].append( pk_tuple )
            for (coef, intercept) in ot_coef:
                if len( ot_coef[ (coef, intercept) ] ) > len( max_o_coef ):
                    max_o_coef = ot_coef[ (coef, intercept) ]
            if max_o_coef not in max_flag:
                max_flag.append(  [coef, intercept, max_o_coef] )
            del ot_coef
        for pk_set in max_flag:
            if len(pk_set) > len(max_coef):
                max_coef = pk_set
        for pk_tuple in max_coef[2]:
            cur_tuples.remove(pk_tuple)
        res_set.append( [ max_coef[0], max_coef[1], min(max_coef[2]), max(max_coef[2]), len(max_coef[2]) ] )
    mods = dict()
    for coef, intercept, min_, max_, range_ in res_set:
        mods['{} tuples from [{}, {}]'.format(range_, min_, max_)] = 'linear modification with coef {:.2f} and intercept {:.2f}'.\
            format(coef, intercept)
    return mods

def nonoverlap_greedy_linear_modifications(colname, origin_col, dest_col):
    """
    Implement a greedy algorithm to match the change of each coloum if the selection
    Under the assumption that the 'Range' predicate does not overlap with each other
    And all operations are finished at the same time
    """
    o_tuples, d_tuples = sorted(origin_col.unique()), sorted(dest_col.unique())
    pre_coef = float('nan')
    res_set, res_op = [ [o_tuples[0]] ], [ (pre_coef, d_tuples[0]) ]
    for i in range(1, len(o_tuples)):
        o_tuple, d_tuple = o_tuples[i], d_tuples[i]
        coef = float(d_tuple - d_tuples[i-1]) / (o_tuple - o_tuples[i-1])
        intercept = float(d_tuples[i]) - o_tuple*coef
        if coef == pre_coef or pre_coef != pre_coef:
            res_set[-1].append( o_tuple )
            res_op[-1] = (coef, intercept)
            pre_coef = coef
        else:
            res_set.append( [ o_tuple ] )
            res_op.append( (float('nan'), d_tuple) )
            pre_coef = float('nan')
    # reduce the same operation 
    reduce_res, reduce_op, reduce_set = dict(), list(), list()
    for cur_op_idx in range( len(res_op) ):
        cur_op = res_op[ cur_op_idx ]
        if cur_op not in reduce_res:
            reduce_res[ cur_op ] = [1, min(res_set[cur_op_idx]), max(res_set[cur_op_idx]), [min(res_set[cur_op_idx]), max(res_set[cur_op_idx])] ]
        for pre_op_idx in range( cur_op_idx+1, len(res_op) ):
            if cur_op_idx != pre_op_idx and res_op[cur_op_idx] == res_op[pre_op_idx]:
                reduce_res[ cur_op ][0] += 1
                reduce_res[ cur_op ][1] = min(reduce_res[ cur_op ][1], min(res_set[pre_op_idx]))
                reduce_res[ cur_op ][2] = max(reduce_res[ cur_op ][2], max(res_set[pre_op_idx]))
                reduce_res[ cur_op ].append( [min(res_set[pre_op_idx]), max( res_set[pre_op_idx] ) ] )
    for op in reduce_res:
        reduce_op.append( [reduce_res[op][0], op, True] )
    reduce_op.sort(reverse=True)
    for op_idx in range(len(reduce_op)):
        if reduce_op[ op_idx ][2]:
            cur_op = reduce_op[ op_idx ][1]
            op_min, op_max = reduce_res[ cur_op ][1], reduce_res[ cur_op ][2]
            reduce_set.append( [ cur_op, op_min, op_max] )
            for modif_op_idx in range(op_idx, len(reduce_op)):
                modif_op =  reduce_op[ modif_op_idx ][1]
                modif_op_min, modif_op_max = reduce_res[ modif_op ][1], reduce_res[ modif_op ][2]
                #since non-overlap
                if (modif_op_min > op_min and modif_op_min < op_max):
                    for modif_range_idx in range(3, len(reduce_res[ modif_op ])):
                        modif_op_min, modif_op_max = reduce_res[ modif_op ][modif_range_idx]
                        if (modif_op_min > op_min and modif_op_min < op_max):
                            modif_op_next = (modif_op[0]/cur_op[0], modif_op[1]-modif_op[0]*cur_op[1]/cur_op[0])
                            reduce_op[ modif_op_idx ][0] -= 1
                            reduce_set.append( [ modif_op_next, modif_op_min, modif_op_max] )
                            reduce_res[ modif_op ][modif_range_idx] = [float('inf'), float('-inf')] 
                    min_, max_ = float('inf'), float('-inf')
                    for modif_op_min, modif_op_max in reduce_res[ modif_op ][3:]:
                        min_ = min(modif_op_min, min_)
                        max_ = max(modif_op_max, max_)
                    reduce_res[ modif_op ][1:3] = min_, max_
                    reduce_op.sort(reverse=True)
    """
    single point needs more concerns
    """
    mods = dict()
    for (coef, intercept), min_, max_ in reduce_set:
        mods['range from [{}, {}]'.format(min_, max_)] = 'linear modification with coef {:.2f} and intercept {:.2f}'.\
            format(coef, intercept)
    return mods
    
if __name__ == "__main__":
    (origin, dest) = fileio.open_datasets(sys.argv[1], sys.argv[2])
    (mapping, pk) = detect_columns.column_diff_detector(origin,
                                                        dest,
                                                        schema_file=sys.argv[3])
    single_modification_per_column(origin, dest, mapping, pk)
