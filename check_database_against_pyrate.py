"""
The critical validation before using the new group theory module is to make sure it produces the same results as
the old one. For that I load the old one collect all the irreps and compare with the ones calculated by pyrate.
"""

import pickle
import pandas as pd
import numpy as np
from PyLie import *
import pudb
import sys
import copy

# 1 load the pickle file with the data base
# db = pickle.load(file('/Users/florian/Documents/work/Projects/Pyrate/git/pyrate/Source/GroupTheory/CGCs-1.2.1-sparse.pickle','r'))
db = pickle.load(file('db-su2.pickle', 'r'))

GroupToCheck = ['SU2']
AttributeToCheck = ['Quartic', 'Trilinear', 'Quartic']  # , 'Casimir', 'Dynkin']
res = pd.DataFrame({'irreps': [], 'match': [], 'error': [], 'attribute': [], 'group': [], 'res_pylie': [], 'res_db': [],
                    'sign': [], 'key_match': [], 'AfterRenorm': []})

for gg in GroupToCheck:
    if gg in db:
        # Declare a LieAlgebra object
        lie = LieAlgebra(CartanMatrix("SU", 2))
        for attribute in AttributeToCheck:
            # collect all the keys
            if attribute in AttributeToCheck[:3]:
                for irreps, val in db[gg][attribute].items():
                    # calculate the corresponding invariant using PyLie
                    try:
                        if irreps ==((1,), (3, True), (1, True), (1, True)):
                            pudb.set_trace()
                        invs = lie.invariants([[ell for ell in el if not type(ell) == bool] for el in irreps],
                                              conj=[el[-1] if type(el[-1]) == bool else False for el in irreps],
                                              returnTensor=True, pyrate_normalization=True)
                        # compare with the values in the database
                        validate = []
                        invs_set = []
                        signs = []
                        for iel, el1 in enumerate(invs):
                            invs_set.append(set(el1))
                            if type(val[0]) == list:
                                list_check = []
                                for ii in range(len(val)):
                                    list_check.append(set(val[ii]).difference(invs_set[-1]) == set([]))
                                check = any(list_check)
                            else:
                                check = set(val).difference(invs_set[-1]) == set([])
                            if not check:
                                invs_set.pop(-1)
                                # make sure it is not just a minus sign difference
                                el1 = [tuple(list(ell)[:-1] + [-1 * ell[-1]]) for ell in el1]
                                invs_set.append(el1)
                                signs.append(iel)
                                # try the subtraction again
                                if type(val[0]) == list:
                                    list_check = []
                                    for ii in range(len(val)):
                                        list_check.append(set(val[ii]).difference(invs_set[-1]) == set([]))
                                    check = any(list_check)

                                else:
                                    check = set(val).difference(invs_set[-1]) == set([])
                            validate.append(check)
                        if not (all(validate)):
                            if type(val[0]) == list:
                                to_store = [set(ell) for ell in val]
                            else:
                                to_store = val
                            res = res.append(
                                {'irreps': irreps, 'match': all(validate), 'attribute': attribute, 'group': gg,
                                 'res_pylie': invs_set, 'res_db': to_store, 'sign': signs},
                                ignore_index=True)
                        else:
                            res = res.append(
                                {'irreps': irreps, 'match': all(validate), 'attribute': attribute, 'group': gg,
                                 'sign': signs},
                                ignore_index=True)
                    except:
                        pudb.set_trace()
                        res = res.append({'irreps': irreps, 'match': False, 'error': sys.exc_info()[1][0],
                                          'attribute': attribute, 'group': gg},
                                         ignore_index=True)
# collect all the keys
non_match = res[res['match'] == 0]
for el in non_match.index:
    py = non_match.loc[el]['res_pylie']
    db = non_match.loc[el]['res_db']
    assert len(py) == 1
    # construct dictionary
    outpy, outdb = {}, {}
    for ell in py[0]:
        outpy[ell[:-1]] = ell[-1]
    for ell in db:
        outdb[ell[:-1]] = ell[-1]
    # check that the keys are the same
    if not (set(outpy.keys()).difference(set(outdb.keys())) == set([])):
        res.loc[el, 'key_match'] = False
    else:
        res.loc[el, 'key_match'] = True
        # If that's the case renormalize then according to any key
        key, val = outdb.items()[0]
        ratio = outpy[key] / val
        for kk, vv in outpy.items():
            # renormalize
            outpy[kk] = vv / ratio
        # now check again if they are the same
        check = True
        for kk, vv in outdb.items():
            if not vv == outpy[kk]:
                check = False
                beak
        if check:
            res.loc[el, 'AfterRenorm'] = True
        else:
            res.loc[el, 'AfterRenorm'] = False

# They are all the same if the following group only has two entries
group = res.groupby(['key_match', 'match', 'AfterRenorm'])
# indeed we are looking at 'match'==0.0 and then key_match == True and AfterRenorm ==True
if len(group.groups.keys()) == 2 and ((True, 0.0, True) in group.groups.keys()) and not (
    (True, 0.0, False) in group.groups.keys()):
    print("They are all the same up to normalization")
pudb.set_trace()
