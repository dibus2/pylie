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
db = pickle.load(file('/Users/florian/Documents/work/Projects/Pyrate/git/pyrate/Source/GroupTheory/CGCs-1.2.1-sparse.pickle','r'))
#db = pickle.load(file('db-su2.pickle', 'r'))

GroupToCheck = ['SU3']#, 'SU3', 'SU4', 'SU5']
AttributeToCheck = ['Bilinear', 'Trilinear', 'Quartic']  # , 'Casimir', 'Dynkin']
res = pd.DataFrame({'irreps': [], 'match': [], 'error': [], 'attribute': [], 'group': [], 'res_pylie': [], 'res_db': [],
                    'sign': [], 'key_match': [], 'AfterRenorm': [], 'BeforeRenorm': []})

for gg in GroupToCheck:
    if gg in db:
        # Declare a LieAlgebra object
        lie = LieAlgebra(CartanMatrix("SU", int(gg[-1])))
        for attribute in AttributeToCheck:
            # collect all the keys
            if attribute in AttributeToCheck[:3]:
                for irreps, val in db[gg][attribute].items():
                    # calculate the corresponding invariant using PyLie
                    try:
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
                            to_store = copy.deepcopy(val)
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
res.to_hdf('comparison_first_step_su3.h5', 'before checking the key and normalization')
for el in non_match.index:
    py = non_match.loc[el]['res_pylie']
    db = non_match.loc[el]['res_db']
    # construct dictionary
    if type(db[0]) != list:
        db = [db]
    py = [{ell[:-1]: ell[-1] for ell in elem} for elem in py]
    db = [{ell[:-1]: ell[-1] for ell in elem} for elem in db]
    # check that the keys are the same
    check_keys = []
    check_renorm = []
    check_norenorm = []
    for ipp, pp in enumerate(py):
        for idd, dd in enumerate(db):
            if set(pp.keys()).difference(set(dd.keys())) == set([]):
                check_keys.append(True)
                # First check without normalization
                check = True
                for kk, vv in pp.items():
                    if not vv == dd[kk]:
                        check_norenorm.append(False)
                        check = False
                        break
                if check:
                    check_norenorm.append(True)
                    break
                key, val = pp.items()[0]
                ratio = dd[key] / val
                check = True
                for kk, vv in pp.items():
                    # renormalize
                    pp[kk] = (vv * ratio).expand()
                    if not pp[kk] == dd[kk]:
                        check_renorm.append(False)
                        check = False
                        break
                if check:
                    check_renorm.append(True)
                    break
    res.loc[el, 'key_match'] = cp.deepcopy(all(check_keys))
    res.loc[el, 'AfterRenorm'] = cp.deepcopy(all(check_renorm))
    res.loc[el, 'BeforeRenorm'] = cp.deepcopy(all(check_norenorm))
# They are all the same if the following group only has two entries
group = res.groupby(['key_match', 'match', 'AfterRenorm', 'BeforeRenorm'])
res.to_hdf('res_comparison_su3.h5', 'comparison_of_databases')
