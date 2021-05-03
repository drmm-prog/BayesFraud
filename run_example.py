import sys
from Fraud_Detection import Bayesian_Fraud_Engine as BFE

def powerset(S):
    if len(S) == 0:
        return set([])
    else:
        L = [[s] for s in S]
        return (merge_lists([[]], L))

def merge_lists(A, B):
    if len(B) == 0:
        return A
    else:
        b = B[0]
        C = []
        for a in A:
            C.append(a+b)
        return merge_lists(A+C, B[1:])

if not (len(sys.argv) == 2):
    print("Please run the script with one argument as follows:")
    print("python3 testfile.py <transaction_data>")
else:
    engine = BFE.Bayesian_Fraud_Engine(sys.argv[1])
    sets = powerset(engine.rule_set)
    for s in sets:
        print(s, engine.posterior(set(s)))
