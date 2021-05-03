from functools import reduce
import numpy as np
import csv

def prod(x):
    return reduce(lambda a, b: a*b, x)

class Bayesian_Fraud_Engine:
    # Initialise the engine from a csv file 
    def __init__(self, filename):
        self.rule_set = set()
        self.num_fraud_triggers = {}
        self.num_genuine_triggers = {}
        self.num_tot = 0
        self.num_fraud = 0
        self.num_genuine = 0
        self.load_history_from_file(filename)

    # Load the transaction history data from a csv file 
    # format
    # totals;n_tot;n_fraud (first line)
    # rule;num triggered by fraud;num triggered by genuine (afterwards)
    def load_history_from_file(self, filename):
        with open(filename) as data_file:
            data_reader = csv.reader(data_file, delimiter=';')
            nf = 0
            ng = 0
            first_line = next(data_reader)
            if not first_line[0] == 'totals':
                raise ValueError("First line must begin with totals")
            else:
                try:
                    self.num_tot = int(first_line[1])
                except: 
                    raise ValueError("Total number is not a valid int")
                try:
                    self.num_fraud = int(first_line[2])
                except:
                    raise ValueError("Total number of frauds is not a valid int")
                self.num_genuine = self.num_tot - self.num_fraud
            for row in data_reader:
                rule_name = row[0]
                try:
                    nf = int(row[1])
                except:
                    raise ValueError("Number of fraud triggers for " + rule_name + " not a valid int")
                try:
                    ng = int(row[2])
                except:
                    raise ValueError("Number of genuine triggers for " + rule_name + " not a valid int")
                self.rule_set.add(rule_name)
                self.num_fraud_triggers[rule_name] = nf
                self.num_genuine_triggers[rule_name] = ng

    # Marginalised probability
    # See theory documentation in README.md
    def marginal_probability(self, n,m):
        return (m+1)/(n+2)

    # Calculate P(f|H), probability of a fraud given no rule information
    # Marginalised over possible rates of fraud 
    def prior(self):
        return self.marginal_probability(self.num_tot, self.num_fraud)

    # For independent rules, likelihood is produce of individual 
    # likelihoods for each rule r in S
    def likelihood_general(self, rules_triggered, n, m_by_rule):
        rules_not_triggered = self.rule_set - rules_triggered
        nums_triggered = np.array([m_by_rule[r] for r in rules_triggered])
        nums_untriggered  = np.array([m_by_rule[r] for r in rules_not_triggered])
        triggered_probs = 1
        untriggered_probs = 1
        if len(nums_triggered) > 0:
            triggered_probs =  prod(self.marginal_probability(n, nums_triggered))
        if len(nums_untriggered) > 0:
            untriggered_probs = prod(1 - self.marginal_probability(n, nums_untriggered))
        return triggered_probs * untriggered_probs

    # Calculate P(S|f,H), probability of rule set being triggered given fraud & history
    def likelihood_fraud(self, rules_triggered):
        return self.likelihood_general(rules_triggered, self.num_fraud, self.num_fraud_triggers)

    # Calculate P(S|~f,H), probability of a rule set being triggered given real & history
    # Independent rules => product of individual likelihoods
    def likelihood_genuine(self, rules_triggered):
        return self.likelihood_general(rules_triggered, self.num_genuine, self.num_genuine_triggers)

    # Calculate P(S|f,H), probability of rule set being triggered given fraud & history
    def likelihood(self, rules_triggered):
        return self.likelihood_fraud(rules_triggered)

    # Calculate P(S|H) = P(S|f,H)P(f|H) + P(S|~f,H)P(~f|H)
    def evidence(self, rules_triggered):
        return self.likelihood_fraud(rules_triggered) * self.prior() + self.likelihood_genuine(rules_triggered) * (1 - self.prior())
    
    # Calculate P(f|S,H), posterior probability that the transaction is a fraud 
    # given the set of rules triggered S 
    # Historic transaction data H is loaded into the engine at initialisation
    # @param rules_triggered: a python set containing the list of triggered rules
    def posterior(self, rules_triggered):
        if not rules_triggered.issubset(self.rule_set):
            raise ValueError("Rules given are not a subset of the rule set of this engine.")
        else:
            return (self.likelihood(rules_triggered) * self.prior() / self.evidence(rules_triggered))

