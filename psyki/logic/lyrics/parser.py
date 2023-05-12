from pyparsing import (
    ParserElement,
    Forward,
    Keyword,
    Group,
    oneOf,
    delimitedList,
    infixNotation,
    opAssoc,
    Word,
    Suppress,
    alphas,
)
from .compiler import *
from .logic import and_n, implies, forall, iff, or_n
from .world import World
from .domain import Domain


ParserElement.enablePackrat()


class Constraint(object):
    def __init__(self, formula, subdomains=None):
        self.variables = {}
        if subdomains is not None:
            for k, v in subdomains.items():
                self._create_or_get_variable(k, v)
        self.tensor = self.parse(formula)

    def _create_or_get_variable(self, id, domain):
        assert isinstance(domain, Domain)
        if id in self.variables:
            assert (
                self.variables[id].domain == domain
                or domain in self.variables[id].domain.ancestors
            )
        else:
            v = variable(domain, id)
            self.variables[id] = v
        return self.variables[id]

    def _createParseAction(self, class_name):
        def _create(tokens):
            if class_name == "Atomic":
                predicate_name = tokens[0]
                predicate = World.predicates[predicate_name]
                args = []
                for i, t in enumerate(tokens[1:]):
                    args.append(self._create_or_get_variable(t, predicate.domains[i]))
                a = atom(predicate, args)
                return a
            elif class_name == "AND":
                args = tokens[0][::2]
                return and_n(*args)
            elif class_name == "OR":
                args = tokens[0][::2]
                return or_n(*args)
            elif class_name == "IMPLIES":
                args = tokens[0][::2]
                return implies(*args)
            elif class_name == "IFF":
                args = tokens[0][::2]
                return iff(*args)
            elif class_name == "FORALL":
                return forall(self.variables[tokens[1]], tokens[2][0])
            # elif class_name == "EXISTS":
            #     return Exists(constraint, tokens, self.world)
            # elif class_name == "EXISTN":
            #     return Exists_n(constraint, tokens, self.world)
            elif class_name == "ARITHM_REL":
                # TODO
                raise NotImplementedError(
                    "Arithmetic Relations not already implemented"
                )
            # elif class_name == "FILTER":
            #      parse_and_filter(constraint, tokens)

        return _create

    def parse(self, definition):
        left_parenthesis, right_parenthesis, colon, left_square, right_square = map(
            Suppress, "():[]"
        )
        symbol = Word(alphas)

        """ TERMS """
        var = symbol
        # var.setParseAction(self._createParseAction("Variable"))

        """ FORMULAS """
        formula = Forward()
        not_ = Keyword("not")
        and_ = Keyword("and")
        or_ = Keyword("or")
        xor = Keyword("xor")
        implies = Keyword("->")
        iff = Keyword("<->")

        forall = Keyword("forall")
        exists = Keyword("exists")
        forall_expression = forall + symbol + colon + Group(formula)
        forall_expression.setParseAction(self._createParseAction("FORALL"))
        exists_expression = exists + symbol + colon + Group(formula)
        exists_expression.setParseAction(self._createParseAction("EXISTS"))

        relation = oneOf(list(World.predicates.keys()))
        atomic_formula = (
            relation + left_parenthesis + delimitedList(var) + right_parenthesis
        )
        atomic_formula.setParseAction(self._createParseAction("Atomic"))
        espression = forall_expression | exists_expression | atomic_formula
        formula << infixNotation(
            espression,
            [
                (not_, 1, opAssoc.RIGHT, self._createParseAction("NOT")),
                (and_, 2, opAssoc.LEFT, self._createParseAction("AND")),
                (or_, 2, opAssoc.LEFT, self._createParseAction("OR")),
                (xor, 2, opAssoc.LEFT, self._createParseAction("XOR")),
                (implies, 2, opAssoc.RIGHT, self._createParseAction("IMPLIES")),
                (iff, 2, opAssoc.RIGHT, self._createParseAction("IFF")),
            ],
        )

        constraint = var ^ formula
        tree = constraint.parseString(definition, parseAll=True)
        return tree[0]


def constraint(formula, subdomains=None):
    c = Constraint(formula, subdomains)
    return c.tensor
