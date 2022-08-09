from tuprolog.core import Clause
from tuprolog.theory import Theory, mutable_theory
from logic.datalog import DatalogFormula, Argument, DefinitionClause, Variable, Number
from logic.datalog.grammar import Predication, Expression, Term

mapping = {
    '<=': '≤',
    '=<': '≤',
    '>=': '≥',
    '=>': '≥',
}


def prolog_to_datalog(t: Theory) -> list[DatalogFormula]:
    mutable_t = mutable_theory(t)
    return [clause_to_formula(c) for c in mutable_t.clauses]


def clause_to_formula(c: Clause) -> DatalogFormula:
    def prolog_atom_to_formula(arg) -> Term:
        if arg.is_var:
            arg = Variable(str(arg.name))
        elif arg.is_number or arg.is_real:
            arg = Number(float(arg.value))
        elif arg.is_constant:
            arg = Predication(str(arg.value))
        else:
            raise Exception('Type error: ' + str(arg) + ' cannot be converted into a datalog formula')
        return arg

    def build_args(args: list) -> Argument:
        arg = prolog_atom_to_formula(args[0])
        if len(args) > 1:
            return Argument(arg, build_args(args[1:]))
        else:
            return Argument(arg)

    def get_standard_functor(functor: str) -> str:
        return str(functor) if functor not in mapping else mapping[functor]

    def create_body(terms: list) -> Clause:
        term = terms[0]
        if len(terms) > 1:
            if term.is_struct:
                return Expression(create_body([term]), create_body(terms[1:]), '∧')
            else:
                raise Exception('Not implemented error: only expressions in clause body')
        else:
            if term.is_struct and not term.is_recursive:
                args = list(term.args)
                return Expression(prolog_atom_to_formula(args[0]),
                                  prolog_atom_to_formula(args[1]),
                                  get_standard_functor(term.functor))
            else:
                raise Exception('Not implemented error: only not recursive expressions in clause body')

    # LHS
    lhs = DefinitionClause(str(c.head.functor), build_args(list(c.head.args)))
    # RHS
    terms = list(c.body.unfolded) if c.body_size > 1 else [c.body]
    rhs = create_body(terms)
    return DatalogFormula(lhs, rhs)
