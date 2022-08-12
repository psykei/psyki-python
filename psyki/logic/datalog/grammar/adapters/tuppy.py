from tuprolog.core import Clause, struct
from tuprolog.theory import Theory, mutable_theory
from psyki.logic.datalog import DatalogFormula, Argument, DefinitionClause, Variable, Number
from psyki.logic.datalog.grammar import Predication, Expression, Term, Boolean, Negation

mapping = {
    '<=': '≤',
    '=<': '≤',
    '>=': '≥',
    '=>': '≥',
}

in_functor = 'in'
not_in_functor = 'not_in'
special_functor = (in_functor, not_in_functor)


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
            if term.is_struct and not term.is_recursive:
                return Expression(create_body([term]), create_body(terms[1:]), '∧')
            elif term.is_struct and term.is_recursive:
                t1, t2 = split_term(term)
                if term.functor == in_functor:
                    return Expression(Expression(t1, t2, '∧'), create_body(terms[1:]), '∧')
                elif term.functor == not_in_functor:
                    return Expression(Negation(Expression(t1, t2, '∧')), create_body(terms[1:]), '∧')
                else:
                    raise Exception('Not expandable functor: ' + str(term.functor))
            else:
                raise Exception('Not implemented error: only expressions in clause body')
        else:
            if term.is_true:
                return Boolean(term.is_true)
            elif term.is_struct and term.functor not in special_functor:
                args = list(term.args)
                return Expression(prolog_atom_to_formula(args[0]),
                                  prolog_atom_to_formula(args[1]),
                                  get_standard_functor(term.functor))
            elif term.is_struct and term.functor in special_functor:
                t1, t2 = split_term(term)
                if term.functor == in_functor:
                    return Expression(t1, t2, '∧')
                elif term.functor == not_in_functor:
                    return Negation(Expression(t1, t2, '∧'))
                else:
                    raise Exception('Not expandable functor: ' + str(term.functor))
            else:
                raise Exception('Not implemented error: only not recursive expressions in clause body')

    def split_term(t):
        args = list(t.args)
        t1 = Expression(prolog_atom_to_formula(args[0]), prolog_atom_to_formula(args[1][0]), mapping['>='])
        t2 = Expression(prolog_atom_to_formula(args[0]), prolog_atom_to_formula(args[1][1][0]), '<')
        return t1, t2

    # LHS
    lhs = DefinitionClause(str(c.head.functor), build_args(list(c.head.args)))
    # RHS
    terms = list(c.body.unfolded) if c.body_size > 1 else [c.body]
    rhs = create_body(terms)
    return DatalogFormula(lhs, rhs)
