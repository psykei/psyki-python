from tuprolog.core import Clause as PrologClause
from psyki.logic import *
from tuprolog.theory import Theory as PrologTheory, mutable_theory
from tuprolog.theory.parsing import parse_theory
from psyki.logic.operators import LogicOperator, Conjunction, LogicNegation


class TuProlog(TheoryAdapter):
    """Adapter for 2ppy library: https://github.com/tuProlog/2ppy"""

    @staticmethod
    def _from_file(filename: str) -> PrologTheory:
        with open(filename, 'r', encoding="utf8") as file:
            textual_rule = file.read()
        return TuProlog._from_string(textual_rule)

    @staticmethod
    def _from_string(textual_theory: str) -> PrologTheory:
        return parse_theory(textual_theory)

    @staticmethod
    def _visit_element(elem: Any) -> Any:
        if isinstance(elem, list):
            if len(elem) > 1:
                return Expression(TuProlog._visit_element(elem[0]), TuProlog._visit_element(elem[1:]), Conjunction())
            elif len(elem) == 1:
                return TuProlog._visit_element(elem[0])
            else:
                raise Exception("Unexpected value")
        elif elem.is_truth:
            return Boolean(elem.is_true)
        elif elem.is_var:
            return Variable(str(elem.name))
        elif elem.is_number:
            return Number(str(float(elem.value)))
        elif elem.is_constant:
            return Predication(str(elem.functor))
        elif elem.is_struct:
            args: list[Any] = list(elem.args)
            operator: LogicOperator = LogicOperator.from_symbol(elem.functor)
            if operator is None:
                return Nary(str(elem.functor), TuProlog._arg_from_list(list(elem.args)))
            if operator.arity == 2:
                return Expression(TuProlog._visit_element(args[0]), TuProlog._visit_element(args[1]), operator)
            elif operator.symbol == LogicNegation.symbol:
                return Negation(TuProlog._visit_element(args))
            else:
                raise Exception()
        else:
            raise Exception("Unexpected type")

    @staticmethod
    def _arg_from_list(args: list[Any]) -> Argument:
        if len(args) > 0:
            return Argument(TuProlog._visit_element(args[0]), TuProlog._arg_from_list(args[1:]))
        else:
            return None

    @staticmethod
    def _convert_clause(clause: PrologClause) -> Formula:
        name: str = str(clause.head.functor)
        args: Argument = TuProlog._arg_from_list(list(clause.head.args))
        rhs: Clause = TuProlog._visit_element(clause.body)
        return DefinitionFormula(DefinitionClause(name, args), rhs)

    @staticmethod
    def from_legacy_theory(legacy_theory: PrologTheory) -> Theory:
        return Theory([TuProlog._convert_clause(clause) for clause in mutable_theory(legacy_theory).clauses])

    @staticmethod
    def from_file(filename: str) -> Theory:
        return TuProlog.from_legacy_theory(TuProlog._from_file(filename))

    @staticmethod
    def from_string(textual_theory: str) -> Theory:
        return TuProlog.from_legacy_theory(TuProlog._from_string(textual_theory))
