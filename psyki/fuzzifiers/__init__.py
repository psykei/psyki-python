from __future__ import annotations
from typing import List, Callable
from tensorflow.python.keras import Model
from psyki import logic
from psyki.logic import *
from pathlib import Path
from psyki.utils import match_case


PATH = Path(__file__).parents[0]


class Fuzzifier(ABC):
    """
    A fuzzifier transforms a theory (list of formulae) representing symbolic knowledge into an injectable object.
    Usually the output consists of layers of a neural network or cost functions.
    In other words, a fuzzifier is a visitor of theories.
    """

    name: str

    @abstractmethod
    def visit(self, rules: List[Formula]) -> Any:
        pass

    @staticmethod
    def get(name: str) -> Callable:
        from psyki.fuzzifiers.netbuilder import NetBuilder
        from psyki.fuzzifiers.lukasciewicz import Lukasiewicz
        from psyki.fuzzifiers.towell import Towell

        if name == Lukasiewicz.name:
            return lambda x: Lukasiewicz(*x)
        elif name == NetBuilder.name:
            return lambda x: NetBuilder(*x)
        elif name == Towell.name:
            return lambda x: Towell(*x)
        else:
            raise Exception('Fuzzifier ' + name + ' is not defined')

    @staticmethod
    def enriched_model(model: Model) -> Model:
        from psyki.ski import EnrichedModel
        return EnrichedModel(model, {})

    @staticmethod
    @abstractmethod
    def custom_objects() -> dict:
        pass


class DatalogFuzzifier(Fuzzifier, ABC):
    """
    A fuzzifier supporting the Datalog logic language.
    """
    """
    A variable can be grounded in two different ways:
    - the variable appears in the head of the predicate, then its value is the one provided by the caller.
      Note that the variable can be one feature of the dataset;
    - the variable appears in the head but it is not grounded for the caller, then it must be assigned in the body.
    It is necessary to keep track of all variable groundings for each rule resolution/visiting.
    """
    VariableMap = dict[Variable, Clause]
    SubMap = dict[Variable, Callable]
    """
    AssignmentType is a data type that represents the required information to apply variable assignments.
    The key of the outer dictionary is the name of the predicate.
    The value is a list of tuple (one for each definition occurrence of the predicate in the theory).
    The first value of the tuple is the body of the predicate (without variable assignments).
    The second value is a map of the assignments.
    
    Example:
        {predicate1: [(body1, {Var1: Value1, Var2: Value2, ...}), (body2, {...}, ...],
         predicate2: ...} 
    """
    AssignmentMap = dict[str, list[tuple[Clause, VariableMap]]]
    assignment_mapping: AssignmentMap = {}
    """
    Map between dataset features (can appear as variables in the head of rules) and indices of the input layer.
    """
    FeatureMap = dict[str, int]
    feature_mapping: FeatureMap = {}
    """
    Map between predicates' names and their fuzzy object plus the variables in the arguments.
    """
    PredicateCallMap = dict[str, (Callable, list[Variable])]
    predicate_call_mapping: PredicateCallMap = {}
    """
    Map between the class and the object obtained by the corresponding classification rules.
    """
    classes: dict[str, Any] = {}
    class_call: dict[str, Any] = {}

    def __init__(self):
        pass

    def visit(self, rules: List[Formula]) -> Any:
        self._clear()

    def _visit(self, formula: Formula, local_mapping: VariableMap, substitutions: SubMap) -> Any:
        cases = [
            (logic.DefinitionFormula, self._visit_formula),
            (logic.Expression, self._visit_expression),
            (logic.Negation, self._visit_negation),
            (logic.Variable, self._visit_variable),
            (logic.Boolean, self._visit_boolean),
            (logic.Number, self._visit_number),
            (logic.Unary, self._visit_unary),
            (logic.Nary, self._visit_nary),
            (type(formula), None)
        ]
        matched = match_case(type(formula), cases)
        if matched is not None:
            if type(formula) in (logic.Boolean, logic.Number, logic.Unary):
                return matched(formula)
            else:
                return matched(formula, local_mapping, substitutions)
        else:
            raise Exception('Unexpected formula')

    @abstractmethod
    def _visit_formula(self, formula: Formula, local_mapping: VariableMap, substitutions: SubMap) -> Any:
        pass

    @abstractmethod
    def _visit_definition_clause(self, lhs: Formula, rhs: Formula, local_mapping: VariableMap, substitutions: SubMap) -> Any:
        pass

    @abstractmethod
    def _visit_expression(self, formula: Formula, local_mapping: VariableMap, substitutions: SubMap) -> Any:
        pass

    @abstractmethod
    def _visit_negation(self, formula: Formula, local_mapping: VariableMap, substitutions: SubMap) -> Any:
        pass

    @abstractmethod
    def _visit_variable(self, formula: Formula, local_mapping: VariableMap, substitutions: SubMap) -> Any:
        pass

    @abstractmethod
    def _visit_boolean(self, formula: Formula) -> Any:
        pass

    @abstractmethod
    def _visit_number(self, formula: Formula) -> Any:
        pass

    @abstractmethod
    def _visit_unary(self, formula: Formula) -> Any:
        pass

    @abstractmethod
    def _assign_variables(self, mappings: list[tuple[Any, VariableMap]], local_mapping: VariableMap, substitutions: SubMap) -> Any:
        pass

    def _visit_nary(self, formula: Formula, local_mapping: VariableMap, substitutions: SubMap):
        assert isinstance(formula, Nary)
        # Check if all variables in the predicate are bounded.
        # If positive then just evaluate the predicate.
        # If negative there is at least one variable to assign.
        arguments: list[Term] = formula.args.unfolded
        keys = local_mapping.keys()
        subs = substitutions.keys()
        all_grounded = all([arg in keys or arg in subs for arg in arguments if isinstance(arg, Variable)])  # or arg in subs
        # Check if the predicate has been defined
        predicate, local_variables = None, {}
        if formula.predicate in self.predicate_call_mapping.keys():
            predicate, local_variables = self.predicate_call_mapping[formula.predicate]
        # Check if you are using a classification/regression rule
        elif not isinstance(formula.args.last, Variable):
            output_value = str(formula.args.last)
            if output_value in self.class_call.keys():
                # Here local variables and substitutions are ignored.
                # Only the class/value overall evaluation of the rule is considered.
                return self.class_call[output_value]
            else:
                raise Exception("Rule " + formula.predicate + "for class/regression " + output_value + " not found")
        # Update mapping with constants
        tmp_mapping: dict[Term, Term] = {k: v for k, v in zip(arguments, local_variables)}
        for k, v in tmp_mapping.items():
            if isinstance(k, Constant) and v not in local_mapping:
                assert isinstance(v, Variable)
                local_mapping[v] = k
        if all_grounded:
            # Simple logic evaluation.
            new_local_mapping = {k: v for k, v in zip(local_variables, arguments)}
            local_mapping.update(new_local_mapping)
            return predicate(local_mapping)(substitutions)
        else:
            # Variables assignment.
            predicate_bodies: list[tuple[Clause, dict[Variable, Clause]]] = self.assignment_mapping[formula.predicate]
            grounded = [arg for arg in arguments if isinstance(arg, Variable) and arg in keys]
            not_grounded = [arg for arg in arguments if isinstance(arg, Variable) and arg not in grounded]
            result: list[tuple[Clause, dict[Variable, Clause]]] = []
            new_mapping = {}
            for body, mapping in predicate_bodies:
                old_keys = list(mapping.keys())
                new_mapping = {old_keys[arguments.index(v)]: v for v in grounded}
                new_mapping.update({old_keys[arguments.index(v)]: v for v in not_grounded})
                new_body: Clause = body.remove_variable_assignment([k for k, v in new_mapping.items() if v in not_grounded])
                subs_mapping = {v: body.get_substitution(k) for k, v in new_mapping.items() if v in not_grounded}
                result.append((new_body, subs_mapping))
            return self._assign_variables(result, new_mapping, substitutions)

    @abstractmethod
    def _clear(self):
        pass


class ConstrainingFuzzifier(DatalogFuzzifier, ABC):
    """
    A fuzzifier that encodes logic formulae into continuous functions (or something equivalent).
    It constrains the behaviour of the predictor during training (it is penalised when it violates the knowledge).
    """

    def visit(self, rules: List[Formula]) -> Any:
        super().visit(rules)
        for rule in rules:
            self._visit(rule, {}, {})
        return self.classes


class StructuringFuzzifier(DatalogFuzzifier, ABC):
    """
    A fuzzifier that encodes logic formulae into new sub parts of the predictors which mimic the logic formulae.
    """

    def visit(self, rules: List[Formula]) -> Any:
        super().visit(rules)
        for rule in rules:
            new_rule = rule.copy()
            new_rule.optimize()
            self._visit(new_rule, {}, {})
        return list(self.classes.values())
