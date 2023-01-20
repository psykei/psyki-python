from psyki.logic import Formula, DefinitionFormula


def set_trainable_rules(trainable_rules: list[str], rules: list[Formula]) -> list[Formula]:
    for rule in rules:
        assert (isinstance(rule, DefinitionFormula))
        if rule.lhs.predication in trainable_rules:
            rule.trainable = True
    return rules
