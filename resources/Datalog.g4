grammar Datalog;

formula
    : lhs=def_clause '←' rhs=clause
    ;

def_clause
    : pred=Predication '(' args=arguments ')' #DefPredicateArgs
    ;

clause
    : '(' left=clause op='*' right=clause ')' # ClauseExpression
    | '(' left=clause op='+' right=clause ')' # ClauseExpression
    | '(' left=clause op=('=' | '<' | '≤' | '>' | '≥' | 'm') right=clause ')' # ClauseExpression
    | '(' left=clause op=('∧' | '∨') right=clause ')' # ClauseExpression
    | '(' left=clause op=('→' | '↔') right=clause ')' # ClauseExpression
    | left=clause op='*' right=clause # ClauseExpressionNoPar
    | left=clause op='+' right=clause # ClauseExpressionNoPar
    | left=clause op=('=' | '<' | '≤' | '>' | '≥' | 'm') right=clause # ClauseExpressionNoPar
    | left=clause op=('∧' | '∨') right=clause # ClauseExpressionNoPar
    | left=clause op=('→' | '↔') right=clause # ClauseExpressionNoPar
    | literal # ClauseLiteral
    ;

literal
    : predicate #LiteralPred
    | '¬' '(' pred=clause ')' #LiteralNeg
    | '¬' pred=clause #LiteralNeg
    ;

predicate
    : '⊤' # PredicateTrue
    | '⊥' # PredicateFalse
    | term # PredicateTerm
    | pred=Predication # PredicateUnary
    | pred=Predication '(' args=arguments ')' #PredicateArgs
    ;

arguments
    : last=term #LastTerm
    | term ',' args=arguments #MoreArgs
    ;

term
    : var=Variable # TermVar
    | constant # TermConst
    ;

constant
    : num=Number # ConstNumber
    | boolean # ConstBool
    | name=Predication # ConstName
    ;

boolean
    : '⊤'
    | '⊥'
    ;

Predication: [a-z]([a-z]|[0-9]|[_])*;
Variable : [A-Z]([a-z]|[A-Z]|[0-9])*;
Number : [-]?([0-9]*[.])?[0-9]+;
WS : [ \t\n]+ -> skip ;