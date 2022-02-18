grammar Prolog;

formula
    : Predication '(' args=arguments ')' ':-' right=clause
    ;

clause
    : literal # ClauseLiteral
    | '(' left=clause op='*' right=clause ')' # ClauseExpression
    | '(' left=clause op='+' right=clause ')' # ClauseExpression
    | '(' left=clause op=('=' | '<' | '≤' | '>' | '≥') right=clause ')' # ClauseExpression
    | '(' left=clause op=('∧' | '∨') right=clause ')' # ClauseExpression
    | '(' left=clause op=('→' | '↔') right=clause ')' # ClauseExpression
    | left=clause op='*' right=clause # ClauseExpressionNoPar
    | left=clause op='+' right=clause # ClauseExpressionNoPar
    | left=clause op=('=' | '<' | '≤' | '>' | '≥') right=clause # ClauseExpressionNoPar
    | left=clause op=('∧' | '∨') right=clause # ClauseExpressionNoPar
    | left=clause op=('→' | '↔') right=clause # ClauseExpressionNoPar
    ;

literal
    : predicate
    | '¬' '(' predicate ')'
    ;

predicate
    : '⊤' # PredicateTrue
    | '⊥' # PredicateFalse
    | term # PredicateTerm
    | Predication # PredicateUnary
    | Predication '(' arguments ')' #PredicateArgs
    ;

arguments
    : term
    | term ',' arguments
    ;

term
    : var=Variable # TermVar
    | Functor '(' arguments ')' # TermStruct
    | constant # TermConst
    ;

constant
    : fun=Functor # ConstFunctor
    | num=Number # ConstNumber
    | boolean # ConstBool
    ;

boolean
    : '⊤'
    | '⊥'
    ;

Functor: [_]([a-z]|[0-9])*;
Predication: [a-z]([a-z]|[0-9])*;
Variable : [A-Z]([a-z]|[A-Z]|[0-9])*;
Number : [-]?([0-9]*[.])?[0-9]+;
WS : [ \t\n]+ -> skip ;