grammar FOL;

formula
    : clause
    | quantifier formula
    ;

quantifier
    : '∀' Variable
    | '∃' Variable
    ;

clause
    : literal # ClauseLiteral
    | '(' left=formula op=connective right=formula ')' # ClauseExpression
    ;

connective
    : '∧'
    | '∨'
    | '→'
    | '↔'
    | '='
    | '<'
    | '≤'
    | '>'
    | '≥'
    ;

literal
    : predicate
    | '¬' predicate
    ;

predicate
    : '⊤' # PredicateTrue
    | '⊥' # PredicateFalse
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
    : Functor
    | Number
    | boolean
    ;

boolean
    : '⊤'
    | '⊥'
    ;

Functor: [_]([a-z]|[0-9])*;
Predication: [a-z]([a-z]|[0-9])*;
Variable : [A-Z]([a-z]|[A-Z]|[0-9])*;
Number : [+-]?([0-9]*[.])?[0-9]+;
WS : [ \t]+ -> skip ;