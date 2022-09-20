grammar Datalog;

formula
    : lhs=def_clause op=('<-' | '<--') rhs=clause
    ;

def_clause
    : pred=Predication '(' args=arguments ')' #DefPredicateArgs
    ;

clause
    : lit=literal # ClauseLiteral
    | '(' left=clause op='*' right=clause ')' # ClauseExpression
    | '(' left=clause op='+' right=clause ')' # ClauseExpression
    | '(' left=clause op=('=' | '<' | '=<' | '>' | '>=' | 'm') right=clause ')' # ClauseExpression
    | '(' left=clause op=(',' | ';') right=clause ')' # ClauseExpression
    | left=clause op='*' right=clause # ClauseExpressionNoPar
    | left=clause op='+' right=clause # ClauseExpressionNoPar
    | left=clause op=('=' | '<' | '=<' | '>' | '>=' | 'm') right=clause # ClauseExpressionNoPar
    | left=clause op=(',' | ';') right=clause # ClauseExpressionNoPar
    | '(' c=clause ')' #ClauseClause
    ;

literal
    : pred=predicate #LiteralPred
    | '¬' '(' pred=clause ')' #LiteralNeg
    | '¬' pred=clause #LiteralNeg
    ;

predicate
    : 'true' # PredicateTrue
    | 'false' # PredicateFalse
    | name=term # PredicateTerm
    | pred=Predication # PredicateUnary
    | pred='m_of_n' '(' m=Number ',' args=complex_arguments ')' # MofN
    | pred=Predication '(' args=arguments ')' # PredicateArgs
    ;

arguments
    : name=term # LastTerm
    | name=term ',' args=arguments # MoreArgs
    | # None
    ;

complex_arguments
    : name=clause # LastClause
    | name=clause ',' args=complex_arguments # MoreComplexArgs
    | # None2
    ;

term
    : var=Variable # TermVar
    | name=constant # TermConst
    ;

constant
    : num=Number # ConstNumber
    | boolean # ConstBool
    | name=Predication # ConstName
    ;

boolean
    : 'true'
    | 'false'
    ;

Predication: [a-z]([a-z]|[0-9]|[_])*;
Variable : [A-Z]([a-z]|[A-Z]|[0-9]|[_])*;
Number : [-]?([0-9]*[.])?[0-9]+;
WS : [ \t\n]+ -> skip ;