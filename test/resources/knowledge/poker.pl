pair(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R2.
pair(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R3.
pair(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R4.
pair(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R5.
pair(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R2 = R3.
pair(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R2 = R4.
pair(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R2 = R5.
pair(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R3 = R4.
pair(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R3 = R5.
pair(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R4 = R5.

two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R2, R3 = R4.
two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R3, R2 = R4.
two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R4, R2 = R3.
two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R2, R3 = R5.
two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R3, R2 = R5.
two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R5, R2 = R3.
two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R2, R4 = R5.
two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R4, R2 = R5.
two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R5, R2 = R4.
two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R3, R4 = R5.
two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R4, R3 = R5.
two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R5, R3 = R4.
two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R2 = R3, R4 = R5.
two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R2 = R4, R3 = R5.
two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R2 = R5, R3 = R4.

three(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R2, R1 = R3.
three(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R2, R1 = R4.
three(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R2, R1 = R5.
three(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R3, R1 = R4.
three(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R3, R1 = R5.
three(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R4, R1 = R5.
three(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R2 = R3, R2 = R4.
three(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R2 = R3, R2 = R5.
three(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R2 = R4, R2 = R5.
three(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R3 = R4, R3 = R5.

min5(X1,X2,X3,X4,X5,R) :- X1 =< X2, X1 =< X3, X1 =< X4, X1 =< X5, R is X1.
min5(X1,X2,X3,X4,X5,R) :- X2 =< X1, X2 =< X3, X2 =< X4, X2 =< X5, R is X2.
min5(X1,X2,X3,X4,X5,R) :- X3 =< X1, X3 =< X2, X3 =< X4, X3 =< X5, R is X3.
min5(X1,X2,X3,X4,X5,R) :- X4 =< X1, X4 =< X2, X4 =< X3, X4 =< X5, R is X4.
min5(X1,X2,X3,X4,X5,R) :- X5 =< X1, X5 =< X2, X5 =< X3, X5 =< X4, R is X5.

straight(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- min5(R1,R2,R3,R4,R5,X), (R1 + R2 + R3 + R4 + R5) = ((5 * X) + 10), \+ pair(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5).

royal(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- min5(R1,R2,R3,R4,R5,X), X = 1, ((R1 + R2 + R3 + R4 + R5) = 47), \+ pair(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5).

flush(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- S1 = S2, S1 = S3, S1 = S4, S1 = S5.

four(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R2, R1 = R3, R1 = R4.
four(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R2, R1 = R3, R1 = R5.
four(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R2, R1 = R4, R1 = R5.
four(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R1 = R3, R1 = R4, R1 = R5.
four(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5) :- R2 = R3, R2 = R4, R2 = R5.

class(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5,pair) :- pair(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5).
class(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5,two) :- two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5).
class(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5,three) :- three(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5).
class(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5,straight) :- straight(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5).
class(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5,straight) :- royal(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5).
class(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5,flush) :- flush(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5).
class(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5,full) :- three(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5), two(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5), \+ four(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5).
class(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5,four) :- four(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5).
class(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5,straight_flush) :- straight(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5), flush(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5).
class(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5,straight_flush) :- royal(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5), flush(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5).
class(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5,royal_flush) :- royal(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5), flush(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5).
class(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5,nothing) :- \+ pair(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5), \+ flush(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5), \+ straight(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5), \+ royal(S1,R1,S2,R2,S3,R3,S4,R4,S5,R5).