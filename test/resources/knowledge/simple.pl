minimum(X1, X2, X3) :- X1 =< X2, X3 is X1.
minimum(X1, X2, X3) :- X2 < X1, X3 is X2.

greater(X, Y, no) :- minimum(X, Y, Z), X = Z.
greater(X, Y, yes) :- minimum(X, Y, Z), Y = Z.