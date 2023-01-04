minimum(X, Y, Z) :- X =< Y, Z is X.
minimum(X, Y, Z) :- Y < X, Z is Y.

greater(X, Y, no) :- minimum(X, Y, Z), X = Z.
greater(X, Y, yes) :- minimum(X, Y, Z), Y = Z.