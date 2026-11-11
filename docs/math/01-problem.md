# 1. The problem

A Deepwoken build is a set of stats, an oath and murmur, and a set of talents and mantras. Players share builds through planners such as the [Deepwoken builder](https://deepwoken-builder.onrender.com/). tri is given many such builds (the corpus, supplied by the user's own data pipeline) and a small number of reviews written by expert players, and it must:

1. **analyse** a build: say which picks belong, which do not, and why;
2. **complete** a partial build: fill the remaining slots in a sensible order;
3. **learn from reviews**: when an expert says "remove this", apply that judgement to this build and to builds like it, and to no others.

The corpus says what players do. Reviews say what is right. The model fitted on the corpus is only a **baseline**: it covers builds and items no review has spoken about, and it gives way whenever a review speaks. Reviews are taken to be fully accurate, and when two reviews disagree, the newer one wins.

The difficulty is the word "like". A review of one build must reach builds with the same purpose, which tri calls an **archetype**, and must not reach builds that only share some picks. So the core of tri is a space in which **coordinates and proximity encode a build's archetype**, built without labels and in a way that cannot overfit to them.
