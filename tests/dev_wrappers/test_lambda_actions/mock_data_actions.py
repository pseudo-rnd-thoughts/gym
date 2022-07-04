from gym.spaces import Box, Dict, Discrete, Tuple

SEED = 1
NUM_ENVS = 3

DISCRETE_VALUE = 1
DISCRETE_ACTION = 0

BOX_LOW, BOX_HIGH, BOX_DIM = -5, 5, 1
NEW_BOX_LOW, NEW_BOX_HIGH = 0, 2

NESTED_BOX_LOW, NESTED_BOX_HIGH = 0, 10
NEW_NESTED_BOX_LOW, NEW_NESTED_BOX_HIGH = 0, 5


TESTING_BOX_ACTION_SPACE = Box(BOX_LOW, BOX_HIGH, (BOX_DIM,))

TESTING_DICT_ACTION_SPACE = Dict(
    discrete=Discrete(DISCRETE_VALUE),
    box=Box(BOX_LOW, BOX_HIGH, (BOX_DIM,)),
)

TESTING_NESTED_DICT_ACTION_SPACE = Dict(
    discrete=Discrete(DISCRETE_VALUE),
    box=Box(BOX_LOW, BOX_HIGH, (BOX_DIM,)),
    nested=Dict(nested=Box(NESTED_BOX_LOW, NESTED_BOX_HIGH, (BOX_DIM,))),
)

TESTING_DOUBLY_NESTED_DICT_ACTION_SPACE = Dict(
    discrete=Discrete(DISCRETE_VALUE),
    box=Box(BOX_LOW, BOX_HIGH, (BOX_DIM,)),
    nested=Dict(nested=Dict(nested=Box(NESTED_BOX_LOW, NESTED_BOX_HIGH, (BOX_DIM,)))),
)

TESTING_TUPLE_ACTION_SPACE = Tuple(
    [Discrete(DISCRETE_VALUE), Box(BOX_LOW, BOX_HIGH, (BOX_DIM,))]
)


TESTING_NESTED_TUPLE_ACTION_SPACE = Tuple(
    [
        Box(BOX_LOW, BOX_HIGH, (BOX_DIM,)),
        Tuple(
            [
                Discrete(DISCRETE_VALUE),
                Box(NESTED_BOX_LOW, NESTED_BOX_HIGH, (BOX_DIM,)),
            ]
        ),
    ]
)


TESTING_DOUBLY_NESTED_TUPLE_ACTION_SPACE = Tuple(
    [
        Box(BOX_LOW, BOX_HIGH, (BOX_DIM,)),
        Tuple(
            [
                Discrete(DISCRETE_VALUE),
                Tuple(
                    [
                        Discrete(DISCRETE_VALUE),
                        Box(NESTED_BOX_LOW, NESTED_BOX_HIGH, (BOX_DIM,)),
                    ]
                ),
            ]
        ),
    ]
)

TESTING_TUPLE_WITHIN_DICT_ACTION_SPACE = Dict(
    discrete=Discrete(DISCRETE_VALUE), tuple=Tuple([Box(BOX_LOW, BOX_HIGH, (BOX_DIM,))])
)

TESTING_DICT_WITHIN_TUPLE_ACTION_SPACE = Tuple(
    [Discrete(DISCRETE_VALUE), Dict(dict=Box(BOX_LOW, BOX_HIGH, (BOX_DIM,)))]
)
