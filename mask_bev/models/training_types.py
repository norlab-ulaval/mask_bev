class OptimizerType:
    ADAM = 'adam'
    LAMB = 'lamb'
    ADAM_W = 'adam_w'
    SGD = 'sgd'


class LrSchedulerType:
    REDUCE_ON_PLATEAU = 'plateau'
    COSINE = 'cosine'
