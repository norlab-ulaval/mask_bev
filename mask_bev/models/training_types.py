class OptimizerType:
    ADAM = 'adam'
    LAMB = 'lamb'
    ADAM_W = 'adam_w'
    SGD = 'sgd'


# TODO add poly and step lr schedulers
class LrSchedulerType:
    STEP = 'step'
    REDUCE_ON_PLATEAU = 'plateau'
    COSINE = 'cosine'
    POLY = 'poly'
