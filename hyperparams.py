from constants import MODEL_INPUT_SIZE

NUM_RES_BLOCKS = 9
INPUT_SHAPE = f"3,{MODEL_INPUT_SIZE},{MODEL_INPUT_SIZE}"
N_EPOCHS = 200
EPOCH_DECAY = 100
LEARNING_RATE = 0.0002
START_EPOCH = 0
B1 = 0.5   # adam: decay of first order momentum of gradient
B2 = 0.999 # adam: decay of second order momentum of gradient

LAMBDA_CYCLE_LOSS = 10
LAMBDA_IDENTITY_LOSS = 5

BATCH_SIZE = 4