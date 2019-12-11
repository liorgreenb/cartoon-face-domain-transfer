import torch
from pytorch_lightning import Trainer
from test_tube import HyperOptArgumentParser

from hyperparams import BATCH_SIZE, NUM_RES_BLOCKS, INPUT_SHAPE, N_EPOCHS, EPOCH_DECAY, LEARNING_RATE, START_EPOCH, B1, \
    B2, LAMBDA_CYCLE_LOSS, LAMBDA_IDENTITY_LOSS
from model.model import Model


def main(hyperparams):
    """

    :type hyperparams: object
    """
    model = Model(hyperparams)

    trainer = Trainer(
        max_nb_epochs=1 if hyperparams.debug else hyperparams.n_epochs,
        gpus=1 if torch.cuda.is_available() and hyperparams.gpu else 0,
        train_percent_check=0.01 if hyperparams.debug else 1.0,
    )

    trainer.fit(model)


if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='random_search')

    parser.add_argument('--debug', default=False, type=bool, help="")
    parser.add_argument('--gpu', default=True, type=bool, help="")

    parser.add_argument('--num_res_blocks', default=NUM_RES_BLOCKS, type=int, help="")
    parser.add_argument('--input_shape', default=INPUT_SHAPE, type=tuple, help="")
    parser.add_argument('--n_epochs', default=N_EPOCHS, type=int, help="")
    parser.add_argument('--epoch_decay', default=EPOCH_DECAY, type=int, help="")
    parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float, help="")
    parser.add_argument('--start_epoch', default=START_EPOCH, type=int, help="")
    parser.add_argument('--b1', default=B1, type=float, help="")
    parser.add_argument('--b2', default=B2, type=float, help="")
    parser.add_argument('--lambda_cycle_loss', default=LAMBDA_CYCLE_LOSS, type=float, help="")
    parser.add_argument('--lambda_identity_loss', default=LAMBDA_IDENTITY_LOSS, type=float, help="")
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help="")

    hyperparams = parser.parse_args()

    main(hyperparams)
