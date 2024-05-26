import argparse
def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',default='C:\\Govno', type=str,help='Directory path of data')
    parser.add_argument('--dataset', default='cats', type=str,help='cats/imagenette/cub200')
    parser.add_argument('--arch', default='vit', type=str,help='Network architecture')
    parser.add_argument('--batch_size', default=16, type=int,help='batch size is, definitely, batch size')
    parser.add_argument('--lr', default=3e-4, type=float,help='learning rate')
    parser.add_argument('--epochs', default=100, type=int,help='Number of epochs')

    parser.add_argument('--drop_channels', default=False,type=bool, help='Chromatic abberation estimation')
    parser.add_argument('--use_grid', default=False, type=bool, help='MaskGrid usage')
    parser.add_argument('--mode', default='train', type=str,help='train/validation/test')
    parser.add_argument('--experiment_name', default='vit_cats',type=str, help='Current experimentname')
    # -------------------------------------------------------------------------------------------
    parser.add_argument('--enable_tb', default=True, type=bool,help='Enable tensorboard logging')
    parser.add_argument('--tb_dir', default='runs', type=str,help='Directory for TensorBoard logs')
    parser.add_argument('--save_every', default=25, type=int, help='Save a model every N epochs')
    parser.add_argument('--continue_training', default=0, type=int, help='if non zero, then training is continue')
    parser.add_argument('--weights_dir', default='./weights', type=str, help='Directory path of weights')
    parser.add_argument('--results_dir', default='./results',type=str, help='Directory path for saving result images')
    parser.add_argument('--model_path', default='', type=str,help='Directory path to load desirable weights')
    parser.add_argument('--crops_path', default='', type=str,help='Path to the dictionary file with coordinates for validation')
    return parser.parse_args()
