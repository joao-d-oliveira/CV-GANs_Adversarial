import os
import time

import dotenv
import torch
import wandb
from torch import nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = ROOT_DIR + "/logs"
DEFAULT_DATA_FOLDER = ROOT_DIR + "/data/"
DEFAULT_ENV_FILE = ROOT_DIR + "/config/tmp.env"
DEFAULT_N_EPOCHS = 200
DEFAULT_BATCH_SIZE = 128
DEFAULT_RUN_NAME = 'experiment'
DEFAULT_NOISE_DIMENSION = 64
DEFAULT_LR = 0.00001
DEFAULT_DISPLAY_STEP = 500


class Config:
    """ Config class to handle all the parameters and environment variables

    Parameters
    ----------
    env_file : str
        Path to the environment file
    kwargs : dict
        Dictionary containing the arguments passed to the script

    Attributes
    ----------
    env_file : str
        Path to the environment file
    verbose : bool
        If the program should print extra information

    """
    def __init__(self, env_file=DEFAULT_ENV_FILE, kwargs=None):
        """ Config class to handle all the parameters and environment variables

        Parameters
        ----------
        env_file : str
            Path to the environment file
        kwargs : dict
            Dictionary containing the arguments passed to the script
        """

        if kwargs is None:
            kwargs = {}

        # Environment and Logger variables
        self.env_file = env_file
        self.load_env()

        self.verbose = kwargs.get('verbose', False) or os.environ.get('VERBOSE', False)
        if isinstance(self.verbose, str):
            self.verbose = self.verbose.lower() == 'true'

        self.batch_size = kwargs.get('batch_size', False) or os.environ.get('BATCH_SIZE', False)
        self.batch_size = int(self.batch_size) if self.batch_size else DEFAULT_BATCH_SIZE

        self.n_epochs = kwargs.get('n_epochs', False) or os.environ.get('N_EPOCHS', None)
        self.n_epochs = int(self.n_epochs) if self.n_epochs else DEFAULT_N_EPOCHS

        self.noise_dimension = kwargs.get('noise_dimension', False) or os.environ.get('NOISE_DIMENSION', None)
        self.noise_dimension = int(self.noise_dimension) if self.noise_dimension else DEFAULT_NOISE_DIMENSION

        self.lr = kwargs.get('lr', False) or os.environ.get('LR', None)
        self.lr = float(self.lr) if self.lr else DEFAULT_LR

        self.display_step = kwargs.get('display_step', False) or os.environ.get('DISPLAY_STEP', None)
        self.display_step = int(self.display_step) if self.display_step else DEFAULT_DISPLAY_STEP

        self.data_folder = kwargs.get('data_folder', False) or os.environ.get('DATA_FOLDER', None)
        self.data_folder = self.data_folder if self.data_folder else DEFAULT_DATA_FOLDER

        self.criterion = nn.BCEWithLogitsLoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.wandb_project = kwargs.get('WANDB_PROJECT', False) or os.environ.get('WANDB_PROJECT', False)

        self.store_wandb_images = kwargs.get('STORE_WANDB_IMAGES', False) or os.environ.get('STORE_WANDB_IMAGES', False)
        self.store_wandb_images = self.store_wandb_images.lower() == 'true'

        self.visualize_images = kwargs.get('VISUALIZE_IMAGES', False) or os.environ.get('VISUALIZE_IMAGES', False)
        self.visualize_images = self.visualize_images.lower() == 'true'

        self.store_local_images = kwargs.get('STORE_LOCAL_IMAGES', False) or os.environ.get('STORE_LOCAL_IMAGES', False)
        self.store_local_images = self.store_local_images.lower() == 'true'

        self.run_name = kwargs.get('RUN_NAME', False) or os.environ.get('RUN_NAME', False)
        self.run_name = self.run_name if self.run_name else DEFAULT_RUN_NAME
        self.run_name = time.strftime("%Y.%m.%d_%H.%M.%S") + '_' + self.run_name

        print(f'Current settings: {self.__dict__}')

        self.init_wandb()

        torch.manual_seed(123)

    def load_env(self):
        """ Load the environment variables from the environment file
        Returns
        -------
        bool
            True if the environment variables were loaded successfully, False otherwise
        """
        res = dotenv.load_dotenv(self.env_file)
        if os.environ.get('VERBOSE', False):
            if res:
                print(f'Loaded config successfully! - {self.env_file}')
            else:
                print(f'Failed to load config! -  {self.env_file}')

        return res

    def init_wandb(self):
        """ Initialize the wandb project if the project name is provided
        """
        if not self.wandb_project or not os.environ.get('WANDB_API_KEY', None):
            return

        wandb.login(key=os.environ.get('WANDB_API_KEY'))
        wandb.init(project=self.wandb_project, config=self.__dict__, name=self.run_name)
