import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ..nmce.manifold_clustering import MaximalCodingRateReduction, Z_loss
from ..nmce.manifold_clustering import chunk_avg, Gumble_Softmax, MLP_net
import yaml


class TrainNMCE:
    """Class TrainNMCE
    Using NMCE from manifold_clustering.py to train a network to
    cluster synthetic population data.
    
    Parameters to constructor:
    configuration_file ...... a yaml file with parameters to model and training
    
    Public functions:
    train ................... returns a trained NMCE model
    """
    def __init__(self,
                 configuration_file=None,
                 encoder=None,
                 decoder=None
                 ):
        self.configuration_file = configuration_file
        self.encoder = encoder
        self.decoder = decoder
        self.cfg = None
        self.mcrr = None
        self.epochs = None
        self.input_dimension = None
        self.latent_dimension = None
        self.z_dimension = None
        self.clusters = None
        self.n_chunks = None
        self.batch_size = None
        self.lambdas = None
        self.model_name = None
        self.initiate()

    def initiate(self):
        self.set_configurations()
        self.ncrr = self.net()

    def set_configurations(self):
        """Read and set configurations from configuration_file

        Parameters
        ----------

        Returns
        -------
        None

        """
        with open(self.configuration_file, 'r') as file:
            self.cfg = yaml.safe_load(file)
        self.learning_rates = self.cfg["model"]["learning_rates"][0]
        self.epochs = self.cfg["model"]["epochs"][0]
        self.input_dimension = self.cfg["model"]["input_dimensions"][0]
        self.latent_dimension = self.cfg["model"]["latent_dimensions"][0]
        self.z_dimension = self.cfg["model"]["z_dimensions"][0]
        self.clusters = self.cfg["model"]["clusters"][0]
        # [0] = 2 (must change .py file if changed!!)
        self.n_chunks = self.cfg["model"]["n_chunks"][0]
        self.batch_size = int(self.cfg["model"]["batch_size"][2])  # 1929
        self.lambdas = self.cfg["model"]["lambdas"][0]
        self.model_name = str(self.cfg["globals"]["name"])

    def net(self):
        self.mcrr = MLP_net(self.input_dimension,
                            self.latent_dimension,
                            self.z_dimension,
                            self.clusters)

    def train(self, x_data):
        # Store losses
        c_loss = []
        d_loss = []
        z_sim_list = []
        loss_list = []
        n_steps = self.epochs
        print_every = 30
        bs = self.batch_size
        n_chunks = self.n_chunks   # MUST be hardcoded to same value in .py file!!!!
        n_clusters = self.clusters
        lambda_ = self.lambdas

        optimizer = optim.Adam(self.mcrr.parameters(),
                               lr=self.learning_rates,
                               betas=(0.9, 0.99),
                               weight_decay=0.00001)
        g_softmax = Gumble_Softmax(0.2, straight_through=False)

        criterion = MaximalCodingRateReduction(eps=0.01, gamma=1.0)
        criterion_z = Z_loss()
        for i in range(n_steps):
            loader = iter(DataLoader(dataset=x_data, batch_size=bs, shuffle=True))
            # Run one batch and update grads
            iterations = len(loader)
            for j in range(iterations):
                x = next(loader)
                aug_latent_1 = self.encoder(x.numpy())
                aug_latent_2 = self.encoder(x.numpy())
                xn1 = self.decoder(aug_latent_1)
                xn1 = torch.tensor(np.array(xn1), dtype=torch.float32)
                xn2 = self.decoder(aug_latent_2)
                xn2 = torch.tensor(np.array(xn2), dtype=torch.float32)
                xt = torch.cat((xn1, xn2, x, x), dim=0).float()  # add original batch at each!
                z, logits = self.mcrr(xt)
                loss_z, z_sim = criterion_z(z)
                z_sim = z_sim.mean()
                prob = g_softmax(logits)
                z, prob = chunk_avg(z, n_chunks=n_chunks,
                                    normalize=True), chunk_avg(prob, n_chunks=n_chunks)
                loss, loss_list = criterion(z, prob, num_classes=n_clusters)
                loss += lambda_ * loss_z
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Save the loss at the end of run of all batches
                if j % (iterations - 1) == 0:
                    c_loss.append(loss_list[0])
                    d_loss.append(loss_list[1])
                    z_sim_list.append(z_sim.item())
            if i % print_every == 0:
                print('{} steps done, loss c {}, loss d {}, z sim {}'
                      .format(i+1, loss_list[0], loss_list[1], z_sim.item()))
        return self.mcrr, c_loss, d_loss, z_sim_list
