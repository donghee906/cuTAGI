###############################################################################
# File:         regression.py
# Description:  Example of regression task using pytagi
# Authors:      Luong-Ha Nguyen & James-A. Goulet
# Created:      October 12, 2022
# Updated:      November 07, 2022
# Contact:      luongha.nguyen@gmail.com & james.goulet@polymtl.ca
# License:      This code is released under the MIT License.
###############################################################################
from typing import Union, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import pytagi.metric as metric
from pytagi import NetProp, TagiNetwork
from pytagi import Normalizer as normalizer
from pytagi import Utils, exponential_scheduler
from visualizer import PredictionViz


class Regression:
    """Regression task using TAGI"""

    utils: Utils = Utils()

    def __init__(self,
                 num_epochs: int,
                 data_loader: dict,
                 net_prop: NetProp,
                 dtype=np.float32,
                 viz: Union[PredictionViz, None] = None) -> None:
        self.num_epochs = num_epochs
        self.data_loader = data_loader
        self.net_prop = net_prop
        self.network = TagiNetwork(self.net_prop)
        self.dtype = dtype
        self.viz = viz
        #self.normalizer = normalizer()

    def train(self) -> None:
        """Train the network using TAGI"""
        # Inputs
        batch_size = self.net_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        # Outputs
        V_batch, ud_idx_batch = self.init_outputs(batch_size)

        input_data, output_data = self.data_loader["train"]
        num_data = input_data.shape[0]
        num_iter = int(num_data / batch_size)
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:
            # Decaying observation's variance
            self.net_prop.sigma_v = exponential_scheduler(
                curr_v=self.net_prop.sigma_v,
                min_v=self.net_prop.sigma_v_min,
                decaying_factor=self.net_prop.decay_factor_sigma_v,
                curr_iter=epoch)
            V_batch = V_batch * 0.0 + self.net_prop.sigma_v**2

            for i in range(num_iter):
                # Get data
                idx = np.random.choice(num_data, size=batch_size)
                x_batch = input_data[idx, :]
                y_batch = output_data[idx, :]

                # Feed forward
                self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)

                # Update hidden states
                self.network.state_feed_backward(y_batch, V_batch,
                                                 ud_idx_batch)

                # Update parameters
                self.network.param_feed_backward()

                # Loss
                norm_pred, _ = self.network.get_network_predictions()
                pred = normalizer.unstandardize(
                    norm_data=norm_pred,
                    mu=self.data_loader["y_norm_param_1"],
                    std=self.data_loader["y_norm_param_2"])
                obs = normalizer.unstandardize(
                    norm_data=y_batch,
                    mu=self.data_loader["y_norm_param_1"],
                    std=self.data_loader["y_norm_param_2"])
                mse = metric.mse(pred, obs)
                pbar.set_description(
                    f"Epoch# {epoch: 0}|{i * batch_size + len(x_batch):>5}|{num_data: 1}\t mse: {mse:>7.2f}"
                )

    def predict(self, std_factor: int = 1) -> None:
        """Make prediction using TAGI"""
        # Inputs
        batch_size = self.net_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        #sy_pred = sy_pred.flatten()
        #sy_test = sy_test.flatten()

        mean_predictions = []
        variance_predictions = []
        y_test = []
        x_test = []
        x_train = self.data_loader["train"][0].flatten()
        y_train = self.data_loader["train"][1].flatten()
        for x_batch, y_batch in self.data_loader["test"]:
            # Predicitons
            self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)
            ma, Sa = self.network.get_network_predictions()

            mean_predictions.append(ma)
            variance_predictions.append(Sa + self.net_prop.sigma_v**2)
            x_test.append(x_batch)
            y_test.append(y_batch)

        mean_predictions = np.stack(mean_predictions).flatten()
        std_predictions = (np.stack(variance_predictions).flatten())**0.5
        y_test = np.stack(y_test).flatten()
        x_test = np.stack(x_test).flatten()

        # Unnormalization
        mean_predictions = normalizer.unstandardize(
            norm_data=mean_predictions,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"])
        std_predictions = normalizer.unstandardize_std(
            norm_std=std_predictions, std=self.data_loader["y_norm_param_2"])

        x_test = normalizer.unstandardize(
            norm_data=x_test,
            mu=self.data_loader["x_norm_param_1"],
            std=self.data_loader["x_norm_param_2"])
        y_test = normalizer.unstandardize(
            norm_data=y_test,
            mu=self.data_loader["y_norm_param_1"],
            std=self.data_loader["y_norm_param_2"])

        # Compute log-likelihood
        mse = metric.mse(mean_predictions, y_test)
        log_lik = metric.log_likelihood(prediction=mean_predictions,
                                        observation=y_test,
                                        std=std_predictions)

            # 조정할 범위
        desired_min = -1
        desired_max = 1

    # 현재 y_train의 최소값과 최대값
        current_min = np.min(y_train)
        current_max = np.max(y_train)

    # 현재 y_train을 조정하여 원하는 범위로 맞춤
       # y_train_adjusted = ((y_train - current_min) / (current_max - current_min)) * (desired_max - desired_min) + desired_min

    # Normalize y_train_adjusted using self.normalizer
        #y_train_normalized = self.normalizer.standardize(data=y_train_adjusted,
                                                      #mu=self.data_loader["y_norm_param_1"],
                                                      #std=self.data_loader["y_norm_param_2"])

        x_train_normalized = self.data_loader["train"][0]  # 정규화된 x_train 값
        x_train_modified = normalizer.unstandardize(
            norm_data=x_train_normalized,
            mu=self.data_loader["x_norm_param_1"],
            std=self.data_loader["x_norm_param_2"])

        y_train_normalized = self.data_loader["train"][1]  # 정규화된 y_train 값
        y_train_modified = normalizer.unstandardize(
          norm_data=y_train_normalized,
          mu=self.data_loader["y_norm_param_1"],
          std=self.data_loader["y_norm_param_2"])

        #밑에 코드는 heteros_Regression을 위한 코드가 아니라서 새로 코드를 짰음
        # Visualization
        #if self.viz is not None:
            #self.viz.plot_predictions(
                #x_train=x_train,
                #y_train=y_train,
                #x_test=x_test,
                #y_test=y_test,
                #y_pred=mean_predictions,
                #sy_pred=std_predictions,
                #std_factor=std_factor,
                #sy_test=std_predictions,
                #label="diag",
                #title="heteriscedastic Noise Inference",

        #직접 짠 heteros_regression의 visualization 코드 부분
                    # Visualization
        if self.viz is not None:
                #x_train = self.data_loader["train"][0].flatten()
                x_train = x_train_modified
                y_train = y_train_modified
                x_test = np.stack(x_test).flatten()
                y_test = np.stack(y_test).flatten()

        # visualizer.py에 noise_inference에 대해 정의 되어있는 부분을 참고하여 새로 작성한 코드 
        self.viz.plot_predictions(
                 x_train=x_train,
                 y_train=y_train,
                 x_test=x_test,
                 y_test=y_test,
                 y_pred=mean_predictions,
                 sy_pred=std_predictions,
                 std_factor=std_factor,
                 sy_test=std_predictions,
                 label="hete_2",
                 title="Heteroscedastic Noise Inference",
                 eq=r"$\begin{array}{rcl}Y &=& -(x + 0.5)\sin(3\pi x) + V, ~V\sim\mathcal{N}(0, \sigma_V^2)\\[4pt]\sigma_V &=& 0.45(x + 0.5)^2\end{array}$",
                 x_eq=-0.98,
                 y_eq=1.6,
                 time_series=False,
                 save_folder=None
                    )


        print("#############")
        print(f"MSE           : {mse: 0.2f}")
        print(f"Log-likelihood: {log_lik: 0.2f}")

    def compute_derivatives(self,
                            layer: int = 0,
                            truth_derv_file: Union[None, str] = None) -> None:
        """Compute dervative of a given layer"""
        # Inputs
        batch_size = self.net_prop.batch_size
        Sx_batch, Sx_f_batch = self.init_inputs(batch_size)

        mean_derv = []
        variance_derv = []
        x_test = []
        for x_batch, _ in self.data_loader["test"]:
            # Predicitons
            self.network.feed_forward(x_batch, Sx_batch, Sx_f_batch)
            mdy, vdy = self.network.get_derivatives(layer)

            mean_derv.append(mdy)
            variance_derv.append(vdy)
            x_test.append(x_batch)

        mean_derv = np.stack(mean_derv).flatten()
        std_derv = (np.stack(variance_derv).flatten())**0.5
        x_test = np.stack(x_test).flatten()

        # Unnormalization
        x_test = normalizer.unstandardize(
            norm_data=x_test,
            mu=self.data_loader["x_norm_param_1"],
            std=self.data_loader["x_norm_param_2"])

        if truth_derv_file is not None:
            truth_dev_test = pd.read_csv(truth_derv_file,
                                         skiprows=1,
                                         delimiter=",",
                                         header=None)
            self.viz.plot_predictions(
                x_train=None,
                y_train=None,
                x_test=x_test,
                y_test=truth_dev_test.values,
                y_pred=mean_derv,
                sy_pred=std_derv,
                std_factor=3,
                label="deriv",
                title="Neural Network's Derivative",
            )

    def init_inputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for inputs"""
        Sx_batch = np.zeros((batch_size, self.net_prop.nodes[0]),
                            dtype=self.dtype)

        Sx_f_batch = np.array([], dtype=self.dtype)
        if self.net_prop.is_full_cov:
            Sx_f_batch = self.utils.get_upper_triu_cov(
                batch_size=batch_size,
                num_data=self.net_prop.nodes[0],
                sigma=self.net_prop.sigma_x)
            Sx_batch = Sx_batch + self.net_prop.sigma_x**2

        return Sx_batch, Sx_f_batch

    def init_outputs(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Initnitalize the covariance matrix for outputs"""
        # Outputs
        V_batch = np.zeros((batch_size, self.net_prop.nodes[-1]),
                           dtype=self.dtype) + self.net_prop.sigma_v**2
        ud_idx_batch = np.zeros((batch_size, 0), dtype=np.int32)

        return V_batch, ud_idx_batch
