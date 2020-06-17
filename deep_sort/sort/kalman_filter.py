import torch

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter:

    def __init__(self, device="cpu"):
        self._device = device

        ndim, dt = 4, 1
        #  Create Kalman filter model matrices (8, 8)
        self._motion_mat = torch.eye(2 * ndim, 2 * ndim, device=self._device, dtype=torch.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._motion_mat = self._motion_mat.t()

        # (8, 4)
        self._update_mat = torch.eye(2 * ndim, ndim, device=self._device, dtype=torch.float32)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

        self._std_pos = torch.tensor([[
            self._std_weight_position,
            self._std_weight_position,
            0,
            self._std_weight_position]], device=self._device)

        self._std_vel = torch.tensor([[
            self._std_weight_velocity,
            self._std_weight_velocity,
            0,
            self._std_weight_velocity]], device=self._device)

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h. shape of (4, 1)

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = torch.zeros_like(mean_pos)
        mean = torch.cat([mean_pos, mean_vel], dim=-1).view(1, -1)  # (1, 8)

        std = torch.tensor([[
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]],
            device=measurement.device)

        covariance = torch.diag_embed(torch.pow(std, 2))  # (1, 8, 8)
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of shape (*, 8) of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step. which shape is (*, 8, 8)

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """

        std_pos = mean[:, 3:4] * self._std_pos
        std_vel = mean[:, 3:4] * self._std_vel

        std_pos[:, 2] = 1e-2
        std_vel[:, 2] = 1e-5

        # (*, 8, 8)
        motion_cov = torch.diag_embed(torch.pow(torch.cat([std_pos, std_vel], dim=-1), 2))

        mean = torch.matmul(mean, self._motion_mat)  # (*, 8)

        # (*, 8, 8)
        covariance = torch.matmul(torch.matmul(covariance.permute(0, 2, 1), self._motion_mat).permute(0, 2, 1),
                                  self._motion_mat)
        return mean, covariance + motion_cov

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (*, 8).
        covariance : ndarray
            The state's covariance matrix (*, 8, 8).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """

        std = mean[:, 3:4] * self._std_pos  # (*, 4)
        std[:, 2] = 1e-1  # (*, 4)

        # (*, 4, 4)
        innovation_cov = torch.diag_embed(torch.pow(std, 2))

        # # (4, 8) dot (*, 8)
        # mean = torch.mm(mean, self._update_mat)  # (*, 4)
        #
        # # (*, 4, 4)
        # covariance = torch.matmul(torch.matmul(covariance.permute(0, 2, 1), self._update_mat).permute(0, 2, 1),
        #                           self._update_mat)

        # Simple way to reduce the amount of calculation
        mean = mean[:, :4].clone()
        covariance = covariance[:, :4, :4].clone()
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (*, 8).
        covariance : ndarray
            The state's covariance matrix (*,8,8).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """

        projected_mean, projected_cov = self.project(mean, covariance)

        # Unfortunately, cholesky will randomly throw CUDA ERROR under linux GPU environment,
        # this error is from pytorch1.2 to 1.5
        # chol_factor = torch.cholesky(projected_cov, upper=False)
        # kalman_gain = torch.cholesky_solve(torch.matmul(covariance, self._update_mat).permute(0, 2, 1),
        #                                    chol_factor,
        #                                    upper=False).permute(0, 2, 1)

        # The alternative solution is theoretically slower than cholesky_solve
        kalman_gain = torch.solve(torch.matmul(covariance, self._update_mat).permute(0, 2, 1), projected_cov)[
            0].permute(0, 2, 1)

        # (*, 4)
        innovation = measurement.view(-1, 4) - projected_mean

        kalman_gain_t = kalman_gain.permute(0, 2, 1)
        new_mean = mean + torch.bmm(innovation.unsqueeze(1), kalman_gain_t).view(-1, 8)  # (*, 8)
        new_covariance = covariance - torch.matmul(
            torch.matmul(projected_cov.permute(0, 2, 1), kalman_gain_t).permute(0, 2, 1),
            kalman_gain_t)

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (n, 8).
        covariance : ndarray
            Covariance of the state distribution (n, 8, 8).
        measurements : ndarray
            An Mx4 dimensional matrix of M measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an matrix of shape N X M, where the i-th row contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[j]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:, None, :2], covariance[:, :2, :2]
            measurements = measurements[None, :, :2]
        else:
            mean = mean.unsqueeze(1)
            measurements = measurements.unsqueeze(0)

        d = - mean + measurements  # (n, m, 4)

        # Unfortunately, cholesky will randomly throw CUDA ERROR under linux GPU environment,
        # this error is from pytorch1.2 to 1.5

        # cholesky_factor = torch.cholesky(covariance)
        # z = torch.triangular_solve(d.permute(0, 2, 1), cholesky_factor, upper=False)[0]
        # squared_maha = torch.sum(z ** 2, dim=1)  # (n, m)

        # The alternative solution is theoretically slower than cholesky_solve
        squared_maha = torch.bmm(torch.bmm(d, torch.inverse(covariance)), d.permute(0, 2, 1))
        squared_maha = torch.diagonal(squared_maha, dim1=-2, dim2=-1)

        return squared_maha


if __name__ == '__main__':
    kf = KalmanFilter()
    mean, covariance = kf.initiate(torch.tensor([10, 15, 0.5, 10]))
    mean, covariance = kf.predict(mean, covariance)
    mean, covariance = kf.update(mean, covariance, torch.tensor([12, 20, 0.6, 11]))

    mean_2, covariance_2 = kf.initiate(torch.tensor([12, 13, 0.7, 5]))
    mean_2, covariance_2 = kf.predict(mean_2, covariance_2)
    mean_2, covariance_2 = kf.update(mean_2, covariance_2, torch.tensor([13, 14, 0.7, 8]))

    squared_maha = kf.gating_distance(torch.cat((mean, mean_2), dim=0),
                                      torch.cat((covariance, covariance_2), dim=0),
                                      torch.tensor([[12, 20, 0.6, 11],
                                                    [20, 16, 0.4, 18]]))
    print(squared_maha)
