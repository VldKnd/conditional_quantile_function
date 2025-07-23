import torch

def estimate_entropy_dual_feasibility_condition(X_tensor, Y_tensor, phi_net, psi_net, epsilon=0.1):
        """
        Estimate the dual objective term for entropy estimation.

        This function implements the core calculation based on potential functions phi and psi.

        Args:
        X_tensor (torch.Tensor): The input tensor for x, with shape [n, p].
        Y_tensor (torch.Tensor): The input tensor for y, with shape [n, q].
        phi_net (nn.Module): The neural network representing the potential function phi(u, x).
        psi_net (nn.Module): The neural network representing the potential function psi(x, y).
        epsilon (float, optional): A small positive constant for the calculation. Defaults to 0.1.

        Returns:
        torch.Tensor: A scalar tensor representing the estimated dual value.
        """
        n, _ = X_tensor.shape
        m = n
        U_tensor = torch.randn_like(Y_tensor)[:m, :]

        U_expanded_for_X = U_tensor.unsqueeze(0).expand(n, -1, -1)
        X_expanded_for_U = X_tensor.unsqueeze(1).expand(-1, m, -1)

        XU = torch.cat((X_expanded_for_U, U_expanded_for_X), dim=-1)
        XY = torch.cat((X_tensor, Y_tensor), dim=-1)

        phi_vals = phi_net(XU).squeeze(-1) # n x n [i, j] = phi(x_i, u_j)
        psi_vals = psi_net(XY) # n x 1 [i] = psi(x_i, y_i)
        objective_term = Y_tensor @ U_tensor.T # n x n [i, j] = y_i @ u_j^T

        slackness = ( objective_term - phi_vals - psi_vals )
        exponent_val = torch.exp(slackness / epsilon)
        dual_estimate = epsilon * torch.mean(exponent_val)
        return dual_estimate

def estimate_entropy_dual_psi(X_tensor, Y_tensor, U_tensor, phi_net, epsilon=0.1):
        """
        Estimate psi: X, Y -> R^1 objective term. Theorem 2.

        Args:
        X_tensor (torch.Tensor): The input tensor for x, with shape [n, p].
        Y_tensor (torch.Tensor): The input tensor for y, with shape [n, q].
        U_tensor (torch.Tensor): The tensor of oversampled variables u, with shape [m, q].
        phi_net (nn.Module): The neural network representing the potential function phi(u, x).
        epsilon (float, optional): A small positive constant for the calculation. Defaults to 0.1.

        Returns:
        torch.Tensor: A scalar tensor representing the estimated psi value.
        """
        n, _ = X_tensor.shape
        m, _ = U_tensor.shape

        U_expanded = U_tensor.unsqueeze(0).expand(n, -1, -1)
        X_expanded_for_U = X_tensor.unsqueeze(1).expand(-1, m, -1)
        XU = torch.cat((X_expanded_for_U, U_expanded), dim=-1)

        phi_vals = phi_net(XU).squeeze(-1)
        cost_matrix = Y_tensor @ U_tensor.T

        slackness = cost_matrix - phi_vals

        log_mean_exp = torch.logsumexp(slackness / epsilon, dim=1, keepdim=True) \
                - torch.log(torch.tensor(m, device=slackness.device, dtype=slackness.dtype))

        psi_estimate = epsilon * log_mean_exp

        return psi_estimate

def estimate_entropy_dual_phi(X_tensor, U_tensor, X_dataset, Y_dataset, psi_net, k=5, epsilon=0.1):
        """
        Estimate phi: X, U -> R^1 objective term. Theorem 2.

        Since the distribution of y|x is not accesible, we are using the nearest neighbors of x to estimate phi.
        By this we mean, that to sample y ~ y|x we sample x ~ x and take y's assosiated with k nearest neighbors of x.

        Args:
        X_tensor (torch.Tensor): The input tensor for x, with shape [n, p].
        Y_tensor (torch.Tensor): The input tensor for y, with shape [n, q].
        U_tensor (torch.Tensor): The tensor of oversampled variables u, with shape [m, q].
        psi_net (nn.Module): The neural network representing the potential function psi(x, y).
        epsilon (float, optional): A small positive constant for the calculation. Defaults to 0.1.

        Returns:
        torch.Tensor: A scalar tensor representing the estimated phi value.
        """
        dists = torch.cdist(X_tensor, X_dataset, p=2.0)
        _, neighbor_indices = torch.topk(dists, k + 1, dim=1, largest=False)
        Y_neighbors = Y_dataset[neighbor_indices] # n, k+1, q

        cost_matrix = torch.einsum('ab,adb->ad', U_tensor, Y_neighbors) # n, m
        X_expanded_for_Y = X_tensor.unsqueeze(1).expand(-1, k + 1, -1)   # n, k+1, p

        psi_potential = psi_net(torch.cat([X_expanded_for_Y, Y_neighbors], dim=-1)).squeeze() # n, k+1
        slackness = cost_matrix - psi_potential # n, k+1

        log_mean_exp = torch.logsumexp(slackness / epsilon, dim=1, keepdim=True) \
                - torch.log(torch.tensor(k+1, device=slackness.device, dtype=slackness.dtype))

        phi_estimate = epsilon * log_mean_exp

        return phi_estimate

def estimate_Y_from_psi(X_tensor, U_tensor, psi_net, verbose=False):
        """
        Estimate Y tensor by minimizing u^T y - psi(x, y) for given x and u.
        psi(x, y) is assume to be a potential function convex in y.

        Args:
        X_tensor (torch.Tensor): The input tensor for x, with shape [n, p].
        U_tensor (torch.Tensor): The tensor of oversampled variables u, with shape [n, q].
        psi_net (PICNN): Partially input convex neural network representing the potential function psi(x, y).

        Returns:
        torch.Tensor: A scalar tensor representing the estimated phi value.
        """
        Y_tensor = torch.randn_like(U_tensor)
        Y_tensor.requires_grad = True

        optimizer = torch.optim.LBFGS(
            [Y_tensor],
            lr=1,
            line_search_fn="strong_wolfe",
            max_iter=1000,
            tolerance_grad=1e-10,
            tolerance_change=1e-10
        )

        def slackness_closure():
            optimizer.zero_grad()
            cost_matrix = torch.sum(U_tensor * Y_tensor, dim=-1, keepdims=True)
            psi_potential = psi_net(X_tensor, Y_tensor)
            slackness = (psi_potential - cost_matrix).sum()
            slackness.backward()
            return slackness

        optimizer.step(slackness_closure)

        if verbose:
            optimal_Y_tensor_potential = psi_net(X_tensor, Y_tensor).sum()
            approximated_U_tensor = torch.autograd.grad(optimal_Y_tensor_potential.sum(), Y_tensor)[0]
            estimation_error = (approximated_U_tensor - U_tensor)
            print(f"Maximal estimation error: {estimation_error.abs().max().item()}, minimal estimation error: {estimation_error.abs().max().item()}")

        return Y_tensor

def estimate_U_from_phi(X_tensor, Y_tensor, phi_net, verbose=False):
        """
        Estimate U tensor by minimizing u^T y - phi(x, u) for given x and y.
        phi(x, u) is assume to be a potential function convex in u.

        Args:
        X_tensor (torch.Tensor): The input tensor for x, with shape [n, p].
        Y_tensor (torch.Tensor): The tensor of oversampled variables y, with shape [n, q].
        phi_net (PICNN): Partially input convex neural network representing the potential function phi(x, u).

        Returns:
        torch.Tensor: A scalar tensor representing the estimated phi value.
        """
        U_tensor = torch.randn_like(Y_tensor)
        U_tensor.requires_grad = True

        optimizer = torch.optim.LBFGS(
            [U_tensor],
            lr=1,
            line_search_fn="strong_wolfe",
            max_iter=1000,
            tolerance_grad=1e-10,
            tolerance_change=1e-10
        )

        def slackness_closure():
            optimizer.zero_grad()
            cost_matrix = torch.sum(U_tensor * Y_tensor, dim=-1, keepdims=True)
            phi_potential = phi_net(X_tensor, U_tensor)
            slackness = (phi_potential - cost_matrix).sum()
            slackness.backward()
            return slackness

        optimizer.step(slackness_closure)

        if verbose:
            optimal_U_tensor_potential = phi_net(X_tensor, U_tensor).sum()
            approximated_Y_tensor = torch.autograd.grad(optimal_U_tensor_potential.sum(), U_tensor)[0]
            estimation_error = (approximated_Y_tensor - Y_tensor)
            print(f"Maximal estimation error: {estimation_error.abs().max().item()}, minimal estimation error: {estimation_error.abs().max().item()}")

        return U_tensor