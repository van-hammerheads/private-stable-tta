"""
from https://github.com/davda54/sam
"""

import torch
from opacus.optimizers import DPOptimizer


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class DPSAM(SAM):
    def __init__(self, params, base_optimizer,
                 noise_multiplier, max_grad_norm, batch_size,
                 rho=0.05, adaptive=False, **kwargs):
        super().__init__(
            params=params,
            base_optimizer=base_optimizer,
            rho=rho,
            adaptive=adaptive,
            **kwargs,
        )
        self.dp_optimizer = DPOptimizer(
            optimizer=self.base_optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size
        )

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        # override zero grad to go through dp_optimizer
        self.dp_optimizer.zero_grad(set_to_none=set_to_none)
        self.base_optimizer.zero_grad()

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

    @torch.no_grad()
    def second_step(self, zero_grad):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                state = self.state.get(p, None)
                if state is None or "old_p" not in state:
                    continue
                p.data = state["old_p"]

        # call dp_optimizer.step in second step
        self.dp_optimizer.step()


class DPSAT(SAM):
    def __init__(
        self,
        params,
        base_optimizer,
        noise_multiplier,
        max_grad_norm,
        batch_size,
        rho=0.05,
        adaptive=False,
        tau=1e-12,
        **kwargs,
    ):
        super().__init__(
            params=params,
            base_optimizer=base_optimizer,
            rho=rho,
            adaptive=adaptive,
            **kwargs,
        )
        self.dp_optimizer = DPOptimizer(
            optimizer=self.base_optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
        )

        self.tau = tau

        # Flattened previous privatized gradient g_{t-1}^p (initialized to zeros)
        self._g_prev = None
        self._init_g_prev()

    def _init_g_prev(self):
        # build a flat buffer matching all params
        vecs = []
        device = None
        for group in self.param_groups:
            for p in group["params"]:
                if device is None:
                    device = p.device
                vecs.append(torch.zeros_like(p, device=p.device).reshape(-1))
        self._g_prev = torch.cat(vecs) if len(vecs) > 0 else None

    def _flatten_like_params(self, tensor_list):
        return torch.cat([t.reshape(-1) for t in tensor_list])

    def _unflatten_to_params(self, flat):
        idx = 0
        outs = []
        for group in self.param_groups:
            for p in group["params"]:
                n = p.numel()
                outs.append(flat[idx:idx+n].view_as(p))
                idx += n
        return outs

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        self.dp_optimizer.zero_grad(set_to_none=set_to_none)
        self.base_optimizer.zero_grad()

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        Perturb params using previous privatized gradient buffer g_prev (post-processing).
        """
        if self._g_prev is None:
            self._init_g_prev()

        # Compute norm of previous gradient
        g_prev_norm = torch.norm(self._g_prev)
        scale = self.param_groups[0]["rho"] / (g_prev_norm + self.tau)

        # Unflatten g_prev to per-parameter tensors
        g_prev_per_param = self._unflatten_to_params(self._g_prev)

        k = 0
        for group in self.param_groups:
            adaptive = group["adaptive"]
            for p in group["params"]:
                # Save original weights
                self.state[p]["old_p"] = p.data.clone()

                # DP-SAT perturbation direction uses g_{t-1}^p (not current p.grad)
                gdir = g_prev_per_param[k]
                k += 1

                if adaptive:
                    # mimic SAM adaptive scaling (ASAM-style)
                    e_w = torch.pow(p, 2) * gdir * scale.to(p)
                else:
                    e_w = gdir * scale.to(p)

                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _update_g_prev_from_dp(self):
        grads = []
        found = False

        # 1) Common case: p.grad contains the final (clipped+noised) grad at step time
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    grads.append(torch.zeros_like(p).reshape(-1))
                else:
                    grads.append(p.grad.detach().clone().reshape(-1))
                    found = True

        if found:
            self._g_prev = torch.cat(grads)
            return

        # 2) Try known internal names (best-effort; may not exist)
        # You can extend this if your Opacus version stores gradients elsewhere.
        try:
            # some versions keep accumulators/noise in dp_optimizer
            # (this is speculative—keep as fallback)
            if hasattr(self.dp_optimizer, "noise"):
                pass
        except Exception:
            pass

        # If we get here, keep g_prev unchanged (it will still work, just less faithful)
        # but practically p.grad should exist in typical training.
        return

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        Restore original weights and apply ONE DP step (this is the only private access per iter).
        """
        # Restore weights
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, None)
                if state is None or "old_p" not in state:
                    continue
                p.data = state["old_p"]

        # Apply DP step (clipping + noise + optimizer update)
        self.dp_optimizer.step()

        # Update g_prev for next iteration's ascent direction
        self._update_g_prev_from_dp()

        if zero_grad:
            self.zero_grad()
