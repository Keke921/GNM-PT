import torch
from torch.distributions import normal

class GNM(torch.optim.Optimizer):
    def __init__(self, config, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(RSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        
        self.simpler = normal.Normal(0, 1/3)
        self.dis_base_mul = config.dis_mul
        self.config = config
        #cls_list = torch.cuda.FloatTensor(config.cls_num_list)
        #m_list = torch.log(cls_list)  
        #m_list = m_list-m_list.min() #m_list.max()-m_list
        #self.m_list = m_list        
        
        
    @torch.no_grad()
    def first_step(self, epoch, zero_grad=False):
        #grad_norm = self._grad_norm()
        for group in self.param_groups:
            #scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.requires_grad is False: continue #if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                noise = self.simpler.sample(p.shape).clamp(-1, 1).to(p)
                noise = noise *self.dis_base_mul*0.5**((epoch-1)//self.config.loss.GCL.reweight_epoch)
                #noise_ = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                e_w = group["rho"]* noise #**(1-epoch//50) * noise_**(epoch//50) 
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()




    @torch.no_grad()
    def second_step(self, zero_grad=False):
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
