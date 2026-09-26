"""State-conditioned latent skill prior for the extended SEED capability set.

Unlike the legacy conditional VAE, the posterior sees the target trajectory while the prior sees
only the state/history, skill-family, manifold and command condition.  The decoder can therefore
sample a skill at run time from ``p(z | s, H, skill, M_e)``.  The diagonal prior is also the
reference distribution for the later Latent Action Barrier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from manifold_motion.stage2.flow import WindowData, _MLP, _batches, _device, _torch_load


class StateConditionedSkillVAE(nn.Module):
    def __init__(self, target_dim: int, condition_dim: int, latent_dim: int = 32,
                 condition_hidden: int = 128, hidden: int = 256):
        super().__init__()
        self.target_dim = target_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.condition_hidden = condition_hidden
        self.hidden = hidden
        self.condition_encoder = _MLP(condition_dim, condition_hidden, condition_hidden, layers=2)
        self.prior = _MLP(condition_hidden, 2 * latent_dim, hidden, layers=2)
        self.posterior = _MLP(target_dim + condition_hidden, 2 * latent_dim, hidden, layers=3)
        self.decoder = _MLP(latent_dim + condition_hidden, target_dim, hidden, layers=3)

    def _condition(self, condition: torch.Tensor) -> torch.Tensor:
        return self.condition_encoder(condition)

    def prior_stats(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.prior(self._condition(condition)).chunk(2, dim=-1)
        return mean, logvar.clamp(-8.0, 8.0)

    def posterior_stats(self, target: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self._condition(condition)
        mean, logvar = self.posterior(torch.cat([target, encoded], dim=-1)).chunk(2, dim=-1)
        return mean, logvar.clamp(-8.0, 8.0)

    def decode(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([latent, self._condition(condition)], dim=-1))

    def forward(self, target: torch.Tensor, condition: torch.Tensor):
        prior_mean, prior_logvar = self.prior_stats(condition)
        post_mean, post_logvar = self.posterior_stats(target, condition)
        latent = post_mean + torch.exp(0.5 * post_logvar) * torch.randn_like(post_mean)
        return self.decode(latent, condition), prior_mean, prior_logvar, post_mean, post_logvar

    def architecture(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in
                ("target_dim", "condition_dim", "latent_dim", "condition_hidden", "hidden")}


def latent_barrier(latent: torch.Tensor, prior_mean: torch.Tensor, prior_logvar: torch.Tensor,
                   radius: float = 3.0) -> torch.Tensor:
    """Project latent residuals into a diagonal state-conditioned prior ellipsoid."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    scale = torch.exp(0.5 * prior_logvar).clamp_min(1e-4)
    standardized = (latent - prior_mean) / scale
    norm = torch.linalg.vector_norm(standardized, dim=-1, keepdim=True).clamp_min(1e-8)
    factor = torch.minimum(torch.ones_like(norm), torch.as_tensor(radius, device=latent.device) / norm)
    return prior_mean + standardized * factor * scale


def _kl_to_prior(post_mean: torch.Tensor, post_logvar: torch.Tensor,
                 prior_mean: torch.Tensor, prior_logvar: torch.Tensor) -> torch.Tensor:
    value = prior_logvar - post_logvar + (
        torch.exp(post_logvar) + (post_mean - prior_mean).square()) / torch.exp(prior_logvar) - 1.0
    return 0.5 * value.sum(dim=-1).mean()


def _loader_loss(model: StateConditionedSkillVAE, data: WindowData, indices: np.ndarray,
                 device: torch.device, batch_size: int, beta: float, sample: bool) -> tuple[float, float, float]:
    if not len(indices): return float("nan"), float("nan"), float("nan")
    model.eval(); recon_sse=prior_sse=kl_sum=0.0; elements=samples=0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch=indices[start:start+batch_size]
            target=torch.as_tensor(data.target[batch],device=device)
            condition=torch.as_tensor(data.condition[batch],device=device)
            reconstruction, prior_mean, prior_logvar, post_mean, post_logvar=model(target,condition)
            latent=prior_mean if not sample else post_mean
            prior_reconstruction=model.decode(latent,condition)
            recon_sse += float(torch.sum((reconstruction-target).square()).cpu())
            prior_sse += float(torch.sum((prior_reconstruction-target).square()).cpu())
            kl_sum += float(_kl_to_prior(post_mean,post_logvar,prior_mean,prior_logvar).cpu()) * len(batch)
            elements += int(target.numel()); samples += len(batch)
    return recon_sse/elements, prior_sse/elements, kl_sum/samples


def _split_diagnostics(model: StateConditionedSkillVAE, data: WindowData, indices: np.ndarray,
                       device: torch.device) -> dict[str, Any]:
    if not len(indices): return {"windows": 0}
    model.eval()
    with torch.no_grad():
        condition=torch.as_tensor(data.condition[indices],device=device)
        target=torch.as_tensor(data.target[indices],device=device)
        prior_mean,_=model.prior_stats(condition)
        prediction=model.decode(prior_mean,condition)
        mse=float(torch.mean((prediction-target).square()).cpu())
        baseline=float(torch.mean(target.square()).cpu())
    families=[]
    raw_ids=data.raw["primitive"][indices]
    names=data.raw.get("primitive_names")
    for family_id in sorted(set(raw_ids.tolist())):
        local=np.flatnonzero(raw_ids==family_id)
        family_mse=float(torch.mean((prediction[local]-target[local]).square()).cpu())
        family_baseline=float(torch.mean(target[local].square()).cpu())
        families.append({"family_id":int(family_id),"name":str(names[family_id]) if names is not None else str(family_id),
                         "windows":len(local),"prior_mse":family_mse,"zero_baseline_mse":family_baseline,
                         "relative_improvement":1.0-family_mse/max(family_baseline,1e-8)})
    return {"windows":len(indices),"prior_mse":mse,"zero_baseline_mse":baseline,
            "relative_improvement":1.0-mse/max(baseline,1e-8),"families":families}


def train(args: argparse.Namespace) -> int:
    device = _device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data=WindowData.load(args.windows, model_target_field=args.model_target_field)
    train_idx=np.flatnonzero(data.split==0); val_idx=np.flatnonzero(data.split==1)
    test_idx=np.flatnonzero(data.split==2)
    if not len(train_idx): raise ValueError("windows have no training split")
    model=StateConditionedSkillVAE(data.target.shape[1],data.condition.shape[1],args.latent_dim,args.condition_hidden,args.hidden).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    rng=np.random.default_rng(args.seed); args.out.mkdir(parents=True,exist_ok=True)
    history=[]; best=float("inf")
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch in _batches(train_idx,args.batch_size,rng):
            target=torch.as_tensor(data.target[batch],device=device); condition=torch.as_tensor(data.condition[batch],device=device)
            reconstruction,prior_mean,prior_logvar,post_mean,post_logvar=model(target,condition)
            recon=torch.mean((reconstruction-target).square()); kl=_kl_to_prior(post_mean,post_logvar,prior_mean,prior_logvar)
            loss=recon+args.beta*kl
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
            losses.append((float(loss.detach().cpu()),float(recon.detach().cpu()),float(kl.detach().cpu())))
        train_metrics=_loader_loss(model,data,train_idx,device,args.batch_size,args.beta,False)
        val_metrics=_loader_loss(model,data,val_idx if len(val_idx) else test_idx,device,args.batch_size,args.beta,False)
        row={"epoch":epoch,"train_loss":float(np.mean([x[0] for x in losses])),"train_recon":train_metrics[0],"train_prior_recon":train_metrics[1],"train_kl":train_metrics[2],"val_recon":val_metrics[0],"val_prior_recon":val_metrics[1],"val_kl":val_metrics[2]}
        history.append(row)
        score=val_metrics[1]
        if score < best:
            best=score
            torch.save({"schema":"manifold-motion.state-conditioned-skill-prior.v1","architecture":model.architecture(),"model_state":model.state_dict(),"normalizer":data.normalizer.state_dict(),"primitive_count":data.primitive_count,"primitive_names":data.raw.get("primitive_names"),"target_shape":data.target_shape,"condition_dim":data.condition.shape[1],"model_target_field":args.model_target_field,"beta":args.beta,"epoch":epoch},args.out/"skill_prior.pt")
        if epoch==1 or epoch%max(1,args.log_every)==0 or epoch==args.epochs: print(json.dumps(row),flush=True)
    checkpoint=_torch_load(args.out/"skill_prior.pt",device); model.load_state_dict(checkpoint["model_state"])
    test_metrics=_loader_loss(model,data,test_idx,device,args.batch_size,args.beta,False)
    diagnostics={"train":_split_diagnostics(model,data,train_idx,device),
                 "validation":_split_diagnostics(model,data,val_idx,device),
                 "test":_split_diagnostics(model,data,test_idx,device)}
    report={"schema":"manifold-motion.state-conditioned-skill-prior-report.v1","windows":str(args.windows),"epochs":args.epochs,"best_epoch":checkpoint["epoch"],"primitive_count":data.primitive_count,"primitive_names":data.raw.get("primitive_names").tolist() if "primitive_names" in data.raw else None,"train_windows":len(train_idx),"validation_windows":len(val_idx),"test_windows":len(test_idx),"test_recon":test_metrics[0],"test_prior_recon":test_metrics[1],"test_kl":test_metrics[2],"diagnostics":diagnostics,"history":history,"checkpoint":"skill_prior.pt"}
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("primitive_count","train_windows","validation_windows","test_windows","best_epoch","test_prior_recon","test_kl")},indent=2)); return 0


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    train_parser=sub.add_parser("train"); train_parser.add_argument("--windows",type=Path,required=True); train_parser.add_argument("--out",type=Path,default=Path("reports/manifold_motion/skill_prior_v1")); train_parser.add_argument("--epochs",type=int,default=40); train_parser.add_argument("--batch-size",type=int,default=64); train_parser.add_argument("--latent-dim",type=int,default=32); train_parser.add_argument("--condition-hidden",type=int,default=128); train_parser.add_argument("--hidden",type=int,default=256); train_parser.add_argument("--learning-rate",type=float,default=3e-4); train_parser.add_argument("--weight-decay",type=float,default=1e-5); train_parser.add_argument("--beta",type=float,default=1e-3); train_parser.add_argument("--model-target-field",choices=("target_ref","target_exec"),default="target_ref"); train_parser.add_argument("--seed",type=int,default=20260927); train_parser.add_argument("--log-every",type=int,default=5); train_parser.add_argument("--device",default="cpu"); train_parser.set_defaults(handler=train)
    args=parser.parse_args(); return int(args.handler(args))


if __name__=="__main__": raise SystemExit(main())