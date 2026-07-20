# TorchScript-clean PyTorch port of the DMFF JAX sGNN (MolGNNForce).
# - Fixed graph arrays + MLP weights baked as buffers (constant for a given topology).
# - forward(positions[, boxvectors]) -> scalar energy, matching the openmm-torch TorchForce contract.
# - Architecture is hard-unrolled (fc0: 3 layers, fc1: 2 layers) so torch.jit.script works.
#
# Verifies: eager == JAX (float64), scripted == eager, autograd forces.  Run in `bff` env:
#   python sgnn_ts.py sgnn_ref.npz
import sys
from typing import Optional, List
import numpy as np
import torch
from torch import Tensor

torch.set_default_dtype(torch.float64)


class SGNNForce(torch.nn.Module):
    def __init__(self, npz, pos_scale=1.0):
        """pos_scale: multiply incoming positions (and provided box) by this before the
        model. The sGNN works in Angstrom (dmff from_pdb does xyz_nm*10). Feed Angstrom
        directly with pos_scale=1.0; as an OpenMM TorchForce (positions arrive in nm) use
        pos_scale=10.0 to convert nm -> Angstrom."""
        super().__init__()
        d = np.load(npz, allow_pickle=True)
        F64 = torch.float64
        self.pos_scale = float(pos_scale)

        def buf(name, arr, dtype=F64):
            self.register_buffer(name, torch.as_tensor(np.asarray(arr), dtype=dtype))

        buf('box', d['box'])
        buf('bonds', d['bonds'], torch.long); buf('b0', d['b0'])
        buf('angles', d['angles'], torch.long); buf('cos_a0', d['cos_a0'])
        buf('diheds', d['diheds'], torch.long)
        buf('feat_atypes', d['feat_atypes'])
        buf('fi_bonds', d['fi_bonds'], torch.long); buf('fi_angles0', d['fi_angles0'], torch.long)
        buf('fi_angles1', d['fi_angles1'], torch.long); buf('fi_diheds', d['fi_diheds'], torch.long)
        buf('nb_connect', d['nb_connect']); buf('weights', d['weights'])

        # MLP weights (fc0: 3 layers, fc1: 2 layers) — named buffers, referenced explicitly in forward
        w0, b0 = d['fc0_w'], d['fc0_b']
        w1, b1 = d['fc1_w'], d['fc1_b']
        assert len(w0) == 3 and len(w1) == 2, "architecture assumed (3,2) layers"
        buf('fc0_w0', w0[0]); buf('fc0_w1', w0[1]); buf('fc0_w2', w0[2])
        buf('fc0_b0', b0[0]); buf('fc0_b1', b0[1]); buf('fc0_b2', b0[2])
        buf('fc1_w0', w1[0]); buf('fc1_w1', w1[1])
        buf('fc1_b0', b1[0]); buf('fc1_b1', b1[1])
        buf('fcf_w', d['fcf_w']); buf('fcf_b', np.atleast_1d(d['fcf_b']))

        # ---- precompute padding-free gather indices (constant per topology) ----
        # The feature gather fic[idx] has -1 padding; fic[-1] sends every padding slot to the
        # SAME source row, so its CUDA backward scatter-adds with massive atomic contention on
        # that one row (~22 ms/gather). Instead, route each padding slot to its own unique dummy
        # row appended to fic (value 0), and use index_select (index_add backward, no contention).
        # Result is bit-identical (padding rows read 0, same as the old heaviside mask) but ~240x
        # faster. Source lengths match features(): fb=(n_bonds,), fa=(n_angles,), fd=(n_diheds,).
        n_bond_src = int(self.bonds.shape[0])
        n_ang_src = int(self.angles.shape[0])
        n_dih_src = int(self.diheds.shape[0]) if self.diheds.numel() > 0 else 1

        def fix_idx(name, idx, nsrc):
            pad = idx < 0
            npad = int(pad.sum().item())
            fixed = idx.clone()
            if npad > 0:
                fixed[pad] = torch.arange(nsrc, nsrc + npad, dtype=torch.long)
            self.register_buffer(name, fixed)
            return npad

        self.fi_bonds_pad = fix_idx('fi_bonds_fixed', self.fi_bonds, n_bond_src)
        self.fi_angles0_pad = fix_idx('fi_angles0_fixed', self.fi_angles0, n_ang_src)
        self.fi_angles1_pad = fix_idx('fi_angles1_fixed', self.fi_angles1, n_ang_src)
        self.fi_diheds_pad = fix_idx('fi_diheds_fixed', self.fi_diheds, n_dih_src)

        # scalar constants
        self.fscale_bond = float(d['fscale_bond']); self.fscale_angle = float(d['fscale_angle'])
        self.max_valence = int(d['max_valence']); self.sigma = float(d['sigma']); self.mu = float(d['mu'])
        self.w = float(d['w'])

    def _pbc(self, dr: Tensor, box: Tensor, box_inv: Tensor) -> Tensor:
        ds = dr @ box_inv
        ds = ds - torch.floor(ds + 0.5)
        return ds @ box

    def _gather(self, fic: Tensor, idx_fixed: Tensor, n_pad: int) -> Tensor:
        # distribute_scalar with padding routed to unique zero dummy rows (see __init__).
        # Equivalent to the old fic[idx]*heaviside(idx) but with index_select's fast, contention-
        # free backward instead of advanced-indexing's atomic scatter.
        if n_pad > 0:
            fic = torch.cat([fic, torch.zeros(n_pad, dtype=fic.dtype, device=fic.device)])
        out = fic.index_select(0, torch.flatten(idx_fixed))
        return out.reshape(idx_fixed.shape)

    def features(self, pos: Tensor, box: Tensor) -> Tensor:
        dt = pos.dtype
        box_inv = torch.linalg.inv(box)
        # bonds
        dr = self._pbc(pos[self.bonds[:, 1]] - pos[self.bonds[:, 0]], box, box_inv)
        fb = (torch.linalg.norm(dr, dim=1) - self.b0) * self.fscale_bond
        # angles
        r_ij = self._pbc(pos[self.angles[:, 0]] - pos[self.angles[:, 1]], box, box_inv)
        r_ik = self._pbc(pos[self.angles[:, 2]] - pos[self.angles[:, 1]], box, box_inv)
        cos_a = (r_ij * r_ik).sum(1) / (torch.linalg.norm(r_ij, dim=1) * torch.linalg.norm(r_ik, dim=1))
        fa = (cos_a - self.cos_a0) * self.fscale_angle
        # diheds
        if self.diheds.numel() == 0:
            fd = torch.zeros(1, dtype=dt)
        else:
            di = self.diheds
            ri = pos[di[:, 0]]; rj = pos[di[:, 1]]; rk = pos[di[:, 2]]; rl = pos[di[:, 3]]
            r_jk = self._pbc(rk - rj, box, box_inv); r_ji = self._pbc(ri - rj, box, box_inv)
            r_kl = self._pbc(rl - rk, box, box_inv); r_kj = -r_jk
            n1 = torch.cross(r_jk, r_ji, dim=1); n2 = torch.cross(r_kl, r_kj, dim=1)
            fd = (n1 * n2).sum(1) / (torch.linalg.norm(n1, dim=1) * torch.linalg.norm(n2, dim=1))
        f_bonds = self._gather(fb, self.fi_bonds_fixed, self.fi_bonds_pad)
        f_ang0 = self._gather(fa, self.fi_angles0_fixed, self.fi_angles0_pad)
        f_ang1 = self._gather(fa, self.fi_angles1_fixed, self.fi_angles1_pad)
        f_dih = self._gather(fd, self.fi_diheds_fixed, self.fi_diheds_pad)
        return torch.cat([self.feat_atypes, f_bonds, f_ang0, f_ang1, f_dih], dim=-1)

    def message_pass(self, fin: Tensor) -> Tensor:
        mv = self.max_valence
        nb0v = self.nb_connect[:, 0:mv - 1]
        nb1v = self.nb_connect[:, mv - 1:2 * (mv - 1)]
        nb0 = nb0v.sum(1); nb1 = nb1v.sum(1)
        H0 = (nb0 > 0).to(fin.dtype); H1 = (nb1 > 0).to(fin.dtype)
        term1 = fin[:, 0] * (1 - H0 * self.w - H1 * self.w).unsqueeze(1)
        term2 = self.w * torch.einsum('rn,rnk->rk', nb0v, fin[:, 1:mv]) / torch.clamp(nb0, min=1e-5).unsqueeze(1)
        term3 = self.w * torch.einsum('rn,rnk->rk', nb1v, fin[:, mv:2 * mv - 1]) / torch.clamp(nb1, min=1e-5).unsqueeze(1)
        return term1 + term2 + term3

    def forward(self, positions: Tensor, boxvectors: Optional[Tensor] = None) -> Tensor:
        pos = positions.to(self.box.dtype) * self.pos_scale
        if boxvectors is None:
            box = self.box                                   # already in model units (Angstrom)
        else:
            box = boxvectors.to(self.box.dtype) * self.pos_scale
        feats = self.features(pos, box)
        # fc0 (3 layers, tanh)
        h = torch.tanh(feats @ self.fc0_w0.T + self.fc0_b0)
        h = torch.tanh(h @ self.fc0_w1.T + self.fc0_b1)
        h = torch.tanh(h @ self.fc0_w2.T + self.fc0_b2)
        h = self.message_pass(h)
        # fc1 (2 layers, tanh)
        h = torch.tanh(h @ self.fc1_w0.T + self.fc1_b0)
        h = torch.tanh(h @ self.fc1_w1.T + self.fc1_b1)
        e_sub = h @ self.fcf_w.T + self.fcf_b
        return (self.weights @ e_sub[:, 0]) * self.sigma + self.mu


def main():
    npz = sys.argv[1]
    d = np.load(npz, allow_pickle=True)
    E_jax = float(d['E'])
    pos = torch.tensor(np.asarray(d['positions']), dtype=torch.float64)

    m = SGNNForce(npz).eval()
    with torch.no_grad():
        E_eager = float(m(pos))

    # TorchScript
    sm = torch.jit.script(m)
    with torch.no_grad():
        E_script = float(sm(pos))

    # scripted with explicit box arg (openmm-torch periodic path)
    with torch.no_grad():
        E_script_box = float(sm(pos, m.box))

    print(f"JAX (float64)      = {E_jax:.10f}")
    print(f"torch eager        = {E_eager:.10f}   |diff vs JAX|    = {abs(E_eager-E_jax):.3e}")
    print(f"torch scripted     = {E_script:.10f}   |diff vs eager|  = {abs(E_script-E_eager):.3e}")
    print(f"torch scripted+box = {E_script_box:.10f}   |diff vs eager|  = {abs(E_script_box-E_eager):.3e}")
    ok = abs(E_eager - E_jax) < 1e-8 and abs(E_script - E_eager) < 1e-12
    print("=> " + ("ALL MATCH" if ok else "MISMATCH"))

    # autograd forces through the scripted module (what openmm-torch uses)
    pg = pos.clone().requires_grad_(True)
    Eg = sm(pg); Eg.backward()
    print(f"scripted autograd force ok: grad {tuple(pg.grad.shape)}, max|F| = {pg.grad.abs().max():.4f}")

    # save the scripted artifact for openmm-torch
    out = npz[:-4] + '_scripted.pt' if npz.endswith('.npz') else npz + '_scripted.pt'
    sm.save(out)
    print("saved scripted model ->", out)


if __name__ == '__main__':
    main()
