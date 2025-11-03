#! /usr/bin/env python3

import sys
import numpy as np
import model.MPNN as MPNN
import dataloader_eval.dataloader as dataloader
import dataloader_eval.cudaloader as cudaloader
import jax
from jax import vmap, jit
import jax.numpy as jnp
import orbax.checkpoint as oc
from src.data_config import ModelConfig
from src.read_json import load_config

# 示例：读取配置文件
full_config = load_config("full_config.json")
if full_config.jnp_dtype=='float64':
    jax.config.update("jax_enable_x64", True)

if full_config.jnp_dtype=='float32':
    jax.config.update("jax_default_matmul_precision", "highest")

data_load = dataloader.Dataloader(full_config.maxneigh, full_config.batchsize, initpot=full_config.initpot, ncyc=full_config.ncyc, cutoff=full_config.cutoff, datafolder=full_config.datafolder, ene_shift=full_config.ene_shift, force_table=full_config.force_table, dipole_table=full_config.dipole_table, cross_val=full_config.cross_val, jnp_dtype=full_config.jnp_dtype, key=full_config.seed, ntrain=full_config.ntrain, eval_mode=True)
# generate random data for initialization

#ntrain = data_load.ntrain
numatoms = data_load.numatoms[:full_config.ntrain]
ntrain = full_config.ntrain #jnp.sum(numatoms)
ntotatoms = jnp.sum(numatoms)
nforce = np.sum(numatoms) * 3

nprop = 1
prop_length = ntotatoms
if full_config.stress_table:
    nprop = 3
    prop_length = jnp.array(np.array([ntotatoms, nforce, 9*ntrain]))
elif full_config.force_table and (not full_config.dipole_table):
    nprop = 2
    prop_length = jnp.array(np.array([ntotatoms, nforce]))
elif full_config.force_table and full_config.dipole_table:
    nprop = 3
    prop_length = jnp.array(np.array([ntotatoms, nforce, 3*ntrain]))
elif full_config.dipole_table:
    nprop = 2
    prop_length = jnp.array(np.array([ntotatoms, 3*ntrain]))


data_load = cudaloader.CudaDataLoader(data_load, queue_size=full_config.queue_size)


#==============================Equi MPNN==============================================================
options = oc.CheckpointManagerOptions()
ckpt = oc.CheckpointManager(full_config.ckpath, options=options)
step = ckpt.latest_step()
restored = ckpt.restore(step)
params = restored["params"]
model_config = restored["config"]

config = ModelConfig(**model_config)

model = MPNN.MPNN(config)
#=================

if full_config.stress_table:
    def get_force_stress(params, coor, field, cell, disp_cell, neighlist, shiftimage, center_factor, species):
        ene, (force, stress) = jax.value_and_grad(model.apply, argnums=[1, 4])(params, coor, field, cell, disp_cell, neighlist, shiftimage, center_factor, species)
        volume = jnp.dot(cell[0], jnp.cross(cell[1], cell[2]))
        return ene, force, stress/volume*jnp.array(full_config.stress_sign)
    vmap_model = vmap(get_force_stress, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0))
elif full_config.force_table and (not full_config.dipole_table):
    vmap_model = vmap(jax.value_and_grad(model.apply, argnums=1), in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0))
elif full_config.force_table and full_config.dipole_table:
    def get_force_dipole(params, coor, field, cell, disp_cell, neighlist, shiftimage, center_factor, species):
        ene, (force, dipole) = jax.value_and_grad(model.apply, argnums=[1, 2])(params, coor, field, cell, disp_cell, neighlist, shiftimage, center_factor, species)
        return ene, force, dipole*jnp.array(full_config.dipole_sign)
    vmap_model = vmap(get_force_dipole, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0))
elif full_config.dipole_table:
    def get_dipole(params, coor, field, cell, disp_cell, neighlist, shiftimage, center_factor, species):
        ene, dipole = jax.value_and_grad(model.apply, argnums=2)(params, coor, field, cell, disp_cell, neighlist, shiftimage, center_factor, species)
        return ene, dipole*jnp.array(full_config.dipole_sign)
    vmap_model = vmap(get_dipole, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0))
else:
    def get_energy(params, coor, field, cell, disp_cell, neighlist, shiftimage, center_factor, species):
        return model.apply(params, coor, field, cell, disp_cell, neighlist, shiftimage, center_factor, species),
    vmap_model = vmap(get_energy, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0))


def make_loss(pes_model, nprop):

    def get_loss(params, coor, field, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop):

        nnprop = pes_model(params, coor, field, cell, disp_cell, neighlist, shiftimage, center_factor, species)
        if full_config.stress_table:
            abpot, abforce, abstress = abprop
            nnpot, nnforce, nnstress = nnprop
            loss1 = jnp.sum(jnp.abs(abpot - nnpot)) 
            loss2 = jnp.sum(jnp.abs(abforce - nnforce)) 
            loss3 = jnp.sum(jnp.abs(abstress - nnstress)) 
            ploss = jnp.stack([loss1, loss2, loss3])
        elif full_config.force_table and (not full_config.dipole_table):
            abpot, abforce = abprop
            nnpot, nnforce = nnprop
            loss1 = jnp.sum(jnp.abs(abpot - nnpot)) 
            loss2 = jnp.sum(jnp.abs(abforce - nnforce))
            ploss = jnp.stack([loss1, loss2])
        elif full_config.force_table and full_config.dipole_table:
            abpot, abforce, abdipole = abprop
            nnpot, nnforce, nndipole = nnprop
            delta_dipole = abdipole - nndipole
            int_modulo = jnp.trunc(jnp.einsum("ij, ijk ->ik", delta_dipole, jnp.linalg.inv(cell)))
            modulo_dipole = jnp.einsum("ij, ijk -> ik", int_modulo, cell)
            delta_dipole = delta_dipole - jax.lax.stop_gradient(modulo_dipole)
            loss1 = jnp.sum(jnp.abs(abpot - nnpot)) 
            loss2 = jnp.sum(jnp.abs(abforce - nnforce)) 
            loss3 = jnp.sum(jnp.abs(delta_dipole))
            ploss = jnp.stack([loss1, loss2, loss3])
        elif full_config.dipole_table:
            abpot, abdipole = abprop
            nnpot, nndipole = nnprop
            delta_dipole = abdipole - nndipole
            int_modulo = jnp.trunc(jnp.einsum("ij, ijk ->ik", delta_dipole, jnp.linalg.inv(cell)))
            modulo_dipole = jnp.einsum("ij, ijk -> ik", int_modulo, cell)
            delta_dipole = delta_dipole - jax.lax.stop_gradient(modulo_dipole)
            loss1 = jnp.sum(jnp.abs(abpot - nnpot)) 
            loss2 = jnp.sum(jnp.abs(delta_dipole))
            ploss = jnp.stack([loss1, loss2])
        else:
            abpot, = abprop
            nnpot, = nnprop
            ploss = jnp.sum(jnp.abs(abpot - nnpot))
        return ploss





    return get_loss
 
value_fn = make_loss(vmap_model, nprop)       

def val_loop(nstep):
    def get_loss(params, coor, field, cell, neighlist, shiftimage, center_factor, species, abprop, ploss_out):
        def body(i, carry):
            params, coor, field, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, ploss_fn = carry
            inabprop = (iabprop[i] for iabprop in abprop)
            ploss = value_fn(params, coor[i], field[i], cell[i], disp_cell[i], neighlist[i], shiftimage[i], center_factor[i], species[i], inabprop)
            ploss_fn = ploss_fn + ploss
            return params, coor, field, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, ploss_fn

        disp_cell = jnp.zeros_like(cell)
        params, coor, field, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, ploss_out = \
        jax.lax.fori_loop(0, nstep, body, (params, coor, field, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, ploss_out))
        return ploss_out
    return jax.jit(get_loss)


val_ens = val_loop(full_config.ncyc)
ploss_val = jnp.zeros(nprop)        
num = 0
for data in data_load:
    ploss_val1 = ploss_val
    coor, field, cell, neighlist, shiftimage, center_factor, species, abprop = data
    ploss_val = val_ens(params, coor, field, cell, neighlist, shiftimage, center_factor, species, abprop, ploss_val)
    print(num, ploss_val - ploss_val1)
    num+=1
    
ploss_val = ploss_val / prop_length

print(ploss_val)


