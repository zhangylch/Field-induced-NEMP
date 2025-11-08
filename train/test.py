# --- 1. 安装 JAX (在 Colab 中需要) ---
# !pip install --upgrade jax jaxlib

import jax
import jax.numpy as jnp
from functools import partial

print(f"JAX 版本: {jax.__version__}")

# --- 2. 定义一个非常简单的能量函数 ---
def energy_fn(x, e):
    """
    一个极简的玩具能量函数
    - x: "坐标" 数组, 形状 (2,)
    - e: "电场" 数组, 形状 (2,)
    
    能量 E = (0.5 * (x[0]² + x[1]²)) - (x[0]*e[0] + x[1]*e[1])
    """
    E_harm = 0.5 * jnp.sum(x**2)  # 简谐势
    E_field = -jnp.sum(x * e)    # 耦合项
    return E_harm + E_field

# --- 3. 定义高效的 "总计算" 函数 (与之前相同) ---

# JIT 编译这个总函数以启用 CSE (公共子表达式消除)
@jax.jit
def get_all_quantities(x, e):
    """
    在一个 JIT 函数中计算所有需要的量。
    JAX/XLA 将优化这里的重复计算。
    """
    
    # --- 1. 获取能量 (E) 和力 (F) ---
    # value_and_grad 会同时返回 E 和 F_grad (dE/dx)
    # argnums=0 表示对第0个参数 'x' 求导
    E, F_grad = jax.value_and_grad(energy_fn, argnums=0)(x, e)
    
    # 力是能量对坐标的负梯度
    F = -F_grad
    
    # --- 2. 获取 Born Charges (Z*) ---
    # argnums=(0, 1) 表示计算 'x' 和 'e' 之间的所有二阶导数
    # H = ( (d²E/dxdx, d²E/dxde),
    #       (d²E/dedx, d²E/dede) )
    H = jax.hessian(energy_fn, argnums=(0, 1))(x, e)
    
    # 提取 Z* = d²E / dx de
    # H[0] = (d²E/dxdx, d²E/dxde)
    # H[0][1] = d²E/dxde
    #Z_star = H[0][1]
    
    # --- 3. 返回所有量 ---
    return E, F, H[0][1]

# --- 4. 运行和验证 ---

# 定义输入数据
x_input = jnp.array([1.0, 2.0])  # 形状 (2,)
e_input = jnp.array([0.1, 0.2])  # 形状 (2,)

# 第一次调用会触发 JIT 编译
print("--- 正在 JIT 编译并执行... ---")
E_out, F_out, H = get_all_quantities(x_input, e_input)

# 确保计算完成以便我们能看到清晰的输出
E_out.block_until_ready()
print("--- 计算完成! ---")

# --- 5. 打印结果 ---

print(f"\n输入 'x' [形状 {x_input.shape}]: {x_input}")
print(f"输入 'e' [形状 {e_input.shape}]: {e_input}")

print("\n--- 高效获取的所有量 ---")

print(f"\n能量 (E):")
print(E_out)

print(f"\n力 (F) [形状 {F_out.shape}]:")
print(F_out)

print(f"\nBorn Charges (Z*) [形状 {H.shape}]:")
print(H)


# --- 6. 手动验证结果 ---
# E = 0.5*(x0²+x1²) - (x0*e0 + x1*e1)
# 当 x=[1, 2], e=[0.1, 0.2]:
# E = 0.5*(1+4) - (1*0.1 + 2*0.2) = 2.5 - (0.1 + 0.4) = 2.0

# dE/dx0 = x0 - e0
# dE/dx1 = x1 - e1
# F = -[dE/dx0, dE/dx1] = -[1-0.1, 2-0.2] = -[0.9, 1.8] = [-0.9, -1.8]

# Z*_00 = d(dE/dx0)/de0 = d(x0 - e0)/de0 = -1
# Z*_01 = d(dE/dx0)/de1 = d(x0 - e0)/de1 = 0
# Z*_10 = d(dE/dx1)/de0 = d(x1 - e1)/de0 = 0
# Z*_11 = d(dE/dx1)/de1 = d(x1 - e1)/de1 = -1
# Z* = [[-1, 0], [0, -1]]

print("\n--- 手动验证 ---")
print(f"E (预期): 2.0,            (JAX 计算): {E_out}")
print(f"F (预期): [-0.9 -1.8],    (JAX 计算): {F_out}")
print(f"Z* (预期): [[-1, 0], [0, -1]], (JAX 计算): \n{H}")
