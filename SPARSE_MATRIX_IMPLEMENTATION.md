# Sparse Matrix Implementation - Summary

## Overview

Successfully implemented sparse matrix support for the LP optimizer, enabling optimization of problems with 33,120+ timesteps without memory errors.

## Results

### Memory Reduction

| Timesteps | Dense Memory | Sparse Memory | Reduction Factor | Sparsity |
|-----------|-------------|---------------|------------------|----------|
| 10 | 0.00 MiB | 0.00 MiB | 5.4x | 87.7% |
| 100 | 0.23 MiB | 0.00 MiB | 50.4x | 98.7% |
| 1,000 | 22.89 MiB | 0.05 MiB | 500.4x | 99.9% |
| 10,000 | 2.29 GiB | 0.46 MiB | 5,000.4x | 99.99% |
| 33,120 | 24.52 GiB | 1.52 MiB | 16,560.4x | 99.996% |

### Full-Scale Problem (33,120 timesteps)

- **Battery A_eq matrix**: Shape (33,120 × 99,360)
  - Total elements: 3,290,803,200
  - Non-zeros: 132,477
  - Sparsity: 99.99597% zeros
  - Memory: 1.52 MiB (vs 24.52 GiB dense)

- **Aggregated A_eq matrix**: Shape (66,240 × 165,600)
  - Total elements: ~11 billion
  - Non-zeros: 264,957
  - Sparsity: 99.998% zeros
  - Memory: ~3.6 MiB (vs ~81 GiB dense)

## Implementation Changes

### 1. Battery Asset (`flex_model/assets/battery.py`)

**Changed**: `get_linear_model()` method (lines 796-908)

- **Before**: Built constraints as list of dense numpy arrays
- **After**: Uses `scipy.sparse.lil_matrix` for efficient construction
- **Output**: CSC sparse matrix (`A_eq.tocsc()`)

```python
# Old approach
constraints_eq = []
for t in range(n_timesteps):
    row = np.zeros(n_vars)  # Dense!
    row[idx] = value
    constraints_eq.append(row)
A_eq = np.array(constraints_eq)

# New approach
A_eq = sparse.lil_matrix((n_eq, n_vars))
for t in range(n_timesteps):
    A_eq[t, idx] = value
A_eq = A_eq.tocsc()
```

### 2. LP Optimizer (`flex_model/optimization/lp_optimizer.py`)

**Changed**: `solve()` method constraint aggregation (lines 123-200)

- **Before**: Expanded each asset's constraints into global-width dense rows
- **After**: Uses `sparse.block_diag()` and `sparse.vstack()`
- **Key improvement**: Block-diagonal structure preserved

```python
# Old approach
for asset in self.assets:
    for i in range(asset.A_eq.shape[0]):
        row = np.zeros(total_vars)  # Dense!
        row[var_offset:var_offset + asset.n_vars] = asset.A_eq[i, :]
        constraints_eq.append(row)

# New approach
asset_eq_blocks = [asset.A_eq for asset in self.assets]
A_eq_assets = sparse.block_diag(asset_eq_blocks, format='csc')
balance_eq = sparse.lil_matrix((n_timesteps, total_vars))
# ... build balance constraints ...
A_eq = sparse.vstack([A_eq_assets, balance_eq], format='csc')
```

### 3. Linear Model (`flex_model/optimization/linear_model.py`)

**Changed**: Type annotations for `A_eq` and `A_ub`

- **Before**: `Optional[np.ndarray]`
- **After**: `Optional[Union[np.ndarray, sparse.spmatrix]]`
- **Benefit**: Supports both dense and sparse matrices (backward compatible)

## Testing

All 45 existing unit tests pass:
```bash
pytest tests/ -v
# 45 passed in 1.57s
```

## Performance Verification

Created verification scripts:
- `test_sparse_performance.py`: Tests 10, 100, 1,000, 10,000 timesteps
- `test_large_scale.py`: Tests full 33,120 timesteps

Both demonstrate:
1. ✓ Sparse matrices are used (type checking)
2. ✓ Massive memory reduction (up to 16,560x)
3. ✓ Optimization completes successfully
4. ✓ Correct results (feasible solutions found)

## Backward Compatibility

The implementation is **fully backward compatible**:
- Accepts both dense and sparse matrices
- Existing code passing dense arrays continues to work
- Type hints use `Union[np.ndarray, sparse.spmatrix]`

## scipy HiGHS Support

The scipy `linprog` function with `method='highs'` natively supports sparse matrices, so no changes were needed to the solver call itself.

## Files Modified

1. `flex_model/assets/battery.py` - Sparse constraint construction
2. `flex_model/optimization/lp_optimizer.py` - Sparse aggregation
3. `flex_model/optimization/linear_model.py` - Type hints

## Next Steps

To test with real data:
```bash
# Option 1: Use the dashboard
streamlit run examples/battery_vs_market/dashboard.py

# Option 2: Run the LP optimizer directly
python examples/battery_vs_market/lp_optimizer.py

# Option 3: Use the verification scripts
python test_large_scale.py
```

## Key Takeaways

1. **Memory reduction**: From ~81 GiB to ~3.6 MiB (23,000x) for 33,120 timesteps
2. **Sparsity**: 99.998% of constraint matrix elements are zeros
3. **Performance**: Optimization completes in seconds instead of causing MemoryError
4. **Scalability**: Can now handle year-long horizons at 15-minute resolution
