# Quantum-Relativity-The-Universal-Preservation-of-Information
Any positive spectral radius can be transported across any admissible carrier.


# Universal Q-R / Q^alpha Transport Kernel

This is the coded cross-domain algebraic transport law.

## Core theorem

For any positive observable radius sequence \(R_i>0\) and any admissible carrier \(q_i>1\),

\[
R_i=s_*q_i^{\alpha_i(q)}
\]

where

\[
s_* = \exp\left(\frac1N\sum_i\log R_i\right),
\qquad
\alpha_i(q)=\frac{\log(R_i/s_*)}{\log q_i}.
\]

## Jump theorem

If the carrier jumps

\[
q_i\rightarrow q'_i,
\]

then the exponent jumps contravariantly:

\[
\alpha'_i=\alpha_i\frac{\log q_i}{\log q'_i}.
\]

Therefore

\[
s_*(q'_i)^{\alpha'_i}=s_*q_i^{\alpha_i}=R_i.
\]

## Why this is universal

The kernel does not know whether \(R\) came from gravitational-wave events, CMB polarization, pulsar timing, quantum readouts, medical imaging, radar, finance, communications, or any other positive spectral signal. The same algebraic transport law applies.

## CLI

```bash
python qr_transport_cli.py --csv qpower_raw_data.csv --output-json qr_transport_certificate.json
```

CSV must include:

```text
dataset,R,qbase
```

Optional columns:

```text
index_value,lambda_Q,rank,pdet,effective_rank
```

## Python

```python
import numpy as np
from qr_transport import path_incidence_qbase, transport, perturb_carrier, jump_transport

R = np.array([10, 12, 8, 15, 11], dtype=float)
q, lam, rank = path_incidence_qbase(len(R))

state = transport(R, q)
q2 = perturb_carrier(q, eps=1e-2)
jump = jump_transport(state, q2)

print(state.metric())
print(jump.metric())
```
