## Attention network version

DPElwFuse:
59, 121, 65, 73

1. Mat (id: 59) = Dot prod [2, 128] x [128, 2] --> (Mat (id: 55)
2. Mat (id: 121) = BinaryOp.MULT (4) --> (Mat (id: 59), C (val: 0.063758715412299
3. Mat (id: 65) = UnaryOp.EXP2 (4) --> (Mat (id: 121))
4. Mat (id: 73) = BinaryOp.MULT (4) --> (Mat (id: 65), Mat (id: 70))

ReduceElwFuse:
69, 70, 73

1. Mat (id: 69) = ReduceOp.SUM on dim: -1 --> Mat (id: 65)
    ** THIS IS PROBLEM **, requires 65.
2. Mat (id: 70, access: Global) = UnaryOp.RECIP (2) --> (Mat (id: 69, access: Global))
4. Mat (id: 73) = BinaryOp.MULT (4) --> (Mat (id: 65), Mat (id: 70))

In this case, it should **remove them** altogether

**NOTE THAT I'VE BEEN UNRESTRICT THE SIZE REQUIREMENT. ANYWAY TO CIRCUMSTANCE IT?**