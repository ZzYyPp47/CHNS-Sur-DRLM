# CHNS-Sur-DRLM

Python code of paper: *Efficient FEM fully discrete schemes for the flow-coupled binary phase-field surfactant system based on the DRLM method by Yunpeng Zhu,Mengchun Yuan,Xiaohan Cheng,Qi Li*

If you use the code, please cite:

```latex

```

# Steps:

You can follow the steps below to run this program:

- Temporal accuracy test:

```bash
OMP_NUM_THREADS=1 python ex_time_accuracy.py
```

- Colliding under the shear flow with different surfactant concentrations:

```bash
OMP_NUM_THREADS=1 python ex_two_circles.py
```

