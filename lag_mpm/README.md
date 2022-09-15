# Mesh-Grid Hybrid Simulation

A soft body simulation using material point method with corotated linear elastic materials. The force model is evaluated from a Lagrangian point of view using meshes. (A MUCH larger version is in our video).


## How to run:

```
mkdir results
python3 run.py --output ./results
python3 render_particles.py -b 0 -e 100 -r 256 -M 12 -i ./results/particles -o ./results/output -f
```

## Results

After executing `run.py`, the immediate visualization results are in `results/armadillo`, and simulation particle data are in `results/particles`. After executing `render_particles.py`, the images are in `results/output`.