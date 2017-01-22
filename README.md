# Different approaches to speed up disparity calculation

Note: The code for the stereolab application is required to run these code fragments.

![Elephant](https://github.com/Tetr4/ParallelDisparityCalculation/blob/master/paper/images/elefant.png?raw=true)

## Factors
- Runtime Environment:
  - Just-in-time compilation (Java)
  - Ahead-of-time compilation (C)
  - On graphics processing unit (CUDA)
- Concurrency Strategy
  - Single thread (no concurrency)
  - 1 thread per pixel
  - 1 thread per image-chunk

## Results
### Hardware
- Intel Xeon E3-1230 v3 (3,30GHz, 4 Cores, 8 Threads)
- GeForce GTX 770

### Parameters
- Image-Resolution: 681 × 681 (see image above)
- Window: 15 × 15
- Tau-Max: 40

![Results](https://github.com/Tetr4/ParallelDisparityCalculation/blob/master/paper/images/graph.png?raw=true)
