# EventSNN — Gesture Recognition on DVS Event Camera Data

Spiking Neural Network for hand gesture classification on the IBM DVS128 Gesture Dataset. Trained with surrogate gradients (BPTT) via snnTorch.

**Result: 67% validation accuracy** on 11-class gesture recognition (chance: 9.1%).

---

## Why this is interesting

DVS (Dynamic Vision Sensor) cameras are fundamentally different from frame cameras: they fire asynchronous events when local pixel brightness changes, not at fixed frame rates. This makes them biologically plausible — the output resembles retinal ganglion cell activity more than a conventional video stream.

Pairing a DVS sensor with a Spiking Neural Network creates a doubly bio-inspired pipeline:
- **Input**: event-driven, sparse, temporally precise (like the retina)
- **Network**: LIF neurons with membrane dynamics and spike-based communication (like cortical neurons)

This project sits at the intersection of neuromorphic computing and computer vision, and connects directly to prior RSA work on visual cortex representations.

---

## Dataset

IBM DVS128 Gesture Dataset — 11 hand gestures recorded with a 128×128 DVS event camera.

Pre-processed `.npy` format available at: https://zenodo.org/records/8060604

Place `train.npy` and `test.npy` in `./data/dvs_gesture/` before running.

| Property | Value |
|---|---|
| Classes | 11 (hand gestures) |
| Sensor | DVS128 event camera |
| Input format | [N, T, C, H, W] binary spike frames |
| T (timesteps) | 20 |
| Spatial resolution | 2 × 32 × 32 |

---

## Model

3-layer fully-connected LIF network:

```
Input (2×32×32 = 2048) → FC → LIF → FC(512) → LIF → FC(128) → LIF → Output(11)
```

- **Neuron model**: Leaky Integrate-and-Fire (`snn.Leaky`, β=0.9)
- **Training**: BPTT with fast sigmoid surrogate gradient (slope=25)
- **Readout**: spike count over T timesteps (rate coding)
- **Loss**: cross-entropy on summed output spikes

Parameters: ~1.1M

---

## Results

| Metric | Value |
|---|---|
| Best Val Accuracy | 67% |
| Chance Level | 9.1% |
| Epochs | 10 |
| Optimizer | Adam (lr=2e-3, cosine decay) |

Training and validation accuracy converge without significant overfitting. The membrane potential heatmap shows clear differentiation between output neurons, confirming the network learns gesture-specific representations.

---

## Known Limitations

These are documented honestly in the notebook:

- **FC-only architecture** — ignores the spatial structure of DVS frames. A convolutional SNN (ConvSNN) would exploit 2D event patterns and likely reach 85–90%.
- **Rate coding** — discards precise spike timing. Temporal coding or population coding would better exploit the DVS camera's sub-millisecond temporal resolution.
- **Surrogate gradient mismatch** — the fast sigmoid approximation of the Heaviside function introduces a gradient estimation error that grows with network depth.
- **Dead neurons** — sparse DVS input (~5% active per timestep) can cause silent LIF neurons if threshold is too high. Mitigated here with threshold=1.0 and β=0.9.

---

## Usage

```bash
pip install snntorch torch numpy matplotlib
```

```bash
# With real DVS data
python -c "import nbformat; ..."  # or run in Jupyter

# Without dataset (synthetic fallback for pipeline testing)
# The notebook auto-detects missing data and uses synthetic DVS-style input
```

Run the notebook cell by cell, or use Kaggle/Colab with GPU.

---

## Connection to Other Projects

This project is part of a broader NeuroAI portfolio:

- **Predictive Coding RSA** — PC network vs. SNN vs. ResNet on THINGS-fMRI, Spearman RSA against human visual cortex (V1–IT)
- **nanoGPT Ablation** — GPT depth/head ablation on FineWeb-Edu
- **Maturaarbeit** — MLP from scratch for Swiss referendum turnout prediction

The DVS-SNN project bridges the gap between the rate-coded SNN in the RSA project and a more realistic neuromorphic setting where the input itself is spike-based.

---

## References

Amir, A., et al. (2017). A low power, fully event-based gesture recognition system. *CVPR*.

Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine*, 36(6), 51–63.

Mahowald, M., & Douglas, R. (1991). A silicon neuron. *Nature*, 354(6354), 515–518.

---

## Author

Nils — pre-university project, Switzerland, 2026.
