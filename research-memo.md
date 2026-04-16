**MEMO**
**To:** Jules
**Subject:** Research Proposal: Spatiotemporal Super-Resolution of Solar Generation via Retrieval-Augmented Diffusion

### **1. Motivation & Problem Formulation**
Current systems research into power-grid operations, local microgrids, and carbon-aware scheduling is bottlenecked by the availability of high-resolution renewable generation traces. Low-resolution data (hourly) smooths out the high-frequency transients (e.g., cloud-edge effects) that dictate transient system states and queuing dynamics.

We frame this as a spatiotemporal super-resolution task. Given a low-resolution meteorological conditioning vector $\mathbf{c}$ and static site metadata $\mathbf{s}$, we want to sample from the conditional distribution of high-resolution generation traces $P(\mathbf{x}_{1:T} \mid \mathbf{c}, \mathbf{s})$. To avoid the extreme computational cost and "regression to the mean" smoothing typical of autoregressive Time-Series Foundation Models (TSFMs), we propose a hybrid system: **Time-Series RAG via Stochastic Differential Editing (SDEdit)**.

### **2. Proposed Architecture: Time-Series RAG**
Instead of hallucinating high-frequency transients from pure Gaussian noise, we retrieve a physically realistic "template" trace and use a conditional diffusion model to edit it to fit local constraints. This drastically bounds the generative search space and allows the system to run efficiently on local hardware.

* **Phase 1: The Canonical Shape Library (Vector DB)**
    * We extract 5+ years of 1-minute NREL solar data, chunk it into daily traces ($L=1440$), and use k-means clustering to extract the $K=10,000$ most representative daily "shapes."
    * This creates an ultra-lightweight (~57 MB) FAISS/ChromaDB index that easily ships to edge devices.
* **Phase 2: Inference-Time Retrieval**
    * The user inputs coordinates and low-resolution weather covariates (e.g., ERA5 hourly cloud cover). We query the vector database to retrieve the nearest-neighbor 1-minute template trace, $x_{\text{ref}}$.
* **Phase 3: SDEdit (Forward Noising)**
    * Rather than starting the reverse diffusion from $t=T$ (pure noise), we inject partial Gaussian noise into the retrieved template up to an intermediate timestep $t_0 \in (0, T)$. This preserves the baseline Power Spectral Density (the "shape" of the high-frequency micro-turbulence).
* **Phase 4: Reverse Denoising (The Mamba-UNet Backbone)**
    * We run the reverse SDE from $t_0$ to $0$, conditioned on the site metadata $\mathbf{s}_{\text{site}}$ (capacity, azimuth, shading).
    * **Architecture:** To handle the $L=1440$ sequence length without the $O(L^2)$ transformer bottleneck, we use a 1D U-Net where the deep bottleneck layers are replaced with **State Space Model (SSM)** blocks (e.g., Mamba). This achieves $O(L)$ scaling, making local generation highly performant.

### **3. Datasets**
* **Target (High-Res Ground Truth):** NREL Oahu Solar Measurement Grid (downsampled to 1-minute) and NREL SRRL baseline measurements.
* **Conditioning (Low-Res Inputs):** NREL NSRDB (hourly irradiance) and ECMWF ERA5 (hourly reanalysis data).

### **4. Baselines**
1.  **Interpolation + Noise:** Spline interpolation of ERA5 combined with an auto-regressive Gaussian noise model.
2.  **Physical Simulation:** NREL System Advisor Model (SAM) using its built-in Markov chain downscaler.
3.  **Standard Diffusion:** CSDI (Conditional Score-based Diffusion Models for Imputation) trained from scratch without retrieval.
4.  **TSFMs:** Zero-shot conditional prompting of Chronos or TimesFM.

### **5. Evaluation Plan**
Standard pointwise metrics (RMSE, MAE) heavily penalize temporally shifted but physically realistic stochasticity. We will focus on:
* **Distributional Realism:** Continuous Ranked Probability Score (CRPS) and Wasserstein distance on 1-minute ramp rates ($x_t - x_{t-1}$).
* **Frequency Domain:** Comparing the Power Spectral Density (PSD) of generated traces against the ground truth to verify the correct spectrum of cloud-edge transients.
* **Downstream Utility:** Simulating a carbon-aware batch job scheduler, comparing queue wait times when provisioned with our synthetic traces versus ground-truth traces.
