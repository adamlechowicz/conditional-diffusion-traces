**MEMO**
**To:** Jules
**Subject:** Research Proposal: Spatiotemporal Super-Resolution of Renewable Generation via Conditional Diffusion

### **1. Motivation & Problem Formulation**
Current systems research into power-grid operations, local microgrids, and carbon-aware scheduling is bottlenecked by the availability of high-resolution renewable generation traces. While low-resolution data (hourly) provides accurate macro-weather trends, it smooths out the high-frequency transients (e.g., cloud-edge effects, wind micro-turbulence) that dictate transient system states, battery degradation, and job queuing dynamics.

We frame this as a spatiotemporal super-resolution task. Given a low-resolution meteorological conditioning vector $\mathbf{c}$ and static site metadata $\mathbf{s}$, we want to sample from the conditional distribution of high-resolution generation traces $P(\mathbf{x}_{1:T} \mid \mathbf{c}, \mathbf{s})$. Pointwise predictive models minimize MSE, which averages out stochasticity. Diffusion models, conversely, learn the score of the data distribution, allowing us to hallucinate physically realistic, high-frequency transients that respect the low-frequency boundary conditions.

### **2. Proposed Architecture: Physics-Guided Conditional Diffusion**
I propose a latent diffusion model conditioned on physical priors. Generating $L=1440$ steps (one day of 1-minute data) per forward pass is computationally heavy for standard attention mechanisms.

* **Physical Prior (The Mean Reversion Baseline):** We do not predict raw power output. For solar, we use standard ephemeris algorithms to generate a deterministic 1-minute Clear Sky Irradiance (CSI) curve. The model predicts the *Clearness Index* ($K_t$), representing atmospheric stochasticity.
* **The Denoiser Backbone:** Instead of a standard U-Net with self-attention ($O(L^2)$), we use a 1D U-Net with **State Space Model (SSM) blocks** (e.g., Mamba or S4). This gives $O(L)$ scaling for long context windows, crucial if we want to expand the generation horizon beyond a single day while remaining computationally feasible for local inference.
* **Conditioning Strategy:** * $\mathbf{c}_{\text{weather}}$: Hourly ERA5/NSRDB variables (temperature, cloud cover, wind speed) interpolated to 1-minute.
    * $\mathbf{s}_{\text{site}}$: Embedding of spatial coordinates, azimuth, tilt, and turbine/panel capacity.
    * We inject these into the SSM blocks via cross-attention or adaptive layernorm.

### **3. Datasets**
To train the model, we need paired low-resolution and high-resolution data.
* **Target (High-Res Ground Truth):** * *Solar:* NREL Oahu Solar Measurement Grid (1-second data, downsampled to 1-minute). NREL SRRL (Solar Radiation Research Laboratory) baseline measurement system.
    * *Wind:* NREL NWTC (National Wind Technology Center) M2 tower telemetry (1-Hz data).
* **Conditioning (Low-Res Inputs):**
    * NREL NSRDB (hourly/half-hourly solar irradiance).
    * ECMWF ERA5 (hourly reanalysis data for wind/weather).

### **4. Baselines**
We need to demonstrate that this is superior to both naive methods and state-of-the-art time-series foundation models.
1.  **Interpolation + Noise:** Spline interpolation of ERA5 combined with an auto-regressive Gaussian noise model.
2.  **Physical/Stochastic Simulation:** The NREL System Advisor Model (SAM) using its built-in Markov chain downscaling stochastic weather generator.
3.  **Time-Series Foundation Models:** Zero-shot prompting of Chronos (Amazon) or TimesFM (Google), conditioned on the hourly data.
4.  **Generative Adversarial Networks:** A 1D WGAN-GP trained for time-series super-resolution (common in recent meteorology literature).

### **5. Evaluation Plan**
Standard forecasting metrics (RMSE, MAE) are fundamentally flawed here; a temporally shifted but physically realistic cloud cover event will be penalized heavily. We must evaluate the *distributional* and *frequency-domain* realism.

* **Statistical Realism:** * Continuous Ranked Probability Score (CRPS).
    * Wasserstein distance on the distribution of 1-minute ramp rates ($x_t - x_{t-1}$).
* **Frequency Domain:** * Compare the Power Spectral Density (PSD) of the generated traces against the ground truth. This proves we are capturing the correct spectrum of micro-turbulence/cloud-transients.
* **Downstream Systems Utility:** * Simulate a basic carbon-aware job scheduler (e.g., delaying batch jobs based on local solar availability). We will compare the queue wait times and carbon-intensity outcomes when using our synthetic 1-minute traces versus the ground-truth 1-minute traces.

---

Do you want to focus the initial prototype strictly on solar irradiance since the clear-sky physical prior is mathematically cleaner, or should we tackle wind and solar simultaneously to prove the generalized capability of the denoiser?
