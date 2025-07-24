# QuantumEnergyDispatch

## Overview

The QuantumEnergyDispatch project implements a quantum computing-based energy dispatch optimization model to manage power allocation for a critical backup system (CBSs) in Wollongong, Australia, over the period of 10–20 January 2025. This repository replaces a traditional rule-based dispatcher with a Quantum Annealing approach using D-Wave's quantum computing framework, optimizing the utilization of renewable energy sources (solar and wind), energy storage systems (ESS), and grid interactions. The model addresses constraints such as severe weather periods and ESS state-of-charge (SOC) limits, producing detailed visualizations and dispatch summaries.

The original data and plotting infrastructure are preserved, with the quantum optimization integrated to enhance efficiency and scalability. This project serves as a proof-of-concept for applying quantum computing to energy management systems.

## Features

- Quantum Annealing-based dispatcher using D-Wave’s Ocean SDK.
- Optimization of power allocation across solar, wind, ESS, and grid imports/exports.
- Support for severe weather constraints (15–18 Jan 2025) and pre-event ESS charging.
- Multi-panel visualization of power flows (CBSs, Grid, Solar, Wind, ESS).
- Comprehensive dispatch summary with renewable energy usage and grid balance metrics.
- Compatible with historical power generation and demand data in CSV format.

## Prerequisites

- Python 3.8 or higher.
- Access to D-Wave Leap cloud service (API token required for quantum annealing).
- Required Python packages:
  - `pandas`
  - `matplotlib`
  - `numpy`
  - `dwave-ocean-sdk` (including `dwave-system` and `dimod`)

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/QuantumEnergyDispatch.git
   cd QuantumEnergyDispatch
   ```

2. **Install Dependencies**
   Ensure you have `pip` installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   Note: Create a `requirements.txt` file with the following content:
   ```
   pandas
   matplotlib
   numpy
   dwave-ocean-sdk
   ```

3. **Configure D-Wave Access**
   - Obtain an API token from the [D-Wave Leap](https://cloud.dwavesys.com/leap/) portal.
   - Set the token as an environment variable:
     ```bash
     export DWAVE_API_TOKEN='your_api_token_here'
     ```
   - Alternatively, configure it in your Python script or a `.env` file using a library like `python-dotenv`.

4. **Prepare Data**
   - Place the input CSV files in the `Data/Test` directory:
     - `PowerGeneration_Solar__10_20_Jan_2025.csv`
     - `PowerGeneration_Wind__10_20_Jan_2025.csv`
     - `PowerDemand__CBSs_Wollongong_specific_area__10_20_Jan_2025.csv`
   - Ensure the `myLibs.backupPowerSystems` module (containing `EnergyStorageSystem`) is available in your Python path.

## Usage

1. **Run the Script**
   Execute the main script to perform the quantum optimization and generate visualizations:
   ```bash
   python main.py
   ```
   Note: Replace `main.py` with the name of your script file if different (e.g., `quantum_dispatch.py`).

2. **Output**
   - A dispatch summary will be printed to the console, detailing renewable energy usage, grid imports/exports, and SOC at key timestamps.
   - A figure (`Wollongong_Resilient_Power_System_10_20_Jan_2025.png`) will be saved in the `Figures` directory, showing five-panel plots of CBSs, Grid, Solar, Wind, and ESS power flows.

3. **Customization**
   - Adjust `power_levels` and `num_bits` in the quantum dispatcher (Section 3) to refine power discretization based on your data range.
   - Tune weights (`w1`, `w2`, `w3`, `w4`) in the QUBO formulation to prioritize objectives (e.g., minimize grid imports during severe weather).
   - Modify the severe weather window (`t_sev_start`, `t_sev_end`) or ESS parameters as needed.

## File Structure

- `main.py`: Main script containing the quantum annealing dispatcher and visualization code.
- `Data/Test/`: Directory for input CSV files.
- `Figures/`: Directory for output plots.
- `myLibs/`: Directory for custom library files (e.g., `backupPowerSystems.py`).
- `requirements.txt`: List of Python dependencies.

## Limitations

- Requires access to D-Wave Leap, with potential qubit limitations affecting problem size.
- Power discretization introduces approximation errors; increase `num_bits` for higher precision.
- Performance depends on quantum hardware availability and may be slower than rule-based methods for small datasets.

## Future Enhancements

- Extend the QUBO to optimize multiple timestamps simultaneously for temporal dependencies.
- Integrate gate-based quantum algorithms (e.g., QAOA via Qiskit) for future hardware.
- Incorporate real-time data updates via web APIs or X posts for dynamic dispatch.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with detailed descriptions of changes. Ensure code adheres to the project’s style guidelines and includes appropriate tests.

## License



## Contact

For questions or support, please open an issue on this repository or contact [your email or username] at [your contact information].

## Acknowledgments

- D-Wave Systems for providing quantum annealing resources via Leap.
- The original rule-based model developers for the foundational data and visualization framework.
```

### Notes for Implementation
1. **File Naming**: Replace `main.py` with the actual name of your script file. If your script is unnamed or split across multiple files, adjust the README accordingly.
2. **Data Path**: Update the `Data/Test` path if your CSV files are located elsewhere.
3. **License**: Add an appropriate open-source license (e.g., MIT, Apache 2.0) or remove the section if not applicable.
4. **Contact Information**: Replace placeholders (e.g., `[your email or username]`) with your actual details.
5. **GitHub URL**: Update the `git clone` URL with your repository’s address (e.g., `https://github.com/yourusername/QuantumEnergyDispatch.git`).

### Steps to Deploy
1. Create a new repository on GitHub named `QuantumEnergyDispatch`.
2. Copy the above content into a `README.md` file in the repository root.
3. Push your script, data, and any additional files (e.g., `myLibs/`) to the repository.
4. Ensure the `requirements.txt` file is created and committed.
