## Hi there 👋 Welcome to my GitHub!

🔬 My research focuses on the **viability of methane pyrolysis** — a promising route to produce low-emission *turquoise hydrogen* by thermally splitting methane into hydrogen gas and functional solid carbon. This repository supports the study titled: ***Techno-economic analysis of catalytic methane pyrolysis in a fluidized bed reactor with reactor-scale catalyst deactivation modeling***.

---

### 📁 Repository Contents

- `run_simulation.ipynb`  
  Main Jupyter notebook to configure and run process simulation scenarios and perform the techno-economic assessment.

- `simulation.py`  
  Coordinates the entire simulation workflow, including reading input from `run_simulation.ipynb`, running Aspen Plus simulations alongside Python reactor models, and retrieving and saving the results.

- `reactor_model.py`  
  Contains the CSTR and PFR fluidized bed reactor models, involving the reaction kinetics and catalyst deactivation. These are integrated with the Aspen Plus process simulation to account for deactivation kinetics.

- `process_plant.py`  
  Retrieves simulation results and converts them into a `ProcessPlant` object, which includes calculation of capital and operating costs, as well as carbon dioxide emissions.

- `equipment.py`  
  Defines the `Equipment` class used within `ProcessPlant` to represent process equipment with attributes such as type, material, design parameters, purchase cost, and direct cost.

- `cost_correlations.py`  
  Provides a database of process equipment cost correlations and includes functions for inflation adjustment.

- `Aspen Plus/`  
  Folder containing 6 Aspen Plus simulations:
  - PFR and CSTR configurations
  - Each simulated with:  
    • Electric heating  
    • CH₄ combustion  
    • H₂ combustion

---

### 🤝 Supervision & Collaboration

This project is supervised by:

- **Dr.ir. Mahinder Ramdin**  
- **Prof.dr.ir. Thijs Vlugt**

in collaboration with **Shell** and **BASF**, under the **ARC CBBC** multilateral initiative.

---

### 🔗 Learn More

📄 [Project Overview](https://lnkd.in/gVKF-_Uu)

---

Thank you for visiting! Feel free to explore, use, or contribute if you work on related topics in hydrogen production or process modeling. For questions or collaboration opportunities, please contact me at [P.B.Tamarona@tudelft.nl](mailto:P.B.Tamarona@tudelft.nl)

