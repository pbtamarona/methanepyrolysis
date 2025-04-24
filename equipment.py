from cost_correlations import *

def create_equipment_list(comp1_power, comp2_power, turb1_power, heatX1_area, heatX2_area, heatX3_area,
                          cooler1_area, cooler2_area, cooler3_area, heater1_duty, reactor_duty,
                          cyclone_volumetric, psa_mole_flow, pump_flow):
    # Compressors
    comp1 = Equipment('Comp-1', 'Fluids', 'Carbon steel', comp1_power, 'Compressor', 'Centrifugal')
    comp2 = Equipment('Comp-2', 'Fluids', 'Carbon steel', comp2_power, 'Compressor', 'Centrifugal')

    # Motors
    motor_comp1 = Equipment('Motor-1', 'Electrical', 'Carbon steel', comp1_power, 'Motor/generator', 'Totally enclosed')
    motor_comp2 = Equipment('Motor-2', 'Electrical', 'Carbon steel', comp2_power, 'Motor/generator', 'Totally enclosed')

    # Turbines
    turb1 = Equipment('Turb-1', 'Fluids', 'Carbon steel', -turb1_power, 'Turbine', 'Condensing steam')

    # Generators
    generator_turb1 = Equipment('Gen-1', 'Electrical', 'Carbon steel', -turb1_power, 'Motor/generator',
                                'Totally enclosed')

    # Heat Exchangers
    heatX1 = Equipment('HeatX-1', 'Fluids', 'Carbon steel', heatX1_area, 'Heat exchanger', 'U-tube shell & tube')
    heatX2 = Equipment('HeatX-2', 'Fluids', 'Carbon steel', heatX2_area, 'Heat exchanger', 'U-tube shell & tube')
    heatX3 = Equipment('HeatX-3', 'Fluids', 'Carbon steel', heatX3_area, 'Heat exchanger', 'U-tube shell & tube')

    # Cooler Heat Exchangers
    cooler1 = Equipment('Cooler-1', 'Fluids', 'Carbon steel', cooler1_area, 'Heat exchanger', 'U-tube shell & tube')
    cooler2 = Equipment('Cooler-2', 'Fluids', 'Carbon steel', cooler2_area, 'Heat exchanger', 'U-tube shell & tube')
    cooler3 = Equipment('Cooler-3', 'Fluids', 'Carbon steel', cooler3_area, 'Heat exchanger', 'U-tube shell & tube')

    # Heater / Furnace / Reactor
    heater1 = Equipment('Heater-1', 'Fluids', 'Carbon steel', heater1_duty, 'Furnace/heater', 'Cylindrical furnace')
    reactor_furnace = Equipment('Reactor', 'Mixed', 'Carbon steel', reactor_duty * 1000, 'Reactor', 'Pyrolysis furnace')

    # Others
    cyclone = Equipment('Cyclone', 'Mixed', 'Carbon steel', cyclone_volumetric, 'Cyclone')
    psa = Equipment('PSA', 'Fluids', 'Carbon steel', psa_mole_flow, 'PSA')
    pump = Equipment('Pump', 'Fluids', 'Carbon steel', pump_flow, 'Pump')

    # List of all equipment
    equipments = [
        comp1, comp2, motor_comp1, motor_comp2,
        turb1, generator_turb1,
        heatX1, heatX2, heatX3,
        cooler1, cooler2, cooler3,
        heater1, reactor_furnace,
        cyclone, psa, pump
    ]

    return equipments

class Equipment:
    process_factors = {
        'Solids': {'fer': 0.6, 'fp': 0.2, 'fi': 0.2, 'fel': 0.15, 'fc': 0.2, 'fs': 0.1, 'fl': 0.05},
        'Fluids': {'fer': 0.3, 'fp': 0.8, 'fi': 0.3, 'fel': 0.2, 'fc': 0.3, 'fs': 0.2, 'fl': 0.1},
        'Mixed': {'fer': 0.5, 'fp': 0.6, 'fi': 0.3, 'fel': 0.2, 'fc': 0.3, 'fs': 0.2, 'fl': 0.1},
        'Electrical': {'fer': 0.4, 'fp': 0.1, 'fi': 0.7, 'fel': 0.7, 'fc': 0.2, 'fs': 0.1, 'fl': 0.1}
    }

    material_factors = {
        'Carbon steel': 1.0,
        'Aluminum': 1.07,
        'Bronze': 1.07,
        'Cast steel': 1.1,
        '304 stainless steel': 1.3,
        '316 stainless steel': 1.3,
        '321 stainless steel': 1.5,
        'Hastelloy C': 1.55,
        'Monel': 1.65,
        'Nickel': 1.7,
        'Inconel': 1.7,
    }

    cost_funcs = {
        'Blower': blower_towler_2010,
        'Compressor': {
            'Centrifugal': centrifugal_compressor_towler_2010,
            'Reciprocating': reciprocating_compressor_towler_2010
        },
        'Cyclone': gas_multi_cyclone_ulrich_2003,
        'Heat exchanger': {
            'U-tube shell & tube': u_shell_tube_hx_towler_2010,
            'Floating head shell & tube': floating_head_shell_tube_hx_towler_2010,
            'Double pipe': double_pipe_hx_towler_2010,
            'Thermosiphon reboiler': thermosiphon_reboiler_towler_2010,
            'U-tube kettle reboiler': u_tube_kettle_reboiler_towler_2010,
            'Plate & frame': plate_frame_hx_towler_2010
        },
        'Furnace/heater': {
            'Cylindrical furnace': cylindrical_furnace_towler_2010,
            'Box furnace': box_furnace_towler_2010,
            'Pyrolysis furnace': pyrolysis_furnace_ulrich_2003,
            'Electric furnace': electric_arc_furnace_parkinson_2016
        },
        'Motor/generator': {
            'Explosion proof': explosion_proof_motor_ulrich_2003,
            'Totally enclosed': totally_enclosed_motor_ulrich_2003
        },
        'PSA': psa_towler_1994,
        'Pump': single_stage_centrifugal_pump_towler_2010,
        'Reactor': {
            'Fluidized Bed': indirect_fluidbed_ulrich_2003,
            'Vertical CS Vessel': vertical_cs_press_vessel_towler_2010,
            'Horizontal CS Vessel': horizontal_cs_press_vessel_towler_2010,
            'Vertical 304SS Vessel': vertical_304ss_press_vessel_towler_2010,
            'Horizontal 304SS Vessel': horizontal_304ss_press_vessel_towler_2010,
        },
        'Turbine': {
            'Condensing steam': condensing_steam_turbine_towler_2010,
            'Axial gas': axial_gas_turbine_ulrich_2003,
            'Radial expander': radial_expander_ulrich_2003
        },
    }

    def __init__(self, name: str, process_type: str, material: str, param: float, type: str, subtype: str = None):
        self.name = name
        self.type = type
        self.subtype = subtype
        self.param = param
        self.material = material
        self.process_type = process_type
        self.cost_year, self.num_units = None, None
        self.purchased_cost = self.calculate_equipment_cost()
        self.direct_cost = self.calculate_direct_cost()

    def calculate_equipment_cost(self) -> float:
        if self.type not in self.cost_funcs:
            raise ValueError(f'No available cost correlations for equipment type: {self.type}')

        cost_funcs = self.cost_funcs[self.type]
        if isinstance(cost_funcs, dict):
            if self.subtype not in cost_funcs:
                raise ValueError(f'No available cost correlations for subtype: {self.subtype}')
            cost_funcs = cost_funcs[self.subtype]

        purchased_cost, self.num_units, self.cost_year = cost_funcs(self.param)
        return inflation_adjustment(purchased_cost, self.cost_year)

    def calculate_direct_cost(self) -> float:
        if self.process_type not in self.process_factors:
            raise ValueError(f'Process type not found: {self.process_type}')

        if self.material not in self.material_factors:
            raise ValueError(f'Material not found: {self.material}')

        factors = self.process_factors[self.process_type]
        fm = self.material_factors[self.material]

        self.direct_cost = self.purchased_cost * ((1 + factors['fp']) * fm + (
                factors['fer'] + factors['fel'] + factors['fi'] + factors['fc'] + factors['fs'] + factors['fl']))
        return self.direct_cost

    def __str__(self) -> str:
        return (f"Name={self.name}, Type={self.type}, Sub-type={self.subtype}, "
                f"Material={self.material}, Process Type={self.process_type}, "
                f"Parameter={self.param}, Purchased Cost={self.purchased_cost}, "
                f"Direct Cost={self.direct_cost})")
