from IPython.display import display
from typing import Tuple, List
from equipment import *
import pandas as pd
import numpy as np
from scipy.optimize import root_scalar
import math


class ProcessPlant:
    """
    A class to represent a process plant and perform economic calculations.

    Attributes:
    processTypes (dict): A dictionary containing process types and their parameters.
    locFactors (dict): A dictionary containing location factors for different regions.
    """

    processTypes = {
        'Solids': {'OS': 0.4, 'DE': 0.2, 'X': 0.1},
        'Fluids': {'OS': 0.3, 'DE': 0.3, 'X': 0.1},
        'Mixed': {'OS': 0.4, 'DE': 0.25, 'X': 0.1},
    }

    locFactors = {
        'United States': {'Gulf Coast': 1.00, 'East Coast': 1.04, 'West Coast': 1.07, 'Midwest': 1.02},
        'Canada': {'Ontario': 1.00, 'Fort McMurray': 1.60},
        'Mexico': 1.03, 'Brazil': 1.14, 'China': {'imported': 1.12, 'indigenous': 0.61},
        'Japan': 1.26, 'Southeast Asia': 1.12, 'Australia': 1.21, 'India': 1.02, 'Middle East': 1.07,
        'France': 1.13, 'Germany': 1.11, 'Italy': 1.14, 'Netherlands': 1.19, 'Russia': 1.53, 'United Kingdom': 1.02,
    }

    def __init__(self, configuration: dict):
        """
        Initialize the ProcessPlant with configuration, parameters, and equipment list.

        Parameters:
        configuration (dict): A dictionary containing plant and economic specifications.
        params (dict): A dictionary containing various parameters for the plant.
        equipment_list (List[Equipment]): A list of equipment objects used in the plant.
        """

        self.contigency = None
        self.DnE = None
        self.ISBL = None
        self.OSBL = None
        self.working_capital = None
        self.fixed_capital = None
        self.total_investment = None
        self.annual_capital_charge = None
        self.ch4_cost = None
        self.co2_emission = None
        self.co2_tax_cost = None
        self.co2_capture_rate = None
        self.co2_capture_opex = None
        self.co2_capture_capex = None
        self.cat_cost = None
        self.solid_c_deposit = None
        self.solid_c_revenue = None
        self.solid_c_specs = None
        self.h2_revenue = None
        self.annual_h2_prod = None
        self.conversion = None
        self.raw_materials_costs = None
        self.total_comp_power = None
        self.total_turb_power = None
        self.consumed_electricity = None
        self.recovered_electricity = None
        self.cooling_cost = None
        self.electricity_cost = None
        self.variable_production_costs = None
        self.total_utilities_costs = None
        self.operating_labor_costs = None
        self.operators_hired = None
        self.supervision_costs = None
        self.direct_salary_overhead = None
        self.taxes_insurance_costs = None
        self.rent_of_land_costs = None
        self.environmental_charges = None
        self.general_plant_overhead = None
        self.operating_supplies = None
        self.maintenance_costs = None
        self.laboratory_charges = None
        self.patents_royalties = None
        self.distribution_selling_costs = None
        self.RnD_costs = None
        self.annualized_working_capital = None
        self.fixed_production_costs = None
        self.cash_cost_of_production = None
        self.total_cost_of_production = None
        self.levelized_opex = None
        self.levelized_capex = None
        self.lcoh = None
        self.specific_co2 = None
        self.annual_cash_flow = None
        self.payback_time = None

        self.config = configuration
        if configuration['plant_specs']['reactor']['type'] == 'AVERAGE':
            self.params, self.equipment_list = self.get_average_params_equipment()
        else:
            self.params, self.equipment_list = self.get_params_equipment()

        plant_specs = configuration['plant_specs']
        self.utilization_rate = plant_specs['utilization_rate']
        self.process_type = plant_specs['process_type']
        self.country = plant_specs['country']
        self.region = plant_specs['region']
        self.co2_ccs = plant_specs['co2_ccs']
        self.co2_intensity_electricity = plant_specs['co2_intensity_electricity']

        economic_specs = configuration['economic_specs']
        self.interest_rate = economic_specs['interest_rate']
        self.tax_rate = economic_specs['tax_rate']
        self.project_lifetime = round(economic_specs['project_lifetime'])
        self.cash_flow = np.zeros(self.project_lifetime)
        self.fc_optimism = economic_specs['fc_optimism']
        self.fp_optimism = economic_specs['fp_optimism']
        self.ch4_price = economic_specs['ch4_price']
        self.co2_tax_rate = economic_specs['co2_tax_rate']
        self.lcca = economic_specs['lcca']
        self.cat_price = economic_specs['cat_price']
        self.electricity_price = economic_specs['electricity_price']
        self.operators_hourly_rate = economic_specs['operators_hourly_rate']
        self.solid_c_yearly_sales = economic_specs['solid_c_yearly_sales']
        self.solid_c_price = economic_specs['solid_c_price']
        self.solid_c_disposal_fee = economic_specs['solid_c_disposal_fee']
        self.h2_price = economic_specs['h2_price']

        reactor_spec = plant_specs['reactor']
        self.heat_source = reactor_spec['heating']

        # Parameters
        self.reactor_units = self.params['reactor_units']
        self.ch4_feed = self.params['ch4_feed']
        self.solid_c_prod = self.params['solid_c_prod']
        self.heat_fuel = self.params['heat_fuel']
        self.ch4_reactor_feed = self.params['ch4_reactor_feed']
        self.ch4_unreacted = self.params['ch4_unreacted']
        self.ch4_rcycl = self.params['ch4_rcycl']
        self.cat_feed = self.params['cat_feed']
        self.cooling_total = self.params['cooling_total']  # kJ/s
        self.reactor_duty = self.params['reactor_duty'] * 1000  # kW

        if self.heat_source == 'h2':
            self.h2_prod = self.params['h2_prod'] - self.heat_fuel
        else:
            self.h2_prod = self.params['h2_prod']

        self.calculate_plant_economics()

    def calculate_plant_economics(self):
        self.calculate_isbl()
        self.sum_comp_power()
        self.sum_turb_power()
        self.calculate_reactor_conversion()
        self.calculate_raw_materials()
        self.calculate_solid_c_specs()
        self.calculate_utilities()
        self.co2_capture_storage()
        self.co2_tax()
        self.calculate_total_investment()
        self.calculate_variable_production_costs()
        self.calculate_operating_labor()
        self.calculate_fixed_production_costs()
        self.annual_capital_charge = self.amortization(self.fixed_capital) + self.co2_capture_capex
        self.calculate_total_cost_of_production()
        self.calculate_total_revenue()
        self.calculate_cash_flow()
        self.calculate_payback_time()
        self.calculate_npv()
        self.calculate_irr()
        self.calculate_lcoh()

    def add_equipment(self, equipment: Equipment):
        """
        Add equipment to the plant and recalculate the economics.

        Parameters:
        equipment (Equipment): The equipment to be added.
        """
        self.equipment_list.append(equipment)
        self.calculate_plant_economics()

    def calculate_isbl(self):
        """
        Calculate the Inside Battery Limits (ISBL) cost.
        """
        if self.co2_ccs:
            self.ISBL = sum(
                equipment.direct_cost * self.reactor_units if equipment.type == 'Reactor' else equipment.direct_cost
                for equipment in self.equipment_list
            ) * self.location_factors()
        else:
            self.ISBL = sum(
                equipment.direct_cost * self.reactor_units if equipment.type == 'Reactor' else equipment.direct_cost
                for equipment in self.equipment_list
                if equipment.name not in ['Comp-4', 'Cooler-4']
            ) * self.location_factors()

    def calculate_total_investment(self, percentage_working_capital: float = 0.15):
        """
        Calculate the total investment required for the plant.

        Parameters:
        percentage_working_capital (float): The percentage of working capital. Default is 0.15.
        """
        if self.process_type not in self.processTypes:
            raise ValueError(f'Process plant type not found: {self.process_type}')

        params = self.processTypes[self.process_type]
        self.OSBL = params['OS'] * self.ISBL
        self.DnE = params['DE'] * (self.ISBL + self.OSBL)
        self.contigency = params['X'] * (self.ISBL + self.OSBL)
        self.fixed_capital = ((self.ISBL + self.OSBL + self.DnE + self.contigency + self.co2_capture_capex)
                              * self.fc_optimism)

        self.working_capital = percentage_working_capital * self.fixed_capital
        self.total_investment = self.fixed_capital + self.working_capital

    def location_factors(self) -> float:
        """
        Get the location factor based on the country and region.

        Returns:
        float: The location factor.
        """
        if self.country not in self.locFactors:
            raise ValueError(f'Country not found: {self.country}')

        loc_factor = self.locFactors[self.country]
        if isinstance(loc_factor, dict):
            if self.region in loc_factor:
                return loc_factor[self.region]
            else:
                raise ValueError(f'Region not found: {self.region}')
        return loc_factor

    def sum_comp_power(self) -> float:
        """
        Calculate the total compressor power.

        Returns:
        float: The total compressor power.
        """
        self.total_comp_power = 0
        for equipment in self.equipment_list:
            # Exclude 'Comp-4' if self.co2_ccs is False
            if equipment.type == 'Compressor' and (self.co2_ccs or equipment.name != 'Comp-4'):
                self.total_comp_power += equipment.param
        return self.total_comp_power

    def sum_turb_power(self) -> float:
        """
        Calculate the total turbine power.

        Returns:
        float: The total turbine power.
        """
        self.total_turb_power = 0
        for equipment in self.equipment_list:
            if equipment.type in ['Turbine']:
                self.total_turb_power += equipment.param
        return self.total_turb_power

    def hourly_to_annual(self, per_hour: float) -> float:
        """
        Convert an hourly rate to an annual rate.

        Parameters:
        per_hour (float): The hourly rate.

        Returns:
        float: The annual rate.
        """
        hours_per_year = 24 * 365
        return per_hour * hours_per_year * self.utilization_rate

    def calculate_reactor_conversion(self):
        """
        Calculate the reactor conversion rate.
        """
        self.conversion = (self.ch4_reactor_feed - self.ch4_unreacted) / self.ch4_reactor_feed

    def calculate_raw_materials(self):
        """
        Calculate the cost of raw materials.
        """
        if self.heat_source == 'ch4':
            self.ch4_feed = self.params['ch4_feed'] + self.heat_fuel
        else:
            self.ch4_feed = self.params['ch4_feed']
        self.ch4_cost = self.hourly_to_annual(self.ch4_feed * self.ch4_price)
        self.cat_cost = self.hourly_to_annual(self.cat_feed * self.cat_price)
        self.raw_materials_costs = self.ch4_cost + self.cat_cost

    def calculate_solid_c_specs(self):
        """
        Calculate the specifications for solid carbon.
        """
        self.solid_c_specs = self.solid_c_prod / self.cat_feed

    def calculate_utilities(self):
        """
        Calculate the utility costs.
        """
        # Assumptions
        eta_mechanical = 0.96
        eta_electrical = 0.95
        eta_heating = 0.90
        recovery_efficiency = 0.8
        chemical_makeup_water_cost = 0.176 / 1000  # US$/kg
        makeup_water_cost = 0.0347 / 1000  # US$/kg

        # Reference for cooling water price calculation:
        # https://www.pearson.de/analysis-synthesis-and-design-of-chemical-processes-9780134177489
        cooling_water_price = (self.electricity_price * (2.38 + 1.61) + 517.3 *
                               (makeup_water_cost + chemical_makeup_water_cost))

        # Electricity consumption and recovery calculation
        if self.heat_source == 'electric':
            self.consumed_electricity = ((self.total_comp_power / (eta_mechanical * eta_electrical))
                                         + (self.reactor_duty / eta_heating))
        else:
            self.consumed_electricity = self.total_comp_power / (eta_mechanical * eta_electrical)
        self.recovered_electricity = self.total_turb_power * recovery_efficiency
        electricity_cost_hourly = (self.consumed_electricity - self.recovered_electricity) * self.electricity_price

        # Cooling cost calculation
        if self.co2_ccs:
            cooling_water_gjperh = self.cooling_total * 3600 / 1000000
        else:
            cooling_water_gjperh = (self.cooling_total-self.params['cooling4']) * 3600 / 1000000
        cooling_cost_hourly = cooling_water_gjperh * cooling_water_price

        # Total utility cost calculation
        total_utilities_hourly = electricity_cost_hourly + cooling_cost_hourly

        # Conversion from hourly to annual cost
        self.total_utilities_costs = self.hourly_to_annual(total_utilities_hourly)
        self.electricity_cost = self.hourly_to_annual(electricity_cost_hourly)
        self.cooling_cost = self.hourly_to_annual(cooling_cost_hourly)

    def co2_capture_storage(self):
        """
        Calculate the costs and emissions related to CO2 capture and storage.
        """
        if self.co2_ccs:
            self.co2_emission = (self.params['co2_with_CCS'] + self.consumed_electricity*self.co2_intensity_electricity)
            self.co2_capture_opex = self.lcca * self.hourly_to_annual(self.params['co2_with_CCS']) * 0.5
            self.co2_capture_capex = self.lcca * self.hourly_to_annual(self.params['co2_with_CCS']) * 0.5
        else:
            self.co2_emission = (self.params['co2_emission'] + self.consumed_electricity*self.co2_intensity_electricity)
            self.co2_capture_opex = 0
            self.co2_capture_capex = 0

        self.specific_co2 = self.co2_emission / self.h2_prod

    def co2_tax(self):
        """
        Calculate the CO2 tax cost.
        """
        self.co2_tax_cost = self.co2_tax_rate * self.hourly_to_annual(
            self.co2_emission-self.consumed_electricity*self.co2_intensity_electricity)

    def calculate_total_revenue(self):
        """
        Calculate the revenue from solid carbon sales.
        """
        self.annual_solid_c_prod = self.hourly_to_annual(self.solid_c_prod)
        self.solid_c_deposit = 1 - self.solid_c_yearly_sales
        self.solid_c_revenue = (self.annual_solid_c_prod *
                                (self.solid_c_yearly_sales * self.solid_c_price -
                                 self.solid_c_deposit * self.solid_c_disposal_fee))
        self.annual_h2_prod = self.hourly_to_annual(self.h2_prod)
        self.h2_revenue = self.annual_h2_prod * self.h2_price
        self.total_revenue = self.solid_c_revenue + self.h2_revenue

    def calculate_variable_production_costs(self):
        """
        Calculate the variable production costs.
        """
        self.variable_production_costs = (self.raw_materials_costs + self.total_utilities_costs +
                                          self.co2_tax_cost + self.co2_capture_opex)

    def calculate_operating_labor(self):
        """
        Calculate the operating labor costs.
        """
        def count_fluids_or_mixed(equipments):
            count = 0
            for equipment in equipments:
                if equipment.process_type in ['Fluids', 'Mixed']:
                    if equipment.type in ['Pump', 'Vessel', 'Cyclone']:
                        pass
                    else:
                        count += 1*equipment.num_units
            return count

        def count_solids_or_mixed(equipments):
            count = 0
            for equipment in equipments:
                if equipment.process_type in ['Solids', 'Mixed']:
                    if equipment.type in ['Pump', 'Vessel', 'Cyclone']:
                        pass
                    else:
                        count += 1 * equipment.num_units
            return count

        no_fluid_process = count_fluids_or_mixed(self.equipment_list)
        no_solid_process = count_solids_or_mixed(self.equipment_list)

        operators_per_shifts = (6.29 + 31.7 * (no_solid_process ** 2) + 0.23 * no_fluid_process) ** 0.5

        working_weeks_per_year = 49
        working_shifts_per_week = 5  # 8-hour shifts
        operating_shifts_per_year = 365 * 3

        working_shifts_per_year = working_weeks_per_year * working_shifts_per_week
        working_hours_per_year = working_shifts_per_year * 8
        self.operators_hired = math.ceil(operators_per_shifts * operating_shifts_per_year / working_shifts_per_year)
        self.operating_labor_costs = self.operators_hired * working_hours_per_year * self.operators_hourly_rate

    def calculate_fixed_production_costs(self):
        """
        Calculate the fixed production costs.
        """
        self.supervision_costs = 0.25 * self.operating_labor_costs
        self.direct_salary_overhead = 0.5 * (self.operating_labor_costs + self.supervision_costs)
        self.laboratory_charges = 0.10 * self.operating_labor_costs
        self.maintenance_costs = 0.05 * self.ISBL
        self.taxes_insurance_costs = 0.015 * self.ISBL
        self.rent_of_land_costs = 0.015 * (self.ISBL + self.OSBL)
        self.environmental_charges = 0.01 * (self.ISBL + self.OSBL)
        self.operating_supplies = 0.009 * self.ISBL
        self.general_plant_overhead = 0.65 * (self.operating_labor_costs + self.supervision_costs +
                                              self.direct_salary_overhead)

        self.interest_working_capital = self.working_capital*self.interest_rate

        self.fixed_production_costs = (self.operating_labor_costs + self.supervision_costs + self.direct_salary_overhead
                                       + self.laboratory_charges + self.maintenance_costs + self.taxes_insurance_costs
                                       + self.rent_of_land_costs + self.environmental_charges + self.operating_supplies
                                       + self.general_plant_overhead + self.interest_working_capital)

        cash_cost_of_production = (self.variable_production_costs + self.fixed_production_costs) / (1 - 0.07)

        self.patents_royalties = 0.02 * cash_cost_of_production
        self.distribution_selling_costs = 0.02 * cash_cost_of_production
        self.RnD_costs = 0.03 * cash_cost_of_production

        self.fixed_production_costs += self.patents_royalties + self.distribution_selling_costs + self.RnD_costs
        self.fixed_production_costs *= self.fp_optimism

        self.cash_cost_of_production = self.variable_production_costs + self.fixed_production_costs

    def calculate_total_cost_of_production(self):
        """
        Calculate the total cost of production by summing the cash cost of production and the annual capital charge.
        """
        self.total_cost_of_production = self.cash_cost_of_production + self.annual_capital_charge

    def amortization(self, capital: float) -> float:
        """
        Calculate the annual amortization charge for a given capital investment.

        Parameters:
        capital (float): The capital investment amount.

        Returns:
        float: The annual amortization charge.
        """
        annual_charge = (capital * self.interest_rate * ((1 + self.interest_rate) ** self.project_lifetime) /
                         (((1 + self.interest_rate) ** self.project_lifetime) - 1))
        return annual_charge

    def calculate_cash_flow(self):
        # Initialize arrays
        capital_cost_array = np.zeros(self.project_lifetime)
        h2_revenue_array = np.zeros(self.project_lifetime)
        solid_c_revenue_array = np.zeros(self.project_lifetime)
        cash_cost_array = np.zeros(self.project_lifetime)
        gross_profit_array = np.zeros(self.project_lifetime)
        depreciation_array = np.zeros(self.project_lifetime)
        taxable_income_array = np.zeros(self.project_lifetime)
        tax_paid_array = np.zeros(self.project_lifetime)
        h2_prod_array = np.zeros(self.project_lifetime)

        utilities_array = np.zeros(self.project_lifetime)
        raw_materials_array = np.zeros(self.project_lifetime)
        co2_tax_array = np.zeros(self.project_lifetime)
        fixed_production_array = np.zeros(self.project_lifetime)

        previous_taxable_income = 0
        depreciation_counter = 0
        depreciation_amount = self.fixed_capital / (self.project_lifetime / 2)

        for year in range(self.project_lifetime):
            if year == 0:
                h2_prod = 0
                cash_cost = 0
                capital_cost = self.fixed_capital * 0.3
                h2_revenue = 0
                solid_c_revenue = 0
                utilities = 0
                raw_materials = 0
                co2_tax = 0
                fixed_production = 0
            elif year == 1:
                h2_prod = 0
                cash_cost = 0
                capital_cost = self.fixed_capital * 0.6
                h2_revenue = 0
                solid_c_revenue = 0
                utilities = 0
                raw_materials = 0
                co2_tax = 0
                fixed_production = 0
            elif year == 2:
                h2_prod = 0.4 * self.annual_h2_prod
                cash_cost = self.fixed_production_costs + 0.4 * self.variable_production_costs
                capital_cost = self.fixed_capital * 0.1 + self.working_capital
                h2_revenue = 0.4*self.h2_revenue
                solid_c_revenue = 0.4*self.solid_c_revenue
                utilities = 0.4*self.total_utilities_costs
                raw_materials = 0.4*self.raw_materials_costs
                co2_tax = 0.4*self.co2_tax_cost
                fixed_production = self.fixed_production_costs
            elif year == 3:
                h2_prod = 0.8 * self.annual_h2_prod
                cash_cost = self.fixed_production_costs + 0.8 * self.variable_production_costs
                capital_cost = 0
                h2_revenue = 0.8 * self.h2_revenue
                solid_c_revenue = 0.8 * self.solid_c_revenue
                utilities = 0.8 * self.total_utilities_costs
                raw_materials = 0.8 * self.raw_materials_costs
                co2_tax = 0.8 * self.co2_tax_cost
                fixed_production = self.fixed_production_costs
            else:
                h2_prod = self.annual_h2_prod
                cash_cost = self.fixed_production_costs + self.variable_production_costs
                capital_cost = 0
                h2_revenue = self.h2_revenue
                solid_c_revenue = self.solid_c_revenue
                utilities = self.total_utilities_costs
                raw_materials = self.raw_materials_costs
                co2_tax = self.co2_tax_cost
                fixed_production = self.fixed_production_costs

            gross_profit = h2_revenue + solid_c_revenue - cash_cost

            if gross_profit > 0 and depreciation_counter < (self.project_lifetime/2):
                depreciation = depreciation_amount
                depreciation_counter += 1
            else:
                depreciation = 0

            taxable_income = gross_profit - depreciation
            tax_paid = self.tax_rate * previous_taxable_income if previous_taxable_income > 0 else 0

            capital_cost_array[year] = capital_cost
            h2_revenue_array[year] = h2_revenue
            solid_c_revenue_array[year] = solid_c_revenue
            cash_cost_array[year] = cash_cost
            gross_profit_array[year] = gross_profit
            depreciation_array[year] = depreciation
            taxable_income_array[year] = taxable_income
            tax_paid_array[year] = tax_paid
            self.cash_flow[year] = gross_profit - tax_paid - capital_cost
            h2_prod_array[year] = h2_prod

            utilities_array[year] = utilities
            raw_materials_array[year] = raw_materials
            co2_tax_array[year] = co2_tax
            fixed_production_array[year] = fixed_production

            previous_taxable_income = taxable_income

        capital_cost_array[-1] -= self.working_capital  # Add working capital recovery in the last year
        self.cash_flow[-1] += self.working_capital  # Add working capital recovery in the last year

        return (capital_cost_array, h2_revenue_array, solid_c_revenue_array, cash_cost_array, gross_profit_array,
                depreciation_array, taxable_income_array, tax_paid_array, self.cash_flow, h2_prod_array,
                utilities_array, raw_materials_array, co2_tax_array, fixed_production_array)

    def calculate_npv(self):
        pv_array = np.zeros(self.project_lifetime)
        npv_array = np.zeros(self.project_lifetime)

        for year in range(self.project_lifetime):
            pv_array[year] = self.cash_flow[year] / ((1 + self.interest_rate) ** (year+1))
            npv_array[year] = np.sum(pv_array)

        self.npv = npv_array[-1]

        return pv_array, npv_array

    def create_cash_flow_table(self):
        (capital_cost, h2_revenue, solid_c_revenue, cash_cost, gross_profit, depreciation, taxable_income, tax_paid,
         cash_flow, _, _, _, _, _)= self.calculate_cash_flow()
        pv, npv = self.calculate_npv()

        data = {
            "Year": np.arange(self.project_lifetime)+1,
            "Capital cost": capital_cost,
            "H2 revenue": h2_revenue,
            "Solid C revenue": solid_c_revenue,
            "Cash cost of prod.": cash_cost,
            "Gross profit": gross_profit,
            "Depreciation": depreciation,
            "Taxable income": taxable_income,
            "Tax paid": tax_paid,
            "Cash flow": cash_flow,
            "PV of cash flow": pv,
            "NPV": npv
        }

        df = pd.DataFrame(data)

        # Format numerical values with dollar sign and thousand separators
        formatted_df = df.style.format(
            {col: "${:,.2f}" for col in df.columns if col != "Year"}
        )

        return formatted_df

    def calculate_lcoh(self):
        (capital_cost, _, solid_c_revenue, cash_cost, _, _, _, _, _, h2_prod,
         utilities, raw_materials, co2_tax, fixed_prod) = self.calculate_cash_flow()
        (capex, opex, solid_c, h2,
         util, rawmat, co2, fixprod) = (np.zeros(self.project_lifetime), np.zeros(self.project_lifetime),
                                        np.zeros(self.project_lifetime), np.zeros(self.project_lifetime),
                                        np.zeros(self.project_lifetime), np.zeros(self.project_lifetime),
                                        np.zeros(self.project_lifetime), np.zeros(self.project_lifetime))
        for year in range(self.project_lifetime):
            capex[year] = (capital_cost[year]) / ((1 + self.interest_rate) ** (year+1))
            opex[year] = (cash_cost[year]) / ((1 + self.interest_rate) ** (year+1))
            solid_c[year] = solid_c_revenue[year] / ((1 + self.interest_rate) ** (year+1))
            h2[year] = h2_prod[year] / ((1 + self.interest_rate) ** (year+1))
            util[year] = utilities[year] / ((1 + self.interest_rate) ** (year+1))
            rawmat[year] = raw_materials[year] / ((1 + self.interest_rate) ** (year+1))
            co2[year] = co2_tax[year] / ((1 + self.interest_rate) ** (year+1))
            fixprod[year] = fixed_prod[year] / ((1 + self.interest_rate) ** (year+1))
        self.levelized_capex = max(np.sum(capex) / np.sum(h2), 0)
        self.levelized_opex = max(np.sum(opex) / np.sum(h2), 0)
        self.levelized_solid_c = max(np.sum(solid_c) / np.sum(h2), 0)
        self.levelized_util = max(np.sum(util) / np.sum(h2), 0)
        self.levelized_rawmat = max(np.sum(rawmat) / np.sum(h2), 0)
        self.levelized_co2 = max(np.sum(co2) / np.sum(h2), 0)
        self.levelized_fixprod = max(np.sum(fixprod) / np.sum(h2), 0)
        self.lcoh = max(np.sum(capex + opex - solid_c) / np.sum(h2), 0)

    def calculate_payback_time(self):
        _, h2_revenue, solid_c_revenue, _, _, _, _, _, cash_flow, _, _, _, _, _ = self.calculate_cash_flow()
        revenue_array = h2_revenue + solid_c_revenue
        revenue_generating_years = cash_flow[revenue_array > 0]

        if len(revenue_generating_years) == 0:
            self.payback_time = float('nan')
        else:
            average_annual_cash_flow = np.mean(revenue_generating_years)
            self.payback_time = self.fixed_capital / average_annual_cash_flow if average_annual_cash_flow > 0 else float(
                'nan')

    def calculate_irr(self):
        # Define NPV function
        def npv(irr):
            return sum(self.cash_flow[year] / ((1 + irr)**(year+1)) for year in range(len(self.cash_flow)))
        sol = root_scalar(npv, bracket=[-10, 10], method='brentq')
        self.irr = sol.root if sol.converged  else float('nan')


    def get_params_equipment(self) -> Tuple[dict, List[Equipment]]:
        """
        Extract parameters and equipment details from the given configuration.

        Parameters:
        configuration (dict): A dictionary containing plant and reactor specifications.

        Returns:
        Tuple[dict, List[Equipment]]: A tuple containing a dictionary of parameters and a list of equipment objects.
        """
        # Extract nested dictionary values into local variables for readability
        plant_specs = self.config['plant_specs']
        reactor_spec = plant_specs['reactor']

        reac_type = reactor_spec['type']
        plant_capacity = plant_specs['plant_capacity']
        heat_source = reactor_spec['heating']
        reac_temp = reactor_spec['temperature']
        reac_press = reactor_spec['pressure']
        catwt = reactor_spec['catwt']
        deactivation = reactor_spec['deactivation']
        psa_press = plant_specs['psa_press']

        # Read the CSV file into a DataFrame
        df = pd.read_csv(f'sim-{reac_type}-{plant_capacity}TPD-{heat_source}-t{reac_temp}-p{round(reac_press, 3)}'
                         f'-c{round(catwt, 3)}-da{round(deactivation, 2)}-psa{psa_press}.csv')
        data = df.values.T.ravel()

        # Unpack data
        (ch4_feed, h2_prod, solid_c_prod, ch4_rcycl, ch4_reactor_feed, ch4_unreacted,
         cat_feed, cat_res_time, reactor_deactivation, reactor_units, reactor_volume,
         comp1_power, comp2_power, comp3_power, comp3_vflow, comp4_power,
         heatX1_area, heatX2_area, heatX3_area,
         turb1_power, pump_flow, cooling_total,
         cooler1_area, cooler2_area, cooler3_area, cooler4_area,
         cooler1_water, cooler2_water, cooler3_water, cooler4_water,
         heat_fuel, co2_emission, co2_with_CCS,
         heater1_duty, heater2_duty, reactor_duty, cyclone_volumetric,
         psa_mole_flow, error) = data

        # Create equipment objects
        comp1_power = max(comp1_power, 75)
        comp1 = Equipment('Comp-1', 'Fluids', 'Carbon steel', comp1_power, 'Compressor', 'Centrifugal')
        comp2 = Equipment('Comp-2', 'Fluids', 'Carbon steel', comp2_power, 'Compressor', 'Centrifugal')
        motor_comp1 = Equipment('Motor-1', 'Electrical', 'Carbon steel', comp1_power, 'Motor/generator', 'Totally enclosed')
        motor_comp2 = Equipment('Motor-2', 'Electrical', 'Carbon steel', comp2_power, 'Motor/generator', 'Totally enclosed')
        heatX1 = Equipment('Cooler-1' if heatX1_area < 10 else 'HeatX-1', 'Fluids', '316 stainless steel',
                           heatX1_area, 'Heat exchanger', 'Double pipe' if heatX1_area < 10 else 'U-tube shell & tube')
        cooler1 = Equipment('Cooler-1', 'Fluids', 'Carbon steel', cooler1_area,
                            'Heat exchanger', 'Double pipe' if cooler1_area < 10 else 'U-tube shell & tube')
        cooler2 = Equipment('Cooler-2', 'Fluids', 'Carbon steel', cooler2_area,
                            'Heat exchanger', 'Double pipe' if cooler2_area < 10 else 'U-tube shell & tube')
        cyclone = Equipment('Cyclone', 'Mixed', '316 stainless steel', cyclone_volumetric, 'Cyclone')
        psa = Equipment('PSA', 'Fluids', 'Carbon steel', psa_mole_flow, 'PSA')
        reactor = Equipment('Reactor', 'Mixed', '316 stainless steel', reactor_volume, 'Reactor',
                            'Fluidized Bed')

        if heat_source != 'electric':
            if comp3_power < 75:
                comp3 = Equipment('Comp-3', 'Fluids', 'Carbon steel', comp3_vflow, 'Blower')
            else:
                comp3 = Equipment('Comp-3', 'Fluids', 'Carbon steel', comp3_power, 'Compressor', 'Centrifugal')
            turb1 = Equipment('Turb-1', 'Fluids', 'Carbon steel', turb1_power * -1, 'Turbine', 'Condensing steam')
            generator_turb1 = Equipment('Gen-1', 'Electrical', 'Carbon steel', turb1_power * -1, 'Motor/generator',
                                        'Totally enclosed')
            heatX2 = Equipment('HeatX-2', 'Fluids', '316 stainless steel',
                               heatX2_area, 'Heat exchanger', 'Double pipe' if heatX2_area < 10 else 'U-tube shell & tube')
            heatX3 = Equipment('HeatX-3', 'Fluids', 'Carbon steel', heatX3_area, 'Heat exchanger', 'U-tube shell & tube')
            cooler3 = Equipment('Cooler-3', 'Fluids', 'Carbon steel', cooler3_area,
                                'Heat exchanger', 'Double pipe' if cooler3_area < 10 else 'U-tube shell & tube')
            total_heating = (heater1_duty + heater2_duty + reactor_duty) * 1000  # Convert to kW
            furnace = Equipment('Furnace', 'Fluids', '316 stainless steel', total_heating/0.85, 'Furnace/heater',
                                'Pyrolysis furnace')
            pump_flow = max(pump_flow, 0.2)
            pump = Equipment('Pump', 'Fluids', 'Carbon steel', pump_flow, 'Pump')
            if heat_source == 'ch4':
                comp4 = Equipment('Comp-4', 'Fluids', 'Carbon steel', comp4_power, 'Compressor', 'Centrifugal')
                cooler4 = Equipment('Cooler-4', 'Fluids', 'Carbon steel', cooler4_area,
                                    'Heat exchanger', 'Double pipe' if cooler4_area < 10 else 'U-tube shell & tube')
                equipments = [comp1, comp2, comp3, comp4, motor_comp1, motor_comp2, turb1, generator_turb1, heatX1, heatX2,
                          heatX3, cooler1, cooler2, cooler3, cooler4, furnace, cyclone, psa, pump, reactor]
            else:
                equipments = [comp1, comp2, comp3, motor_comp1, motor_comp2, turb1, generator_turb1, heatX1, heatX2,
                          heatX3, cooler1, cooler2, cooler3, furnace, cyclone, psa, pump, reactor]
        else:
            total_heating = heater1_duty + heater2_duty + reactor_duty
            furnace = Equipment('Furnace', 'Electrical', '316 stainless steel',
                                total_heating/0.9, 'Furnace/heater', 'Electric furnace')

            equipments = [comp1, comp2, motor_comp1, motor_comp2, heatX1, cooler1, cooler2, furnace, cyclone, psa, reactor]

        # Return the list of parameters and equipment
        params = {
            'ch4_feed': ch4_feed, 'h2_prod': h2_prod, 'solid_c_prod': solid_c_prod, 'ch4_rcycl': ch4_rcycl,
            'ch4_reactor_feed': ch4_reactor_feed, 'ch4_unreacted': ch4_unreacted, 'cat_feed': cat_feed,
            'cat_res_time': cat_res_time, 'reactor_deactivation': reactor_deactivation, 'reactor_units': reactor_units,
            'reactor_volume': reactor_volume, 'comp1_power': comp1_power, 'comp2_power': comp2_power,
            'comp3_power': comp3_power, 'comp4_power': comp4_power, 'turb1_power': turb1_power, 'pump_flow': pump_flow,
            'heatX1_area': heatX1_area, 'heatX2_area': heatX2_area, 'heatX3_area': heatX3_area,
            'cooler1_area': cooler1_area, 'cooler2_area': cooler2_area, 'cooler3_area': cooler3_area,
            'cooler4_area': cooler4_area, 'cooling1': cooler1_water, 'cooling2': cooler2_water,
            'cooling3': cooler3_water, 'cooling4': cooler4_water, 'heat_fuel': heat_fuel, 'co2_emission': co2_emission,
            'co2_with_CCS': co2_with_CCS, 'cooling_total': cooling_total, 'heater1_duty': heater1_duty,
            'heater2_duty': heater2_duty, 'reactor_duty': reactor_duty, 'cyclone_volumetric': cyclone_volumetric,
            'psa_mole_flow': psa_mole_flow, 'error': error
        }

        return params, equipments


    def get_average_params_equipment(self) -> Tuple[dict, List[str]]:
        plant_specs = self.config['plant_specs']
        reactor_spec = plant_specs['reactor']

        plant_capacity = plant_specs['plant_capacity']
        heat_source = reactor_spec['heating']
        reac_temp = reactor_spec['temperature']
        reac_press = reactor_spec['pressure']
        catwt = reactor_spec['catwt']
        deactivation = reactor_spec['deactivation']
        psa_press = plant_specs['psa_press']

        # Read the CSV files into DataFrames
        pfr = pd.read_csv(f'sim-PFR-{plant_capacity}TPD-{heat_source}-t{reac_temp}-p{round(reac_press, 3)}'
                          f'-c{round(catwt, 3)}-da{round(deactivation, 2)}-psa{psa_press}.csv')
        cstr = pd.read_csv(f'sim-CSTR-{plant_capacity}TPD-{heat_source}-t{reac_temp}-p{round(reac_press, 3)}'
                           f'-c{round(catwt, 3)}-da{round(deactivation, 2)}-psa{psa_press}.csv')

        pfr = pfr.values.T.ravel()
        cstr = cstr.values.T.ravel()

        # Compute average values between PFR and CSTR
        avg_values = ((pfr + cstr) / 2)

        # Unpack data
        (ch4_feed, h2_prod, solid_c_prod, ch4_rcycl, ch4_reactor_feed, ch4_unreacted,
         cat_feed, cat_res_time, reactor_deactivation, reactor_units, reactor_volume,
         comp1_power, comp2_power, comp3_power, comp3_vflow, comp4_power,
         heatX1_area, heatX2_area, heatX3_area,
         turb1_power, pump_flow, cooling_total,
         cooler1_area, cooler2_area, cooler3_area, cooler4_area,
         cooler1_water, cooler2_water, cooler3_water, cooler4_water,
         heat_fuel, co2_emission, co2_with_CCS,
         heater1_duty, heater2_duty, reactor_duty, cyclone_volumetric,
         psa_mole_flow, error) = avg_values

        # Create equipment objects
        comp1_power = max(comp1_power, 75)
        comp1 = Equipment('Comp-1', 'Fluids', 'Carbon steel', comp1_power, 'Compressor', 'Centrifugal')
        comp2 = Equipment('Comp-2', 'Fluids', 'Carbon steel', comp2_power, 'Compressor', 'Centrifugal')
        motor_comp1 = Equipment('Motor-1', 'Electrical', 'Carbon steel', comp1_power, 'Motor/generator', 'Totally enclosed')
        motor_comp2 = Equipment('Motor-2', 'Electrical', 'Carbon steel', comp2_power, 'Motor/generator', 'Totally enclosed')
        heatX1 = Equipment('Cooler-1' if heatX1_area < 10 else 'HeatX-1', 'Fluids', '316 stainless steel',
                           heatX1_area, 'Heat exchanger', 'Double pipe' if heatX1_area < 10 else 'U-tube shell & tube')
        cooler1 = Equipment('Cooler-1', 'Fluids', 'Carbon steel', cooler1_area,
                            'Heat exchanger', 'Double pipe' if cooler1_area < 10 else 'U-tube shell & tube')
        cooler2 = Equipment('Cooler-2', 'Fluids', 'Carbon steel', cooler2_area,
                            'Heat exchanger', 'Double pipe' if cooler2_area < 10 else 'U-tube shell & tube')
        cyclone = Equipment('Cyclone', 'Mixed', '316 stainless steel', cyclone_volumetric, 'Cyclone')
        psa = Equipment('PSA', 'Fluids', 'Carbon steel', psa_mole_flow, 'PSA')
        reactor = Equipment('Reactor', 'Mixed', '316 stainless steel', reactor_volume, 'Reactor',
                            'Fluidized Bed')

        if heat_source != 'electric':
            if comp3_power < 75:
                comp3 = Equipment('Comp-3', 'Fluids', 'Carbon steel', comp3_vflow, 'Blower')
            else:
                comp3 = Equipment('Comp-3', 'Fluids', 'Carbon steel', comp3_power, 'Compressor', 'Centrifugal')
            turb1 = Equipment('Turb-1', 'Fluids', 'Carbon steel', turb1_power * -1, 'Turbine', 'Condensing steam')
            generator_turb1 = Equipment('Gen-1', 'Electrical', 'Carbon steel', turb1_power * -1, 'Motor/generator',
                                        'Totally enclosed')
            heatX2 = Equipment('HeatX-2', 'Fluids', '316 stainless steel',
                               heatX2_area, 'Heat exchanger', 'Double pipe' if heatX2_area < 10 else 'U-tube shell & tube')
            heatX3 = Equipment('HeatX-3', 'Fluids', 'Carbon steel', heatX3_area, 'Heat exchanger', 'U-tube shell & tube')
            cooler3 = Equipment('Cooler-3', 'Fluids', 'Carbon steel', cooler3_area,
                                'Heat exchanger', 'Double pipe' if cooler3_area < 10 else 'U-tube shell & tube')
            total_heating = (heater1_duty + heater2_duty + reactor_duty) * 1000  # Convert to kW
            furnace = Equipment('Furnace', 'Fluids', '316 stainless steel', total_heating/0.85, 'Furnace/heater',
                                'Pyrolysis furnace')
            pump_flow = max(pump_flow, 0.2)
            pump = Equipment('Pump', 'Fluids', 'Carbon steel', pump_flow, 'Pump')
            if heat_source == 'ch4':
                comp4 = Equipment('Comp-4', 'Fluids', 'Carbon steel', comp4_power, 'Compressor', 'Centrifugal')
                cooler4 = Equipment('Cooler-4', 'Fluids', 'Carbon steel', cooler4_area,
                                    'Heat exchanger', 'Double pipe' if cooler4_area < 10 else 'U-tube shell & tube')
                equipments = [comp1, comp2, comp3, comp4, motor_comp1, motor_comp2, turb1, generator_turb1, heatX1, heatX2,
                              heatX3, cooler1, cooler2, cooler3, cooler4, furnace, cyclone, psa, pump, reactor]
            else:
                equipments = [comp1, comp2, comp3, motor_comp1, motor_comp2, turb1, generator_turb1, heatX1, heatX2,
                              heatX3, cooler1, cooler2, cooler3, furnace, cyclone, psa, pump, reactor]
        else:
            total_heating = heater1_duty + heater2_duty + reactor_duty
            furnace = Equipment('Furnace', 'Electrical', '316 stainless steel',
                                total_heating/0.9, 'Furnace/heater', 'Electric furnace')

            equipments = [comp1, comp2, motor_comp1, motor_comp2, heatX1, cooler1, cooler2, furnace, cyclone, psa, reactor]

        # Return the list of parameters and equipment
        params = {
            'ch4_feed': ch4_feed, 'h2_prod': h2_prod, 'solid_c_prod': solid_c_prod, 'ch4_rcycl': ch4_rcycl,
            'ch4_reactor_feed': ch4_reactor_feed, 'ch4_unreacted': ch4_unreacted, 'cat_feed': cat_feed,
            'cat_res_time': cat_res_time, 'reactor_deactivation': reactor_deactivation, 'reactor_units': round(reactor_units),
            'reactor_volume': reactor_volume, 'comp1_power': comp1_power, 'comp2_power': comp2_power,
            'comp3_power': comp3_power, 'comp4_power': comp4_power, 'turb1_power': turb1_power, 'pump_flow': pump_flow,
            'heatX1_area': heatX1_area, 'heatX2_area': heatX2_area, 'heatX3_area': heatX3_area,
            'cooler1_area': cooler1_area, 'cooler2_area': cooler2_area, 'cooler3_area': cooler3_area,
            'cooler4_area': cooler4_area, 'cooling1': cooler1_water, 'cooling2': cooler2_water,
            'cooling3': cooler3_water, 'cooling4': cooler4_water, 'heat_fuel': heat_fuel, 'co2_emission': co2_emission,
            'co2_with_CCS': co2_with_CCS, 'cooling_total': cooling_total, 'heater1_duty': heater1_duty,
            'heater2_duty': heater2_duty, 'reactor_duty': reactor_duty, 'cyclone_volumetric': cyclone_volumetric,
            'psa_mole_flow': psa_mole_flow, 'error': error
        }

        return params, equipments