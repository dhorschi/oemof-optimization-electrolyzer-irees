import pandas as pd
import random
from oemof import solph


# hier werden die annualisierten Investitionskosten berechnet
def get_annualized_cost(power_ely= 20, investment_cost_ely = 1500, power_h2_storage = 150, investment_cost_h2_storage = 30,
                        power_el_storage = 5, investment_cost_el_storage = 300, number_years = 20, interest_rate = 0.06, w=0.04):

    #Berechnen der gesamten Investitionskosten aus Elektrolyseur und Speicher
    investment_cost_total = (investment_cost_ely * power_ely + investment_cost_h2_storage* power_h2_storage
                             + investment_cost_el_storage * power_el_storage) * 1000

    #Berechnung Annuitätenfaktor
    a = (((1 + interest_rate) ** number_years) * interest_rate) / (((1 + interest_rate) ** number_years) - 1)
    annualized_cost = investment_cost_total * (a + w)

    min_rate = 1/power_ely #gehört hier eigentlich nicht hin, ist notwendig, um minimale Menge an vorgehaltener Regelenergie zu bestimmen

    return annualized_cost, min_rate

# Definition der technischen Parameter der Speicher
def get_storage_data(el_storage_capacity = 5, el_storage_input_flow = 5, el_storage_output_flow = 5, el_storage_loss_rate = 0.002,
                     el_storage_variable_costs = 0, el_storage_initial_storage_level = 0, hydrogen_storage_capacity = 150,
                     hydrogen_storage_input_flow = 150, hydrogen_storage_output_flow = 150, hydrogen_storage_variable_costs = 0,
                     hydrogen_storage_loss_rate = 0.002, hydrogen_storage_initial_storage_level = 0):

    return (el_storage_capacity, el_storage_input_flow, el_storage_output_flow, el_storage_loss_rate, el_storage_variable_costs,
            el_storage_initial_storage_level, hydrogen_storage_capacity, hydrogen_storage_input_flow, hydrogen_storage_output_flow,
            hydrogen_storage_variable_costs, hydrogen_storage_loss_rate, hydrogen_storage_initial_storage_level)

# Laden der Strompreisdaten vom Großhandel
# Daten müssen, wie in Default-Datei aufbereitet werden
def get_el_price(file="/Users/lucajahner/Documents/GitHub/oemof-optimization_electrolyzer-irees/Daten/DayAhead_Boersenstrompreis_stuendlich_2019_energy_charts.xlsx",
                 konzessionsabgabe = 1.1,
                 umlage_strom_nev = 4,
                 umsatzsteuer = 0.19):
    c_el = pd.read_excel(file)
    c_el = c_el["Preis (EUR/MWh)"].iloc[:]
    c_el = (c_el + konzessionsabgabe + umlage_strom_nev) * (1 + umsatzsteuer)

    return c_el

#Strompreisdaten bei festem PPA-Preis
def get_ppa_price(ppa=50, konzessionsabgabe = 1.1, umlage_strom_nev = 4, umsatzsteuer = 0.19, num_tsteps=8760):
    c_el = (ppa + konzessionsabgabe + umlage_strom_nev) * (1 + umsatzsteuer)
    c_el = [c_el]*num_tsteps

    return c_el

#Laden der Leistungspreise von positiver und negativer Regelenergie
#Daten müssen, wie in Default-Datei aufbereitet werden
def get_lp(fn = "/Users/lucajahner/Documents/GitHub/oemof-optimization_electrolyzer-irees/Daten/Regelenergie_affr_pos_neg_stuendliche_Daten_2019_smard.xlsx",
           column_pos = "Leistungspreis (+) [€/MW]" ,
           column_neg = "Leistungspreis (-) [€/MW]"
           ):
    affr = pd.read_excel(fn)
    lp_pos_affr = affr[column_pos].iloc[:]
    lp_neg_affr = affr[column_neg].iloc[:]

    return lp_pos_affr, lp_neg_affr

#Laden der Information, ob Regelenergie abgerufen wurde oder nicht
#Daten müssen, wie in Default-Datei aufbereitet werden
def abruf_affr(fn = "/Users/lucajahner/Documents/GitHub/oemof-optimization_electrolyzer-irees/Daten/Aktivierte_aFRR_2019_qualitaetsgesichert_stuendliche_Daten.xlsx",
               column_pos = "b_pos",
               column_neg = "b_neg",
               num_tsteps=8760):

    abfrage_affr = pd.read_excel(fn)
    b_pos = abfrage_affr[column_pos].iloc[:num_tsteps]
    b_neg = abfrage_affr[column_neg].iloc[:num_tsteps]

    return b_pos, b_neg


def get_ap(c_el, c_h2_virtual, efficiency=0.55):
    ap_neg_affr = c_el + efficiency * c_h2_virtual
    ap_neg_affr = [x if x >= 0 else 0 for x in
                   ap_neg_affr]  # bei den Werten die kleiner 0 sind lohnt es sich bereits zu produzieren, heißt der AP ist irrelevant um wirtschaftlich zu sein
    ap_neg_affr = pd.Series(ap_neg_affr)  # positive Werte - bei variable_cost muss ein minus davor

    ap_pos_affr = c_el + efficiency * c_h2_virtual  # bei den Werten die kleiner 0 sind lohnt es sich eh nicht zu produzieren, daher wird der AP auf 0 gesetzt
    ap_pos_affr = [x if x <= 0 else 0 for x in ap_pos_affr]
    ap_pos_affr = pd.Series(ap_pos_affr)

    return ap_pos_affr, ap_neg_affr

#Erstellen einer zufälligen Nachfrage in einer bestimmten Range
def get_random_demand(d_min = 0, d_max = 10, num_tsteps = 8760):#Anzahl der zu betrachtenden Zeitschritte in der Optimierung
    random.seed(42) # Bestandteil, damit zufällige Nachfrage bei mehrfacher Ausführung des Codes immer dieselbe Nachfrage ist
    demand_h2 = [random.randint(d_min, d_max) for _ in range(num_tsteps)]

    return demand_h2

#Definition der Parameter für das Teillastverhalten des Elektrolyseurs
def part_load_data(P_in_max=20, P_in_min = 2,
                   eta_h2_max=0.5, eta_h2_min=0.6,
                   P_in_max_2=2, P_in_min_2=0,
                   eta_h2_max_2=0.6, eta_h2_min_2=0):

    eta_heat_max = 1-eta_h2_max
    eta_heat_min = 1-eta_h2_min
    eta_heat_max_2 = 1-eta_h2_max_2
    eta_heat_min_2 = 1-eta_h2_min_2

    return P_in_max, P_in_min, eta_h2_max, eta_h2_min, eta_heat_max, eta_heat_min, P_in_max_2, P_in_min_2, eta_h2_max_2, eta_h2_min_2, eta_heat_max_2, eta_heat_min_2


#Hilfsfunktion, um Funktion aus oemof.solph zu nutzen
def get_slope_offset(P_in_max, P_in_min, eta_at_max, eta_at_min):
    slope, offset = solph.components._offset_converter.slope_offset_from_nonconvex_input(P_in_max, P_in_min,
                                                                                         eta_at_max=eta_at_max,
                                                                                         eta_at_min=eta_at_min)
    return slope, offset


#Nutzen der Funktionen get_slope_offset und part_load_data, um Teillastwirkungsgrad zu berechnen
def part_load():
    p_in_max, p_in_min, eta_h2_max, eta_h2_min, eta_heat_max, eta_heat_min, p_in_max_2, p_in_min_2, eta_h2_max_2, eta_h2_min_2, eta_heat_max_2, eta_heat_min_2 = part_load_data()
    # slope und offset for part-load behavior hydrogen, heat, oxygen
    slope_h2, offset_h2 = get_slope_offset(P_in_max=p_in_max, P_in_min=p_in_min, eta_at_max=eta_h2_max, eta_at_min=eta_h2_min)
    slope_heat, offset_heat = get_slope_offset(P_in_max=p_in_max, P_in_min=p_in_min, eta_at_max=eta_heat_max, eta_at_min=eta_heat_min)

    # slope und offset for part-load behavior hydrogen, heat, oxygen
    slope_h2_2, offset_h2_2 = get_slope_offset(P_in_max=p_in_max_2, P_in_min=p_in_min_2, eta_at_max=eta_h2_max_2, eta_at_min=eta_h2_min_2)
    slope_heat_2, offset_heat_2 = get_slope_offset(P_in_max=p_in_max_2, P_in_min=p_in_min_2, eta_at_max=eta_heat_max_2, eta_at_min=eta_heat_min_2)

    return slope_h2, offset_h2, slope_heat, offset_heat, slope_h2_2, offset_h2_2, slope_heat_2, offset_heat_2