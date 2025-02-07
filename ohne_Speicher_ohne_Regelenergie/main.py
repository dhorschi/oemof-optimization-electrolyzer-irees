from oemof import solph
import pandas as pd
import numpy as np
import pyomo.environ as po
#import random
import datetime as dt
from ohne_Speicher_ohne_Regelenergie.Inputdaten import get_annualized_cost, get_el_price, get_random_demand, part_load, \
    part_load_data

annualized_cost = get_annualized_cost()
c_el = get_el_price()
demand_h2 = get_random_demand()
slope_h2, offset_h2, slope_heat, offset_heat, slope_h2_2, offset_h2_2, slope_heat_2, offset_heat_2 = part_load()
P_in_max, P_in_min, eta_h2_max, eta_h2_min, eta_heat_max, eta_heat_min, P_in_max_2, P_in_min_2, eta_h2_max_2, eta_h2_min_2, eta_heat_max_2, eta_heat_min_2 = part_load_data()

#c_h2 = 0
c_heat = 0 #Preis für Verkauf von Abwärme
c_oxygen = 0 #Preis für Verkauf von Sauerstoff
num_tsteps = 8760
power_ely = P_in_max




# In der Funktion wird ein Energiesystem zur Optimierung erstellt
def find_min_lcoh2(c_h2_virtual):

    start_time = dt.datetime(2019, 1, 1, 0, 0, 0)  # festlegen von Startzeitpunkt der Optimierung
    datetime_index = solph.create_time_index(number=num_tsteps, start=start_time)

    ## Bau des Energiesystems
    # Definition des Energiesystems mit zuvor definiertem TimeIndex
    es2 = solph.EnergySystem(timeindex=datetime_index, infer_last_interval=False)

    # Definition der Bus-Components
    b_el = solph.Bus("electricity bus")
    b_h2 = solph.Bus("hydrogen bus")
    b_heat = solph.Bus("heat bus")
    b_o2 = solph.Bus("oxygen bus")
    b_h2o = solph.Bus("water bus")

    # Definiton einer beliebeigen Stromquelle
    source_el = solph.components.Source(
        "electricity import",
        outputs={
            b_el: solph.Flow(
                variable_costs=c_el # Strompreis
            )
        }
    )

    # Definition einer Wasserquelle
    source_h2o = solph.components.Source(
        "water import",
        outputs={
            b_h2o: solph.Flow(
                variable_costs=0.0015  # €/l Wasser - fester Wasserpreis
            )
        }
    )

    # Definition einer Hilfsquelle für die Modellierung von Sauerstoff
    source_o2 = solph.components.Source(
        "oxygen import",
        outputs={
            b_o2: solph.Flow(
            )
        }
    )

    # Sink für Wasserstoffabnahme - inklusive Nachfrage
    # Hier wird festgelegt, ob die Optimierung mit oder ohne Nachfrage stattfinden soll
    sink_h2_demand = solph.components.Sink(
        "hydrogen demand",
        inputs={
            b_h2: solph.Flow(
                #fix=demand_h2, # Auskommentierung entfernen, um Nachfrage zu berücksichtigen
                #nominal_value=1, # muss aktiv sein, wenn Nachfrage betrachtet wird
                variable_costs=-c_h2_virtual # imaginärer Wasserstoffpreis für die Optimierung ohne Nachfrage
            )
        }
    )

    # sink für Wärmeabnahme
    sink_heat = solph.components.Sink(
        "heat export",
        inputs={
            b_heat: solph.Flow(
                variable_costs=c_heat # potentieller Preis für die Abnahme von Wärme
            )
        }
    )

    # sink for byproduct oxygen
    sink_o2 = solph.components.Sink(
        "oxygen export",
        inputs={
            b_o2: solph.Flow(
                variable_costs=c_oxygen # potentieller Preis für die Abnahme von Sauerstoff
            )
        }
    )

    # Sink als Hilfskomponente für die Betrachtung des Wasserbedarfs
    sink_h2o = solph.components.Sink(
        "water export",
        inputs={
            b_h2o: solph.Flow(
            )
        }
    )

    #### Definition Elektrolyseur zur Wasserstoffproduktion ####
    ## Es werden zwei OffsertConverter definiert, um einen Elektrolyseur mit einem variablen Wirkungsgrad abzudbilden
    # Erster Teilelektrolyseur für den Lastbereich P_in_min bis P_in_max
    electrolyzer1_1 = solph.components.OffsetConverter(
        label='electrolyzer market 1',
        inputs={
            b_el: solph.Flow(
                nominal_value=P_in_max,
                nonconvex=solph.NonConvex(),
                min=P_in_min / P_in_max,
                # full_load_time_min=5000,
                # full_load_time_max=5000
            )
        },
        outputs={
            b_heat: solph.Flow(),
            b_h2: solph.Flow()
        },
        conversion_factors={
            b_heat: slope_heat,
            b_h2: slope_h2
        },
        normed_offsets={
            b_heat: offset_heat,
            b_h2: offset_h2
        }
    )

    # Zweiter Teilelektrolyseur für den Lastbereich P_in_min_2 bis P_in_max_2
    electrolyzer1_2 = solph.components.OffsetConverter(
        label='electrolyzer market 2',
        inputs={
            b_el: solph.Flow(
                nominal_value=P_in_max_2,
                nonconvex=solph.NonConvex(),
                min=P_in_min_2 / P_in_max_2,
            )
        },
        outputs={
            b_heat: solph.Flow(),
            b_h2: solph.Flow()
        },
        conversion_factors={
            b_heat: slope_heat_2,
            b_h2: slope_h2_2
        },
        normed_offsets={
            b_heat: offset_heat_2,
            b_h2: offset_h2_2,
        }
    )

    # Komponenten dem Energiesystem hinzufügen
    es2.add(b_el, b_h2, b_heat, b_o2, b_h2o,
            source_el, source_h2o, source_o2, sink_h2_demand, sink_heat, sink_o2, sink_h2o,
            electrolyzer1_1, electrolyzer1_2
            )

    om = solph.Model(es2)

    # Definieren eines pyomo-Blocks um zusätzliche Constraints zu definieren
    myblock = po.Block()
    om.add_component("MyBlock", myblock)

    # Damit beide OffsetConverter einen Elektrolyseur abbilden, wird mit dieser Constraint sichergestellt, dass
    # entweder nur Ely1_1 oder Ely1_2 in einem Zeitschritt produzieren kann, nicht beide gleichzeitig
    solph.constraints.limit_active_flow_count(
        om, "flow count", [(b_el, electrolyzer1_1), (b_el, electrolyzer1_2)], lower_limit=0, upper_limit=1
    )

    # Es wird definiert, dass für den Bezug von einer MWh Strom 162 Liter Wasser bezogen werden
    def water_flow(m, t):
        expr = om.flow[source_h2o, b_h2o, t] == 162 * om.flow[source_el, b_el, t]
        return expr

    myblock.water_flow = po.Constraint(om.TIMESTEPS, rule=water_flow)

    # Es wird definiert, dass bei der Produktion von 1 MWh Wasserstoff 240 kg Sauerstoff produziert werden (resultiert aus dem Massenverhältnis)
    def oxygen_flow(m, t):
        expr = om.flow[b_o2, sink_o2, t] == 240 * (
                    om.flow[electrolyzer1_1, b_h2, t] + om.flow[electrolyzer1_2, b_h2, t])
        return expr

    myblock.oxygen_flow = po.Constraint(om.TIMESTEPS, rule=oxygen_flow)

    # lösen des Optimierungsproblems
    om.solve("gurobi")

    # Konvertieren der Ergebnisse der Optimierung
    results = solph.views.convert_keys_to_strings(om.results(), keep_none_type=True)


    ### Sammlung der Ergebnisse
    c_el_neu = list(c_el.copy())
    c_el_neu.append(np.nan)
    demand_h2_neu = list(demand_h2.copy())
    demand_h2_neu.append(np.nan)

    df = pd.DataFrame()
    df['Input Flow Ely 1_1 [MWh]'] = results[("electricity bus", "electrolyzer market 1")]["sequences"]["flow"]
    df['Input Flow Ely 1_2 [MWh]'] = results[("electricity bus", "electrolyzer market 2")]["sequences"]["flow"]
    df['Output Flow Ely 1_1 [MWh]'] = results[("electrolyzer market 1", "hydrogen bus")]["sequences"]["flow"]
    df['Output Flow Ely 1_2 [MWh]'] = results[("electrolyzer market 2", "hydrogen bus")]["sequences"]["flow"]
    df['Efficiency Ely'] = (
            (results[("electrolyzer market 1", "hydrogen bus")]["sequences"]["flow"] +
             results[("electrolyzer market 2", "hydrogen bus")]["sequences"]["flow"]) /
            (results[("electricity bus", "electrolyzer market 1")]["sequences"]["flow"] +
             results[("electricity bus", "electrolyzer market 2")]["sequences"]["flow"]))

    df['Strompreis [€/MWh]'] = c_el_neu
    df['feste H2-Nachfrage'] = demand_h2_neu



    arr = results[("electricity bus", "electrolyzer market 1")]["sequences"]["flow"].values[:-1] + \
          results[("electricity bus", "electrolyzer market 2")]["sequences"]["flow"].values[:-1]
    betriebsstunden = sum(np.array(arr) > 0)
    betrieb_bei_volllast = sum(np.array(arr) == power_ely)

    volllaststunden_el = ((sum(results[("electricity bus", "electrolyzer market 1")]["sequences"]["flow"].values[:-1]) +
                           sum(results[("electricity bus", "electrolyzer market 2")]["sequences"]["flow"].values[
                               :-1])) /
                          power_ely)

    menge_abwärme = sum(results[("electrolyzer market 1", "heat bus")]["sequences"]["flow"].values[:-1] +
                        results[("electrolyzer market 2", "heat bus")]["sequences"]["flow"].values[:-1])
    menge_sauerstoff = sum(results[("electrolyzer market 1", "hydrogen bus")]["sequences"]["flow"].values[:-1] +
                           results[("electrolyzer market 2", "hydrogen bus")]["sequences"]["flow"].values[:-1])*240
    einnahmen_abwärme = sum(results[("electrolyzer market 1", "heat bus")]["sequences"]["flow"].values[:-1] +
                            results[("electrolyzer market 2", "heat bus")]["sequences"]["flow"].values[:-1]) * c_heat
    einnahmen_sauerstoff = sum(results[("electrolyzer market 1", "hydrogen bus")]["sequences"]["flow"].values[:-1] +
                               results[("electrolyzer market 2", "hydrogen bus")]["sequences"]["flow"].values[:-1])*240 * c_oxygen

    # totale variable Kosten werden berechnet, indem die bezogene Strommenge mit den Strompreisen multipliziert werden
    cost_el_ely1 = sum((results[("electricity bus", "electrolyzer market 1")]["sequences"]["flow"].values[:-1] +
                        results[("electricity bus", "electrolyzer market 2")]["sequences"]["flow"].values[:-1]) * c_el)

    cost_water = sum(results[("water import", "water bus")]["sequences"]["flow"].values[:-1]) * 0.0015
    # annualisierte Investitionskosten werden zu den variablen Kosten addiert um die gesamten Kosten zu betrachten
    total_cost_el = cost_el_ely1 + annualized_cost + cost_water - einnahmen_abwärme - einnahmen_sauerstoff
    # Berechnung der produzierten Wasserstoffmenge
    produced_h2 = sum(results[("hydrogen bus", "hydrogen demand")]["sequences"]["flow"].values[:-1])

    # Gestehungskosten werden berechnet, indem die Gesamtkosten durch die Summe der produzierten Menge dividiert wird
    if produced_h2 == 0:
        lcoh2 = total_cost_el
    else:
        lcoh2 = total_cost_el / produced_h2

    df2 = pd.DataFrame()
    df2['LCOH2 [€/MWh]'] = [lcoh2]
    df2['LCOH2 [€/kg]'] = [(lcoh2 * 33.33) / 1000]
    df2['Betriebsstunden'] = [betriebsstunden]
    df2['Betrieb bei Volllast'] = [betrieb_bei_volllast]
    df2['Volllaststunden (elektrisch)'] = [volllaststunden_el]
    df2['jährliche Investitionskosten [€]'] = [annualized_cost]
    df2['Stromkosten [€]'] = [cost_el_ely1]
    df2['Wasserkosten [€]'] = [cost_water]
    df2['Gesamtkosten [€]'] = [total_cost_el]
    df2['produzierte Menge Wasserstoff [MWh]'] = [produced_h2]
    df2['Menge Abwärme [MWh]'] = [menge_abwärme]
    df2['Menge Sauerstoff [t]'] = [menge_sauerstoff]
    df2['Einnahmen Abwärme'] = [einnahmen_abwärme]
    df2['Einnahmen O2'] = [einnahmen_sauerstoff]

    return lcoh2, df, df2


# Funktion um minimale LCOH2 zu finden, wenn keine Nachfrage vorliegt
def get_min_lcoh2(c_h2_virtual_min, c_h2_virtual_max):
    # festlegen von sehr hohen Ausgangsgestehungskosten
    min_lcoh2 = 10000000

    # Die Optimierung wird mit einem imaginären H2-Preis durchgeführt und die optimalen LCOH2 berechnet. Im nächsten Schritt wird der
    # imaginäre H2-Preis um einen Schritt erhöht und die optimialen LCOH2 erneut berechnet. Sind die LCOH2 nun geringer, als
    # bei der ersten Optimierung wird der imaginäre H2-Preis weiter erhöht bis höhere LCOH2 entstehen oder die Schleife zu Ende ist
    for c_h2_virtual in range(c_h2_virtual_min, c_h2_virtual_max):
        lcoh2, df, df2 = find_min_lcoh2(c_h2_virtual) # hier wird die Optimierung des Energiesystems durchgeführt
        print(c_h2_virtual, lcoh2)

        if lcoh2 <= min_lcoh2:
            min_lcoh2 = lcoh2
            min_df = df
            min_df2 = df2
        else:
            break

    return min_df, min_df2

min_df, min_df2 = get_min_lcoh2(100,101)