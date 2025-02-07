from oemof import solph
import pandas as pd
import numpy as np
import pyomo.environ as po
import datetime as dt

from mit_Speicher_mit_Regelenergie.Inputdaten import get_lp, abruf_affr, get_ap, get_annualized_cost, get_el_price, get_random_demand, part_load, \
    part_load_data, get_storage_data

# Abrufen aller Funktionen aus Inputdaten.py, um Ausgangsparameter zu definieren
annualized_cost, min_rate = get_annualized_cost()
c_el = get_el_price()
demand_h2 = get_random_demand()
slope_h2, offset_h2, slope_heat, offset_heat, slope_h2_2, offset_h2_2, slope_heat_2, offset_heat_2 = part_load()

(el_storage_capacity, el_storage_input_flow, el_storage_output_flow, el_storage_loss_rate, el_storage_variable_costs,
el_storage_initial_storage_level, hydrogen_storage_capacity, hydrogen_storage_input_flow, hydrogen_storage_output_flow,
hydrogen_storage_variable_costs, hydrogen_storage_loss_rate, hydrogen_storage_initial_storage_level) = get_storage_data()

P_in_max, P_in_min, eta_h2_max, eta_h2_min, eta_heat_max, eta_heat_min, P_in_max_2, P_in_min_2, eta_h2_max_2, eta_h2_min_2, eta_heat_max_2, eta_heat_min_2 = part_load_data()

lp_pos_affr, lp_neg_affr = get_lp()
ap_pos_affr, ap_neg_affr = get_ap(c_el=c_el, c_h2_virtual=60)
b_pos, b_neg = abruf_affr()

# Definition noch relevanter Parameter
power_ely = P_in_max
c_h2 = 0
c_heat = 0 # Preis für den Verkauf von Wärme (muss negativ angegeben werden
c_oxygen = 0

num_tsteps = 5 # Anzahl an Zeitschritten, für die letztendlich Ergebnisse vorliegen werden

# Anzahl an Zeitschritten, die bei der Optimierung betrachtet werden sollen
# die Produktion wird für die nächsten 100 Zeitschritte optimiert
window_size = 100

# Definition von Listen für spätere Auswertung
opt1_input_ely_1_1 = []
opt1_input_ely_1_2 = []
input_ely_1_1 = []
output_ely_1_1 = []
input_ely_1_2 = []
output_ely_1_2 = []
nachfrage = []
vorhalten_neg_affr = []
abrufen_neg_affr = []
hydrogen_neg_affr = []
neg_signal = []
vorhalten_pos_affr = []
abrufen_pos_affr = []
pos_signal = []
storage_content_h2 = []
input_h2_storage = []
input2_h2_storage = []
output_h2_storage = []
output2_h2_storage = []
storage_content_el = []
input_el_storage = []
input2_el_storage = []
output_el_storage = []
output2_el_storage = []
time = []
strombezug=[]
menge_abwärme = []
menge_sauerstoff=[]
menge_wasser=[]




for n in range(num_tsteps):

    start_idx = n
    end_idx = min(n + window_size, num_tsteps)

    # Anpassen der Preise und der Nachfrage an Index der for-Schleife
    c_el_angepasst = c_el.iloc[start_idx:end_idx].reset_index(drop=True)
    lp_pos_affr_angepasst = lp_pos_affr.iloc[start_idx:end_idx].reset_index(drop=True)
    lp_neg_affr_angepasst = lp_neg_affr.iloc[start_idx:end_idx].reset_index(drop=True)
    demand_h2_angepasst = demand_h2[start_idx:end_idx]

    # Hier wird der Startzeitpunkt in jeder Iteration um eine Stunde nach hinten verschoben - passend zu den Listen
    start_time = dt.datetime(2019, 1, 1, 0, 0) + dt.timedelta(hours=n)
    # definition of time index
    datetime_index = solph.create_time_index(number=end_idx - start_idx, start=start_time)

    #erstellen des Energiesystems
    es2 = solph.EnergySystem(timeindex=datetime_index, infer_last_interval=False)

    # Definition Bus-Components
    b_el = solph.Bus("electricity bus")
    b_h2 = solph.Bus("hydrogen bus")
    b_heat = solph.Bus("heat bus")
    b_o2 = solph.Bus("oxygen bus")
    b_h2o = solph.Bus("water bus")

    #Hilskomponenten für Regelenergie
    b_el_neg_affr_virt = solph.Bus("neg affr virt electricity bus")
    b_h2_neg_affr_virt = solph.Bus("neg affr virt hydrogen bus")
    b_el_pos_affr_virt = solph.Bus("pos affr virt electricity bus")
    b_h2_pos_affr_virt = solph.Bus("pos affr virt hydrogen bus")

    ##### Definition der Komponenten #####
    # electricity source for basic hydrogen demand
    source_el = solph.components.Source(
        "electricity import",
        outputs={
            b_el: solph.Flow(
                variable_costs=c_el_angepasst #Strompreis in €/MWh
            )
        }
    )

    source_h2o = solph.components.Source(
        "water import",
        outputs={
            b_h2o: solph.Flow(
                variable_costs=0.0015  # €/l Wasserpreis
            )
        }
    )

    source_o2 = solph.components.Source(
        "oxygen import",
        outputs={
            b_o2: solph.Flow(
            )
        }
    )

    source_el_neg_affr_virt = solph.components.Source(
        "electricity neg affr virt",
        outputs={
            b_el_neg_affr_virt: solph.Flow(
                nominal_value=P_in_max,
                min=min_rate, #minimaler Fluss, um Mindestmenge an Regelenergie zu definieren
                nonconvex=solph.NonConvex(
                )
            )
        }
    )

    source_el_pos_affr_virt = solph.components.Source(
        "electricity pos affr virt",
        outputs={
            b_el_pos_affr_virt: solph.Flow(
                nominal_value=P_in_max,
                min=min_rate, #minimaler Fluss, um Mindestmenge an Regelenergie zu definieren
                nonconvex=solph.NonConvex(

                )
            )
        }
    )

    # Sink for fix haydrogen demand via contract
    sink_h2_demand = solph.components.Sink(
        "hydrogen demand",
        inputs={
            b_h2: solph.Flow(
                fix=demand_h2_angepasst, # Wasserstoffnachfrage in MWh
                nominal_value=1, # muss aktiv sein, wenn Nachfrage aktiv ist
                # variable_costs=c_h2_virtual # virtueller Wasserstoffpreis - nur relevant, wenn keine Wasserstoffnachfrage vorliegt
            )
        }
    )

    # sink for byproduct heat
    sink_heat = solph.components.Sink(
        "heat export",
        inputs={
            b_heat: solph.Flow(
                variable_costs=c_heat # Preis für Verkauf von Wärme in €/MWh
            )
        }
    )

    # sink for byproduct oxygen
    sink_o2 = solph.components.Sink(
        "oxygen export",
        inputs={
            b_o2: solph.Flow(
                variable_costs=c_oxygen # Preis für Verkauf von Sauertoff in €/kg
            )
        }
    )

    sink_h2o = solph.components.Sink(
        "water export",
        inputs={
            b_h2o: solph.Flow(
            )
        }
    )

    sink_h2_neg_affr_virt = solph.components.Sink(
        "neg affr h2 sink virt",
        inputs={
            b_h2_neg_affr_virt: solph.Flow(
                variable_costs=-lp_neg_affr_angepasst  # price for keeping neg. balancing energy available
            )
        }
    )

    sink_h2_pos_affr_virt = solph.components.Sink(
        "pos affr h2 sink virt",
        inputs={
            b_h2_pos_affr_virt: solph.Flow(
                variable_costs=-lp_pos_affr_angepasst  # price for keeping neg. balancing energy available
            )
        }
    )

    #### Electrolyzer hydrogen market ####
    # firt part electrolyzer to cover hydrogen demand/production
    electrolyzer1_1 = solph.components.OffsetConverter(
        label='electrolyzer market 1',
        inputs={
            b_el: solph.Flow(
                nominal_value=P_in_max,
                nonconvex=solph.NonConvex(),
                min=P_in_min / P_in_max,
            )
        },
        outputs={
            b_heat: solph.Flow(),
            b_h2: solph.Flow(),
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

    # firt part electrolyzer to cover hydrogen demand/production
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
            b_h2: solph.Flow(),
        },
        conversion_factors={
            b_heat: slope_heat_2,
            b_h2: slope_h2_2,
        },
        normed_offsets={
            b_heat: offset_heat_2,
            b_h2: offset_h2_2,
        }
    )

    #### Electrolyzer negative aFRR ####
    electrolyzer2_1 = solph.components.OffsetConverter(
        label='electrolyzer neg affr holding',
        inputs={
            b_el_neg_affr_virt: solph.Flow(
                nominal_value=P_in_max,
                nonconvex=solph.NonConvex(),
                min=P_in_min / P_in_max,
                # max=1
            )
        },
        outputs={
            b_h2_neg_affr_virt: solph.Flow()
        },
        conversion_factors={
            b_h2_neg_affr_virt: slope_h2,
        },
        normed_offsets={
            b_h2_neg_affr_virt: offset_h2,
        }
    )


    electrolyzer2_2 = solph.components.OffsetConverter(
        label='electrolyzer neg affr holding 2',
        inputs={
            b_el_neg_affr_virt: solph.Flow(
                nominal_value=P_in_max_2,
                nonconvex=solph.NonConvex(),
                min=1 / P_in_max_2,  # die 1 steht für 1MW der mindestens für Regelenergie angeboten werden muss
            )
        },
        outputs={
            b_h2_neg_affr_virt: solph.Flow()
        },
        conversion_factors={
            b_h2_neg_affr_virt: slope_h2_2
        },
        normed_offsets={
            b_h2_neg_affr_virt: offset_h2_2
        }
    )

    #### Electrolyzer positive aFRR ####

    electrolyzer3_1 = solph.components.OffsetConverter(
        label='electrolyzer pos affr holding',
        inputs={
            b_el_pos_affr_virt: solph.Flow(
                nominal_value=P_in_max,
                nonconvex=solph.NonConvex(),
                min=P_in_min / P_in_max,
            )
        },
        outputs={
            b_h2_pos_affr_virt: solph.Flow()
        },
        conversion_factors={
            b_h2_pos_affr_virt: slope_h2,
        },
        normed_offsets={
            b_h2_pos_affr_virt: offset_h2,
        }
    )

    electrolyzer3_2 = solph.components.OffsetConverter(
        label='electrolyzer pos affr holding 2',
        inputs={
            b_el_pos_affr_virt: solph.Flow(
                nominal_value=P_in_max_2,
                nonconvex=solph.NonConvex(),
                min=1 / P_in_max_2,
            )
        },
        outputs={
            b_h2_pos_affr_virt: solph.Flow()
        },
        conversion_factors={
            b_h2_pos_affr_virt: slope_h2_2
        },
        normed_offsets={
            b_h2_pos_affr_virt: offset_h2_2
        }
    )

    #### Storages ####
    # electricity storage
    el_storage = solph.components.GenericStorage(
        label="electricity storage",
        nominal_storage_capacity=el_storage_capacity,
        inputs={
            b_el: solph.Flow(
                nominal_value=el_storage_input_flow,
                variable_costs=el_storage_variable_costs,
                nonconvex=solph.NonConvex()
            )
        },
        outputs={
            b_el: solph.Flow(
                nominal_value=el_storage_output_flow
            )
        },
        loss_rate=el_storage_loss_rate,
        initial_storage_level=el_storage_initial_storage_level,
        balanced=False
        # inflow_conversion_factor=0.9,
        # outflow_conversion_factor=0.9
    )

    # hydrogen storage
    h2_storage = solph.components.GenericStorage(
        label="hydrogen storage",
        nominal_storage_capacity=hydrogen_storage_capacity,
        inputs={
            b_h2: solph.Flow(
                nominal_value=hydrogen_storage_input_flow,
                variable_costs=hydrogen_storage_variable_costs,
                nonconvex=solph.NonConvex()
            )
        },
        outputs={
            b_h2: solph.Flow(
                nominal_value=hydrogen_storage_output_flow
            )
        },
        loss_rate=hydrogen_storage_loss_rate,
        initial_storage_level=hydrogen_storage_initial_storage_level,
        balanced=False
        # inflow_conversion_factor=0.9,
        # outflow_conversion_factor=0.9
    )

    es2.add(b_el, b_h2, b_heat, b_o2, b_h2o, b_el_neg_affr_virt, b_h2_neg_affr_virt, b_el_pos_affr_virt,
            b_h2_pos_affr_virt,
            source_el, source_h2o, source_o2, sink_h2_demand, sink_heat, sink_o2, sink_h2o, source_el_neg_affr_virt,
            sink_h2_neg_affr_virt, source_el_pos_affr_virt, sink_h2_pos_affr_virt,
            electrolyzer1_1, electrolyzer1_2, el_storage, h2_storage, electrolyzer2_1, electrolyzer2_2, electrolyzer3_1,
            electrolyzer3_2
            )

    om = solph.Model(es2)

    myblock = po.Block()
    om.add_component("MyBlock", myblock)

    # es kann nur Ely1_1 oder Ely1_2 produzieren, nicht beide gleichzeitig
    solph.constraints.limit_active_flow_count(
        om, "flow count", [(b_el, electrolyzer1_1), (b_el, electrolyzer1_2)], lower_limit=0, upper_limit=1
    )
    # es kann nur Ely2_1 oder Ely2_2 produzieren, nicht beide gleichzeitig
    solph.constraints.limit_active_flow_count(
        om, "flow count2", [(b_el_neg_affr_virt, electrolyzer2_1), (b_el_neg_affr_virt, electrolyzer2_2)],
        lower_limit=0, upper_limit=1
    )
    # es kann nur Ely3_1 oder Ely3_2 produzieren, nicht beide gleichzeitig
    solph.constraints.limit_active_flow_count(
        om, "flow count3", [(b_el_pos_affr_virt, electrolyzer3_1), (b_el_pos_affr_virt, electrolyzer3_2)],
        lower_limit=0, upper_limit=1
    )


    # Funktion zur Bestimmung des Wasserbezugs zur Produktion
    # pro bezogener MWh Strom werden 162l Wasser benötigt
    def water_flow(m, t):
        expr = om.flow[source_h2o, b_h2o, t] == 162 * (
                    om.flow[b_el, electrolyzer1_1, t] + om.flow[b_el, electrolyzer1_2, t])
        return expr


    myblock.water_flow = po.Constraint(om.TIMESTEPS, rule=water_flow)


    # Funktion zur Bestimmung des produzierten Sauerstoffs
    # pro produzierter MWh Wasserstoff werden 240kg Sauerstoff produziert
    def oxygen_flow(m, t):
        expr = om.flow[b_o2, sink_o2, t] == 240 * (
                    om.flow[electrolyzer1_1, b_h2, t] + om.flow[electrolyzer1_2, b_h2, t])
        return expr


    myblock.oxygen_flow = po.Constraint(om.TIMESTEPS, rule=oxygen_flow)


    # Die Produktion von Ely1_1, Ely1_2 und Ely2_1, Ely2_2 muss kleiner/gleich der Nennleistung sein
    def limit_active_flow_count_rule_neg(m, t):
        expr = (om.flow[b_el, electrolyzer1_1, t] + om.flow[b_el, electrolyzer1_2, t] + om.flow[
            b_el_neg_affr_virt, electrolyzer2_1, t] + om.flow[b_el_neg_affr_virt, electrolyzer2_2, t] <= P_in_max)
        return expr


    myblock.limit_active_flow_count_neg = po.Constraint(om.TIMESTEPS, rule=limit_active_flow_count_rule_neg)


    # Es muss mindestens so viel Kapazität in Nutzung sein, wie pos. Regelenergie angeboten wird
    def limit_active_flow_count_rule_pos(m, t):
        expr = om.flow[b_el, electrolyzer1_1, t] + om.flow[b_el, electrolyzer1_2, t] >= om.flow[
            b_el_pos_affr_virt, electrolyzer3_1, t] + om.flow[b_el_pos_affr_virt, electrolyzer3_2, t]
        return expr


    myblock.limit_active_flow_count_pos = po.Constraint(om.TIMESTEPS, rule=limit_active_flow_count_rule_pos)

    # Erstellen eines "help time index", über den eine weiterer TimeIndex erstellt wird, in dem alle zeitpunkte enthalten sind, in denen die vorgehaltene Regelenergiemenge bestimmt wird
    help_datetime_index = solph.create_time_index(year=2019, number=8761)  # num_tsteps
    # DateTimeIndex with start times of the control energy time slices
    decision_times = pd.date_range(help_datetime_index[0], help_datetime_index[-1], freq='4h')


    # Immer zu Beginn eines 4h-Slots kann die angebotene Regelenergiemenge geändert werden, die restlichen 3h muss der selbe Wert angeboten werden
    def time_constraint_neg_affr(m, t):
        if datetime_index[t] not in decision_times:
            if t == 0:
                expr = om.flow[source_el_neg_affr_virt, b_el_neg_affr_virt, t] == electricity_flow_ely2_1
                return expr
            else:
                expr = om.flow[source_el_neg_affr_virt, b_el_neg_affr_virt, t] == om.flow[
                    source_el_neg_affr_virt, b_el_neg_affr_virt, t - 1]
                return expr
        else:
            return po.Constraint.Skip


    myblock.time_constraint_neg_affr = po.Constraint(om.TIMESTEPS, rule=time_constraint_neg_affr)


    def time_constraint_pos_affr(m, t):
        if datetime_index[t] not in decision_times:
            if t == 0:
                expr = om.flow[source_el_pos_affr_virt, b_el_pos_affr_virt, t] == electricity_flow_ely3_1
                return expr
            else:
                expr = om.flow[source_el_pos_affr_virt, b_el_pos_affr_virt, t] == om.flow[
                    source_el_pos_affr_virt, b_el_pos_affr_virt, t - 1]
                return expr
        else:
            return po.Constraint.Skip


    myblock.time_constraint_pos_affr = po.Constraint(om.TIMESTEPS, rule=time_constraint_pos_affr)


    # Sicherstellen, dass für alle Perioden eines Regelenergiezeitlots genug Speicherkapazität vorhanden ist, falls neg. Regelenergie abgerufen wird
    # Der Verfügbare Speicherplatz muss größer sein, als die vorgehaltene Leistung an Regelenergie (mal Wirkungsgrad) und die produzierte Menge Wasserstoff abzüglich der Nachfrage
    # Da hierbei die Reihenfolge der Produktion und Nachfrage entscheidend sein kann, wird je eine Bedingung für die nächsten vier Stunden definiert
    def min_storage_capa(m, t):
        if datetime_index[t] in decision_times:
            expr = (hydrogen_storage_capacity - om.GenericStorageBlock.storage_content[h2_storage, t] >=
                    om.flow[electrolyzer2_1, b_h2_neg_affr_virt, t] + om.flow[electrolyzer2_2, b_h2_neg_affr_virt, t] +
                    om.flow[electrolyzer1_1, b_h2, t] + om.flow[electrolyzer1_2, b_h2, t] -
                    demand_h2_angepasst[t])
            return expr
        else:
            return po.Constraint.Skip


    myblock.min_storage_capa = po.Constraint(om.TIMESTEPS, rule=min_storage_capa)


    def min_storage_capa1(m, t):
        if datetime_index[t] in decision_times:
            if t + 1 < len(om.TIMESTEPS):  # Sicherstellen, dass t+1 im Bereich ist
                expr = (hydrogen_storage_capacity - om.GenericStorageBlock.storage_content[h2_storage, t] >=
                        om.flow[electrolyzer2_1, b_h2_neg_affr_virt, t] * 2 + om.flow[
                            electrolyzer2_2, b_h2_neg_affr_virt, t] * 2 +
                        om.flow[electrolyzer1_1, b_h2, t] + om.flow[electrolyzer1_2, b_h2, t] +
                        om.flow[electrolyzer1_1, b_h2, t + 1] + om.flow[electrolyzer1_2, b_h2, t + 1] -
                        demand_h2_angepasst[t] - demand_h2_angepasst[t + 1])
                return expr
            else:
                # Falls t+1 außerhalb des Bereichs liegt, Constraint für diesen Fall auslassen
                return po.Constraint.Skip
        else:
            return po.Constraint.Skip


    myblock.min_storage_capa1 = po.Constraint(om.TIMESTEPS, rule=min_storage_capa1)


    def min_storage_capa2(m, t):
        if datetime_index[t] in decision_times:
            if t + 2 < len(om.TIMESTEPS):  # Sicherstellen, dass t+2 im Bereich ist
                expr = (hydrogen_storage_capacity - om.GenericStorageBlock.storage_content[h2_storage, t] >=
                        om.flow[electrolyzer2_1, b_h2_neg_affr_virt, t] * 3 + om.flow[
                            electrolyzer2_2, b_h2_neg_affr_virt, t] * 3 +
                        om.flow[electrolyzer1_1, b_h2, t] + om.flow[electrolyzer1_2, b_h2, t] +
                        om.flow[electrolyzer1_1, b_h2, t + 1] + om.flow[electrolyzer1_2, b_h2, t + 1] +
                        om.flow[electrolyzer1_1, b_h2, t + 2] + om.flow[electrolyzer1_2, b_h2, t + 2] -
                        demand_h2_angepasst[t] - demand_h2_angepasst[t + 1] - demand_h2_angepasst[t + 2])
                return expr
            else:
                # Falls t+2 außerhalb des Bereichs liegt, Constraint für diesen Fall auslassen
                return po.Constraint.Skip
        else:
            return po.Constraint.Skip


    myblock.min_storage_capa2 = po.Constraint(om.TIMESTEPS, rule=min_storage_capa2)


    def min_storage_capa3(m, t):
        if datetime_index[t] in decision_times:
            if t + 3 < len(om.TIMESTEPS):  # Sicherstellen, dass t+3 im Bereich ist
                expr = (hydrogen_storage_capacity - om.GenericStorageBlock.storage_content[h2_storage, t] >=
                        om.flow[electrolyzer2_1, b_h2_neg_affr_virt, t] * 4 + om.flow[
                            electrolyzer2_2, b_h2_neg_affr_virt, t] * 4 +
                        om.flow[electrolyzer1_1, b_h2, t] + om.flow[electrolyzer1_2, b_h2, t] +
                        om.flow[electrolyzer1_1, b_h2, t + 1] + om.flow[electrolyzer1_2, b_h2, t + 1] +
                        om.flow[electrolyzer1_1, b_h2, t + 2] + om.flow[electrolyzer1_2, b_h2, t + 2] +
                        om.flow[electrolyzer1_1, b_h2, t + 3] + om.flow[electrolyzer1_2, b_h2, t + 3] -
                        demand_h2_angepasst[t] - demand_h2_angepasst[t + 1] - demand_h2_angepasst[t + 2] -
                        demand_h2_angepasst[t + 3])
                return expr
            else:
                # Falls t+3 außerhalb des Bereichs liegt, Constraint für diesen Fall auslassen
                return po.Constraint.Skip
        else:
            return po.Constraint.Skip


    myblock.min_storage_capa3 = po.Constraint(om.TIMESTEPS, rule=min_storage_capa3)


    # Die Produktion von Ely1_1/Ely1_2 plus den Speicherfüllstand abzüglich der pos. Regelenergie muss mindestens so groß sein, wie die Nachfrage
    # damit soll sichergestellt werden, dass die Nachfrage gedeckt ist auch wenn pos. Regelenergie abgerufen wird
    def production_constraint_pos_affr(m, t):
        expr = (om.flow[electrolyzer1_1, b_h2, t] + om.flow[electrolyzer1_2, b_h2, t] -
                om.flow[electrolyzer3_1, b_h2_pos_affr_virt, t] - om.flow[electrolyzer3_2, b_h2_pos_affr_virt, t] +
                om.GenericStorageBlock.storage_content[h2_storage, t] * (1 - hydrogen_storage_loss_rate) >=
                demand_h2_angepasst[t])  #
        return expr


    myblock.production_constraint_pos_affr = po.Constraint(om.TIMESTEPS, rule=production_constraint_pos_affr)


    def production_constraint_pos_affr1(m, t):
        if datetime_index[t] in decision_times:
            if t + 1 < len(om.TIMESTEPS):  # Sicherstellen, dass t+1 im Bereich ist
                expr = (om.flow[electrolyzer1_1, b_h2, t] + om.flow[electrolyzer1_2, b_h2, t] +
                        om.flow[electrolyzer1_1, b_h2, t + 1] + om.flow[electrolyzer1_2, b_h2, t + 1] +
                        om.GenericStorageBlock.storage_content[h2_storage, t] * (1 - hydrogen_storage_loss_rate) -
                        om.flow[electrolyzer3_1, b_h2_pos_affr_virt, t] * 2 - om.flow[
                            electrolyzer3_2, b_h2_pos_affr_virt, t] * 2 >=
                        demand_h2_angepasst[t] + demand_h2_angepasst[t + 1])
                return expr
            else:
                # Falls t+1 außerhalb des Bereichs liegt, Constraint für diesen Fall auslassen
                return po.Constraint.Skip
        else:
            return po.Constraint.Skip


    myblock.production_constraint_pos_affr1 = po.Constraint(om.TIMESTEPS, rule=production_constraint_pos_affr1)


    def production_constraint_pos_affr2(m, t):
        if datetime_index[t] in decision_times:
            if t + 2 < len(om.TIMESTEPS):  # Sicherstellen, dass t+2 im Bereich ist
                expr = (om.flow[electrolyzer1_1, b_h2, t] + om.flow[electrolyzer1_2, b_h2, t] +
                        om.flow[electrolyzer1_1, b_h2, t + 1] + om.flow[electrolyzer1_2, b_h2, t + 1] +
                        om.flow[electrolyzer1_1, b_h2, t + 2] + om.flow[electrolyzer1_2, b_h2, t + 2] +
                        om.GenericStorageBlock.storage_content[h2_storage, t] * (1 - hydrogen_storage_loss_rate) ** 2 -
                        om.flow[electrolyzer3_1, b_h2_pos_affr_virt, t] * 3 - om.flow[
                            electrolyzer3_2, b_h2_pos_affr_virt, t] * 3 >=
                        demand_h2_angepasst[t] + demand_h2_angepasst[t + 1] + demand_h2_angepasst[t + 2])
                return expr
            else:
                # Falls t+2 außerhalb des Bereichs liegt, Constraint für diesen Fall auslassen
                return po.Constraint.Skip
        else:
            return po.Constraint.Skip


    myblock.production_constraint_pos_affr2 = po.Constraint(om.TIMESTEPS, rule=production_constraint_pos_affr2)


    def production_constraint_pos_affr3(m, t):
        if datetime_index[t] in decision_times:
            if t + 3 < len(om.TIMESTEPS):  # Sicherstellen, dass t+3 im Bereich ist
                expr = (om.flow[electrolyzer1_1, b_h2, t] + om.flow[electrolyzer1_2, b_h2, t] +
                        om.flow[electrolyzer1_1, b_h2, t + 1] + om.flow[electrolyzer1_2, b_h2, t + 1] +
                        om.flow[electrolyzer1_1, b_h2, t + 2] + om.flow[electrolyzer1_2, b_h2, t + 2] +
                        om.flow[electrolyzer1_1, b_h2, t + 3] + om.flow[electrolyzer1_2, b_h2, t + 3] +
                        om.GenericStorageBlock.storage_content[h2_storage, t] * (1 - hydrogen_storage_loss_rate) ** 3 -
                        om.flow[electrolyzer3_1, b_h2_pos_affr_virt, t] * 4 - om.flow[
                            electrolyzer3_2, b_h2_pos_affr_virt, t] * 4 >=
                        demand_h2_angepasst[t] + demand_h2_angepasst[t + 1] + demand_h2_angepasst[t + 2] +
                        demand_h2_angepasst[t + 3])
                return expr
            else:
                # Falls t+3 außerhalb des Bereichs liegt, Constraint für diesen Fall auslassen
                return po.Constraint.Skip
        else:
            return po.Constraint.Skip


    myblock.production_constraint_pos_affr3 = po.Constraint(om.TIMESTEPS, rule=production_constraint_pos_affr3)

    # lösen des Optimierungsproblems
    om.solve("gurobi")

    results = solph.views.convert_keys_to_strings(om.results(), keep_none_type=True)

    # Werte, die nach der ersten Optimierung an die zweite Optimierung (Auswertung) übergeben werden
    # es werden nur die Werte der ersten Stunde der Optimierung gespeichert, da für die Stunde in der nächsten Optimierung geprüft wird,
    # ob Regelenergie abgerufen wird
    electricity_flow_ely1_1 = results[("electricity bus", "electrolyzer market 1")]["sequences"]["flow"].iloc[0]
    electricity_flow_ely1_2 = results[("electricity bus", "electrolyzer market 2")]["sequences"]["flow"].iloc[0]
    electricity_flow_ely2_1 = \
    results[("neg affr virt electricity bus", "electrolyzer neg affr holding")]["sequences"]["flow"].iloc[0]
    electricity_flow_ely2_2 = \
    results[("neg affr virt electricity bus", "electrolyzer neg affr holding 2")]["sequences"]["flow"].iloc[0]
    electricity_flow_ely3_1 = \
    results[("pos affr virt electricity bus", "electrolyzer pos affr holding")]["sequences"]["flow"].iloc[0]
    electricity_flow_ely3_2 = \
    results[("pos affr virt electricity bus", "electrolyzer pos affr holding 2")]["sequences"]["flow"].iloc[0]

    storage_content_el_storage = results[("electricity storage", None)]["sequences"]["storage_content"].iloc[
                                     0] / el_storage_capacity
    storage_content_h2_storage = results[("hydrogen storage", None)]["sequences"]["storage_content"].iloc[
                                     0] / hydrogen_storage_capacity

    print(n)

    # definition of time index
    datetime_index = solph.create_time_index(number=1, start=start_time)

    es3 = solph.EnergySystem(timeindex=datetime_index, infer_last_interval=False)

    # Definition Bus-Components
    b_el = solph.Bus("electricity bus")
    b_h2 = solph.Bus("hydrogen bus")
    b_heat = solph.Bus("heat bus")
    b_o2 = solph.Bus("oxygen bus")
    b_h2o = solph.Bus("water bus")

    ##### Definition der Komponenten #####

    # electricity source for basic hydrogen demand
    source_el = solph.components.Source(
        "electricity import",
        outputs={
            b_el: solph.Flow(
                variable_costs=c_el_angepasst
            )
        }
    )

    source_h2o = solph.components.Source(
        "water import",
        outputs={
            b_h2o: solph.Flow(
                variable_costs=0.0015  # €/l Wasser
            )
        }
    )

    source_o2 = solph.components.Source(
        "oxygen import",
        outputs={
            b_o2: solph.Flow(
            )
        }
    )

    # Sink for fix haydrogen demand via contract
    sink_h2_demand = solph.components.Sink(
        "hydrogen demand",
        inputs={
            b_h2: solph.Flow(
                fix=demand_h2_angepasst,
                nominal_value=1,
                # variable_costs=c_h2_virtual
            )
        }
    )

    # sink for byproduct heat
    sink_heat = solph.components.Sink(
        "heat export",
        inputs={
            b_heat: solph.Flow(
                variable_costs=c_heat
            )
        }
    )

    # sink for byproduct oxygen
    sink_o2 = solph.components.Sink(
        "oxygen export",
        inputs={
            b_o2: solph.Flow(
                variable_costs=c_oxygen
            )
        }
    )

    sink_h2o = solph.components.Sink(
        "water export",
        inputs={
            b_h2o: solph.Flow(
            )
        }
    )

    #### Electrolyzer hydrogen market ####
    # firt part electrolyzer to cover hydrogen demand/production
    electrolyzer1_1 = solph.components.OffsetConverter(
        label='electrolyzer market 1',
        inputs={
            b_el: solph.Flow(
                nominal_value=P_in_max,
                nonconvex=solph.NonConvex(),
                min=P_in_min / P_in_max,  #
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

    # firt part electrolyzer to cover hydrogen demand/production
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
            b_h2: offset_h2_2
        }
    )

    #### Storages ####
    # battery storage
    el_storage = solph.components.GenericStorage(
        label="electricity storage",
        nominal_storage_capacity=el_storage_capacity,
        inputs={
            b_el: solph.Flow(
                nominal_value=el_storage_input_flow,
                variable_costs=el_storage_variable_costs,
                nonconvex=solph.NonConvex()
            )
        },
        outputs={
            b_el: solph.Flow(
                nominal_value=el_storage_output_flow
            )
        },
        loss_rate=el_storage_loss_rate,
        initial_storage_level=storage_content_el_storage,
        balanced=False
        # inflow_conversion_factor=0.9,
        # outflow_conversion_factor=0.9
    )

    # hydrogen storage
    h2_storage = solph.components.GenericStorage(
        label="hydrogen storage",
        nominal_storage_capacity=hydrogen_storage_capacity,
        inputs={
            b_h2: solph.Flow(
                nominal_value=hydrogen_storage_input_flow,
                variable_costs=hydrogen_storage_variable_costs,
                nonconvex=solph.NonConvex()
            )
        },
        outputs={
            b_h2: solph.Flow(
                nominal_value=hydrogen_storage_output_flow
            )
        },
        loss_rate=hydrogen_storage_loss_rate,
        initial_storage_level=storage_content_h2_storage,
        balanced=False
        # inflow_conversion_factor=0.9,
        # outflow_conversion_factor=0.9
    )

    es3.add(b_el, b_h2, b_heat, b_o2, b_h2o,
            source_el, source_h2o, source_o2, sink_h2_demand, sink_heat, sink_o2, sink_h2o,
            electrolyzer1_1, electrolyzer1_2, el_storage, h2_storage
            )

    om3 = solph.Model(es3)

    myblock3 = po.Block()
    om3.add_component("MyBlock3", myblock3)


    def water_flow(m, t):
        expr = om3.flow[source_h2o, b_h2o, t] == 162 * (
                    om3.flow[b_el, electrolyzer1_1, t] + om3.flow[b_el, electrolyzer1_2, t])
        return expr


    myblock3.water_flow = po.Constraint(om3.TIMESTEPS, rule=water_flow)


    def oxygen_flow(m, t):
        expr = om3.flow[b_o2, sink_o2, t] == 240 * (
                    om3.flow[electrolyzer1_1, b_h2, t] + om3.flow[electrolyzer1_2, b_h2, t])
        return expr


    myblock3.oxygen_flow = po.Constraint(om3.TIMESTEPS, rule=oxygen_flow)


    def flow_ely1_1(m, t):
        if b_neg[n] > 0:
            if electricity_flow_ely1_1 + electricity_flow_ely1_2 + electricity_flow_ely2_1 + electricity_flow_ely2_2 > 2:
                expr = om3.flow[
                           b_el, electrolyzer1_1, t] == electricity_flow_ely1_1 + electricity_flow_ely1_2 + electricity_flow_ely2_1 + electricity_flow_ely2_2
                return expr
            else:
                expr = om3.flow[b_el, electrolyzer1_1, t] == electricity_flow_ely1_1
                return expr
        elif b_pos[n] > 0:
            if electricity_flow_ely1_1 > 0:
                if electricity_flow_ely1_1 - electricity_flow_ely3_1 - electricity_flow_ely3_2 <= 2:
                    expr = om3.flow[b_el, electrolyzer1_1, t] == 0
                    return expr
                else:
                    expr = om3.flow[
                               b_el, electrolyzer1_1, t] == electricity_flow_ely1_1 - electricity_flow_ely3_1 - electricity_flow_ely3_2
                    return expr
            else:
                expr = om3.flow[b_el, electrolyzer1_1, t] == electricity_flow_ely1_1
                return expr
        else:
            expr = om3.flow[b_el, electrolyzer1_1, t] == electricity_flow_ely1_1
            return expr


    myblock3.flow_ely1_1 = po.Constraint(om3.TIMESTEPS, rule=flow_ely1_1)


    def flow_ely1_2(m, t):
        if b_neg[n] > 0:
            if electricity_flow_ely1_1 + electricity_flow_ely1_2 + electricity_flow_ely2_1 + electricity_flow_ely2_2 <= 2:
                expr = om3.flow[
                           b_el, electrolyzer1_2, t] == electricity_flow_ely1_1 + electricity_flow_ely1_2 + electricity_flow_ely2_1 + electricity_flow_ely2_2
                return expr
            else:
                expr = om3.flow[b_el, electrolyzer1_2, t] == 0
                return expr
        elif b_pos[n] > 0:
            if electricity_flow_ely1_1 > 0:
                if electricity_flow_ely1_1 - electricity_flow_ely3_1 - electricity_flow_ely3_2 <= 2:
                    expr = om3.flow[
                               b_el, electrolyzer1_2, t] == electricity_flow_ely1_1 - electricity_flow_ely3_1 - electricity_flow_ely3_2
                    return expr
                else:
                    expr = om3.flow[b_el, electrolyzer1_2, t] == electricity_flow_ely1_2
                    return expr
            else:
                expr = om3.flow[
                           b_el, electrolyzer1_2, t] == electricity_flow_ely1_2 - electricity_flow_ely3_1 - electricity_flow_ely3_2
                return expr
        else:
            expr = om3.flow[b_el, electrolyzer1_2, t] == electricity_flow_ely1_2
            return expr


    myblock3.flow_ely1_2 = po.Constraint(om3.TIMESTEPS, rule=flow_ely1_2)

    # lösen des Optimierungsproblems
    om3.solve("gurobi")

    results3 = solph.views.convert_keys_to_strings(om3.results(), keep_none_type=True)

    # Füllstände der Speicher, die in erste Optimierung wieder übergeben werden
    el_storage_initial_storage_level = results3[("electricity storage", None)]["sequences"]["storage_content"].iloc[
                                           1] / el_storage_capacity
    hydrogen_storage_initial_storage_level = results3[("hydrogen storage", None)]["sequences"]["storage_content"].iloc[
                                                 1] / hydrogen_storage_capacity



    # Sammeln aller Informationen für DataFrame
    time.append(start_time.strftime("%d-%m-%Y %H:%M:%S"))
    # start_time += dt.timedelta(hours=1)

    opt1_input_ely_1_1.append(electricity_flow_ely1_1)
    input_ely_1_1.append(results3[("electricity bus", "electrolyzer market 1")]["sequences"]["flow"].iloc[0])
    output_ely_1_1.append(results3[("electrolyzer market 1", "hydrogen bus")]["sequences"]["flow"].iloc[0])

    opt1_input_ely_1_2.append(electricity_flow_ely1_2)
    input_ely_1_2.append(results3[("electricity bus", "electrolyzer market 2")]["sequences"]["flow"].iloc[0])
    output_ely_1_2.append(results3[("electrolyzer market 2", "hydrogen bus")]["sequences"]["flow"].iloc[0])

    nachfrage.append(demand_h2_angepasst[0])

    vorhalten_neg_affr.append(electricity_flow_ely2_1 + electricity_flow_ely2_2)
    abrufen_neg_affr.append(b_neg[n] * (electricity_flow_ely2_1 + electricity_flow_ely2_2))
    neg_signal.append(b_neg[n])

    vorhalten_pos_affr.append(electricity_flow_ely3_1 + electricity_flow_ely3_2)
    abrufen_pos_affr.append(b_pos[n] * (electricity_flow_ely3_1 + electricity_flow_ely3_2))
    pos_signal.append(b_pos[n])

    storage_content_h2.append(results3[("hydrogen storage", None)]["sequences"]["storage_content"].iloc[0])
    input2_h2_storage.append(results3[("hydrogen bus", "hydrogen storage")]["sequences"]["flow"].iloc[0])
    output2_h2_storage.append(results3[("hydrogen storage", "hydrogen bus")]["sequences"]["flow"].iloc[0])

    storage_content_el.append(results3[("electricity storage", None)]["sequences"]["storage_content"].iloc[0])
    input2_el_storage.append(results3[("electricity bus", "electricity storage")]["sequences"]["flow"].iloc[0])
    output2_el_storage.append(results3[("electricity storage", "electricity bus")]["sequences"]["flow"].iloc[0])

    strombezug.append(results3[("electricity import", "electricity bus")]["sequences"]["flow"].iloc[0])
    menge_abwärme.append(results3[("heat bus", "heat export")]["sequences"]["flow"].iloc[0])
    menge_sauerstoff.append(results3[("oxygen bus", "oxygen export")]["sequences"]["flow"].iloc[0])
    menge_wasser.append(results3[("water import", "water bus")]["sequences"]["flow"].iloc[0])

    print(n)



'''
#Berechnung der LCOH2 ohne Berücksichtigung des Arbeitspreises

cost_el_ely1 = sum(np.array(strombezug)*np.array(c_el))
cost_water =  sum(menge_wasser) * 0.0015
einnahmen_pos_afrr = sum(vorhalten_pos_affr*lp_pos_affr)
einnahmen_neg_afrr = sum(vorhalten_neg_affr*lp_neg_affr)
einnahmen_abwärme = sum(np.array(menge_abwärme) * np.array(c_heat))
einnahmen_sauerstoff = sum(np.array(menge_sauerstoff) * np.array(c_oxygen))
total_cost_el = cost_el_ely1 + annualized_cost + cost_water - einnahmen_pos_afrr - einnahmen_neg_afrr #+ (1/3)*einnahmen_abwärme #+ 0.5*einnahmen_sauerstoff
produced_h2 = sum(np.array(output_ely_1_1)+np.array(output_ely_1_2))
lcoh2 = total_cost_el/produced_h2
'''



'''
#Listen in DataFrame umwandeln
df4 = pd.DataFrame()
df4['Zeit'] = time[:]
df4['Input Ely1 ohne affr'] = opt1_input_ely_1_1
df4['Input Flow Ely 1 [MWh]'] = input_ely_1_1
df4['Output Flow Ely 1 [MWh]'] = output_ely_1_1
df4['Input Ely1_2 ohne affr'] = opt1_input_ely_1_2
df4['Input Flow Ely 1_2 [MWh]'] = input_ely_1_2
df4['Output Flow Ely 1_2 [MWh]'] = output_ely_1_2

df4['feste H2-Nachfrage'] = nachfrage

df4['neg. aFFR vorgehalten'] = vorhalten_neg_affr
df4['neg. aFFR abgerufen'] = abrufen_neg_affr
df4['b_neg'] = neg_signal

df4['pos. aFFR vorgehalten'] = vorhalten_pos_affr
df4['pos. aFFR abgerufen'] = abrufen_pos_affr
df4['b_pos'] = pos_signal

df4['hydrogen storage content'] = storage_content_h2
df4['Input Flow 2 Hydrogen Storage'] = input2_h2_storage
df4['Output Flow 2 Hydrogen Storage'] = output2_h2_storage

df4['electricity storage content'] = storage_content_el
df4['Input Flow 2 Electricity Storage'] = input2_el_storage
df4['Output Flow 2 Electricity Storage'] = output2_el_storage

df4['Strombezug'] = strombezug
df4['Menge Abwärme'] = menge_abwärme
df4['Menge Sauerstoff'] =menge_sauerstoff
df4['Menge Wasser'] = menge_wasser

'''

