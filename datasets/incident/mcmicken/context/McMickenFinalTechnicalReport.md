# McMickenFinalTechnicalReport

## 1 Executive Summary

Davion M. Hill, Ph.D., Energy Storage Leader US, DNV GL, was retained by Arizona Public Service Company (APS) as an expert consultant, to provide technical advice and analysis regarding the investigation into a thermal event and subsequent explosion that occurred at the APS McMicken Battery Energy Storage facility. Dr. Hill was asked to prepare this report on behalf of APS, in order to summarize the findings and conclusions of the investigation. This report includes a review of all the reports completed to date by APS’ retained experts, on the critical technical factors related to the battery energy storage system (BESS) failure. This report also includes Dr. Hill’s expert opinion and analysis.

The BESS was commissioned and integrated by AES, on behalf of APS. The BESS was assembled with Lithium ion (Li-ion) batteries manufactured by LG Chem. On April 19, 2019, 25 months after the BESS was placed into service, a suspected fire was reported at the BESS. At 17:48 local time (PST), first responders arrived to investigate. Several hours later, at approximately 20:04, an explosion occurred from inside the BESS. The explosion injured several firefighters and essentially destroyed the BESS and its container.

The factual conclusions reached in this investigation are as follows:

• The suspected fire was actually an extensive cascading thermal runaway event, initiated by an internal cell failure within one battery cell in the BESS: cell pair 7, module 2, rack 15 (battery 7-2).

It is believed to a reasonable degree of scientific certainty that this internal failure was caused by an internal cell defect, specifically abnormal Lithium metal deposition and dendritic growth within the cell.

The total flooding clean agent fire suppression system installed in the BESS operated early in the incident and in accordance with its design. However, clean agent fire suppression systems are designed to extinguish incipient fires in ordinary combustibles. Such systems are not capable of preventing or stopping cascading thermal runaway in a BESS.

As a result, thermal runaway cascaded and propagated from cell 7-2 through every cell and module in Rack 15, via heat transfer. This propagation was facilitated by the absence of adequate thermal barrier protections between battery cells, which may have stopped or slowed the propagation of thermal runaway.

The uncontrolled cascading of thermal runaway from cell-to-cell and then module-to-module in Rack 15 led to the production of a large quantity of flammable gases within the BESS. Analysis and modeling from experts in this investigation confirmed that these gases were sufficient to create a flammable atmosphere within the BESS container.

• Approximately three hours after thermal runaway began, the BESS door was opened by firefighters, agitating the remaining flammable gases, and allowing the gases to make contact with a heat source or spark.

There were five main contributing factors that led to the explosion:

• Contributing Factor #1: Internal failure in a battery cell initiated thermal runaway

• Contributing Factor #2: The fire suppression system was incapable of stopping thermal runaway

• Contributing Factor #3: Lack of thermal barriers between cells led to cascading thermal runaway

• Contributing Factor #4: Flammable off-gases concentrated without a means to ventilate

• Contributing Factor #5: Emergency response plan did not have an extinguishing, ventilation, and entry procedure

This report concludes that today’s standards better address hazard assessment and training for first responders, although the industry expectation should go even further and require that hazard assessments and training take place before and during the commissioning of energy storage systems. In today’s practice, the systems integrator and EPC contractor typically coordinate safety response plans on behalf of the owner, and then train the operations and maintenance (O&M) personnel to execute them. 1

While today’s energy storage safety codes and standards acknowledge cascading thermal runaway as a risk, they stop short of prohibiting it, and fail to address the risk of non-flaming heat transfer to neighboring cells, modules, and racks. Standards today focus on the means to manage a fire, but have so far avoided prescribing solutions that restrict or slow cell-to-cell and module-to-module thermal runaway propagation (likely due to a reticence to prescribe anything that may be perceived as prohibitively expensive or noncommercial). Standards today, therefore, also fall short in addressing the issue and risks associated with off-gassing. However, there are commercially available technologies and design methods available that can address thermal runaway propagation, and the standards should be appropriately updated to acknowledge these methods and technologies. The main codes examined in this report are National Fire Protection Agency (NFPA) 855 and past and current versions of the International Fire Code, along with Underwriters’ Laboratories (UL) 1973, UL 9540, and the UL 9540A test method.

In addition, better practices for ventilation, extinguishing, and cooling thermal runaway scenarios exist today and should be implemented in future energy storage systems. Finally, clean agent systems may still be appropriate for use in energy storage facilities to manage incipient fires, but they must be used in conjunction with additional practices—i.e., ventilation, extinguishing, and cooling—to manage thermal runaway scenarios. Clean agent or aerosol extinguishing methods should not be the only barrier against thermal runaway.

### 1.1 Related McMicken investigations and reports

APS began its investigation on April 22, 2019. APS went to great lengths to preserve the investigation site and perform a full, in depth, objective forensics investigation. As part of its investigation, and as the asset owner, APS coordinated the investigation and actively included representatives, experts, and consultants from AES, Fluence, and LG Chem.

APS retained the services of the following industry experts, to perform certain forensic analyses, scientific modeling, and investigation:

• Wood Environment & Infrastructure Solutions, Inc. (Wood) and AECOM Technical Services, Inc. (AECOM) were hired to perform an environmental assessment of the facility and site, in order to establish whether any environmental hazards (particulate matter, heavy metals, etc.) presented health and human safety concerns.

• Safety Engineering Laboratories, Inc. (SEL) was retained to analyze and forensically inspect the evidentiary data to determine the originating cause of the battery failure.

• Colwell Consulting, LLC (Colwell) was retained to forensically analyze all available data and evidence in an effort to determine the cause of the container explosion.

• CP Fire, LLC (CP Fire) was retained to assess the suitability, functional performance, and corresponding effectiveness of the BESS’s fire suppression system.

DNV GL has reviewed each of the experts’ independent analyses and reports and, in combination with DNV GL’s own subject matter expertise in battery systems and battery system failures, compiled this report and its associated findings.2

Beyond those experts retained by APS, and explicitly referenced above, numerous other third parties (along with their retained experts) were invited to observe and participate throughout the investigation into this event, as described in Section 2.4. However, this report does not include the work or the opinions of any of those third parties (who are listed below), and none of them participated in the development of the findings and conclusions outlined in this report.

1. AES Corporation

SEA Limited

2. Fluence Energy, LLC

Safe Labs

\- Forensic Investigations Group

\- South River Engineering, LLC

3. LG Chem Ltd.

\- Exponent

\- Seneca Fire Engineering

4. City of Peoria and City of Surprise

\- Rimkus Consulting Group, Inc.

Fire department representatives from the City of Surprise and City of Peoria

Fire Protection Parties – The companies listed below each had one or more consultants:

5. Kidde-Fenwal, Inc. (manufacturer of the Kidde Fire System ECS-500)
6. American Fire Equipment, Inc. (designer/installer related to fire suppression system)
7. Intermountain Electronics, Inc. (designer/installer related to fire suppression system)
8. 3M Company (manufacturer of the fire suppressant agent)
9. Honeywell Building Technologies (manufacturer of the VESDA smoke detection system and Notifier Fire Protection Panel)

Other Parties/Government Agencies:

10. Pacific Northwest National Laboratory
11. Sandia National Laboratories
12. Underwriters Laboratories, Inc.

Dr. Hill of DNV GL as well as APS and its retained experts—SEL, Colwell Consulting, and C P Fire—attended a presentation in mid-June 2020, organized by LG Chem and its retained expert, Exponent, regarding LG Chem’s root cause analysis of the McMicken event and preliminary conclusions. Exponent presented its findings and opinions regarding the precipitating cause of the McMicken thermal event. APS and its experts carefully analyzed and studied the information presented and disagree with Exponent’s findings.

## 2 Background

The majority of the background information concerning commissioning and design of the system in the following sections was provided to DNV GL by APS.

According to APS, to facilitate the addition of intermittent, renewable energy on its grid, the company sought to install two distribution level, utility scale battery systems on two separate distribution feeders. APS determined that a 2 MW / 2 MWh battery would be complementary to the APS distribution systems at two locations, McMicken and Festival Ranch. The systems were commissioned to study the effectiveness of energy storage in the following areas:

• Ability to complement and integrate with distributed rooftop solar generation

• Ability to assist in regulating voltage on distribution feeders with heavy concentrations of rooftop solar

• Ability to provide power factor correction to these feeders

• Ability to provide real and imaginary power for distribution system optimization

• Ability to provide energy capacity after sundown

• Ability to perform other stackable and inter-related functions

### 2.1 Procurement and commissioning

On June 7, 2016, APS and The AES Corporation (AES) entered into an agreement wherein AES would design, engineer, procure, construct, and commission the McMicken Battery Energy Storage System (BESS). APS sourced the system from AES by a competitive bid and placed the purchase order on June 7, 2016. As is typical in the energy storage industry, APS evaluated multiple bids from different engineering, procurement, and construction (EPC) contractors for their energy storage systems.

It is characteristic of bidders to offer a combination of systems integration, construction, and operations and maintenance (O&M) services. Some entities can offer this as a full package, while others may manage a number of subcontractors to provide these services. Since 2014-2015, EPC contractors have offered “full wrap” services, which may include the following:

Systems integration: Design of the system including the matching of batteries to inverters, along with racking, containerization, and support systems such as heating, ventilation, and air conditioning (HVAC), fire suppression systems, monitoring, and controls.

• General construction services: This may include site preparation, electrical and civil contractors, structural foundations for the system, and other services such as fencing.

• Operations and maintenance: In addition to routine maintenance of the facility, these services may include certain guarantees for capacity or system availability and performance.

APS ultimately selected AES because of AES’ long tenure in the industry, with a credible reputation to provide a complete package of services as described above.

In September 2016, APS and AES entered into a long-term maintenance and services contract (O&M) for the McMicken BESS. In January 2018, AES and Siemens created Fluence Energy LLC (Fluence) as a joint venture, and in May 2018 AES assigned the long-term O&M contract for the McMicken BESS to Fluence.

APS placed the McMicken BESS into service on March 14, 2017. [46] The general layout of the system is shown in Figure 1. From the date of commissioning until January 2018, APS used the BESS for the various research and development purposes described earlier. Following completion of that research, the BESS was used daily as a load serving resource to charge with solar energy produced during the daytime and discharge the solar energy back onto the grid in the evening when solar systems were no longer producing.

### 2.2 BESS design and component parts

The BESS was a containerized system with all components installed within a container roughly the size and shape of a standard shipping container (50’ long x 13’ wide x 12’ high). Internal components included all the battery modules, battery management systems (BMS), power inverters, HVAC, fire suppression, data capture and monitoring equipment, and other auxiliary components. The only devices external to the container were BESS interconnection apparatuses such as transformers, breakers, and switch gear.

The BESS was configured similar to a computer data center with 36 individual vertical racks separated into 2 rows on either side of a 3-ft hallway. Twenty-seven of the racks each contained 14 battery modules manufactured by LG Chem, an 80 kW inverter manufactured by Parker, an AES Advancion node controller used for data collection and communication sourced from AES, and a Battery Protection Unit (BPU) manufactured by LG Chem. The battery modules contained 28 Li-ion battery cells (14 series, 2 parallel) of Nickel Manganese Cobalt (NMC) chemistry and a module-level BMS. The 14 battery modules were connected in series, which provided a per-rack nominal voltage of 721 V. The entire system in aggregate was specified at nameplate to supply 2 MW of power over 1 hour for a lifetime energy rating of 2 MWh. As is normal for a facility of this nature, there was additional battery capacity built in initially to allow for normal degradation over time without impacting the contracted for 2 MWh rating.

A single rack housed the communications equipment for the system and eight racks were in place to eventually hold additional battery modules in the event of system expansion or for augmentation in order to retain the specified power and energy levels as the original battery modules aged. On the day of the event, these eight racks were being used for some spare components as well as storage of documentation and other small items.

Historically, energy storage systems have been packaged into standardized intermodal shipping containers. In 2019-2020, there has been a gradual shift away from shipping containers in favor of compartmentalized, pod-like, modular cabinets that have ingress protection (IP) ratings and stand-alone BMS and/or inverter capability, such that multiple cabinets can be strung together to build a large system.

All products and vendors used in the construction of the BESS were selected by AES as part of its EPC contractor and system integrator role.

#### 2.2.1 Li-ion battery energy storage systems

Li-ion energy storage systems today comprise battery cells that are wired in series or parallel to form modules. Modules typically have a control system called a battery management system (BMS) that prevents electrical imbalances in the modular assembly of battery cells, such as over- or under-voltage, over-current, and over- or under-temperature. The BMS’ main function is to shut down operations of the module (and sometimes the system) if any of the voltage, current, or temperature boundary conditions are violated.

Battery modules are assembled into larger sub-assemblies with varying names; they can be racks, packs, or sometimes cabinets. These module sub-assemblies are then strung together, usually in strings, which comprise the system. A system may have one or more strings. The architecture of a typical energy storage system (ESS) is shown in Figure 2.3 In the McMicken BESS, the modules were assembled into racks.

The McMicken BESS was built with LG Chem Li-ion battery cells and modules. Li-ion battery cells are classified by several different chemistries.

The most commonly used chemistries in Li-ion stationary energy storage today are Nickel-Manganese-Cobalt (NMC or sometimes NCM) and Li-ion iron phosphate $( \mathsf { L i F e P O } _ { 4 }$ or sometimes abbreviated as LFP). Earlier in the development of Li-ion batteries, the main active ingredient was Cobalt in Lithium-Cobalt (Li-Co) batteries. Other variants include Nickel-Cobalt-Aluminum (NCA) and Lithium Manganese oxide (LMO).

Energy density of Li-ion batteries has evolved over time, with early LFP batteries having an energy density of \~90 Wh/kg, exceeding 120 Wh/kg today, and early NMC, NCA, and LMO batteries having \~150 Wh/kg in 2012 are approaching or exceeding 200 Wh/kg today. Incremental innovation in roll-coating manufacturing, including coating homogeneity, cleanliness, purity, thickness, and speed are improving cathode and anode materials performance. Better, lower density separators and more innovative packaging are also contributing to improved energy density at the cell and system level. The LG Chem batteries in the McMicken BESS were NMC, with an energy density greater than 200 Wh/kg.

Battery cell form factors are commonly pouch, cylindrical, or prismatic. Pouch cells are constructed by vacuum sealing a metalized polymer film around layers of alternating anode-separator-cathode sets. The electrodes in pouch cells exit the pouch in the form of large, thin, flat tabs. The tabs can exit from one side or opposing sides. Cylindrical cells are formed with a “jelly roll” of anode-separator-cathode materials, which are inserted into an aluminum “can” and sealed. Most cylindrical Li-ion cells are a bit larger than a common D-cell household battery. Prismatic cells are constructed in a similar fashion to cylindrical cells, but the “can” is typically shaped like a rectangular prism with a positive and negative terminal post on the top of the battery cell. The LG NMC cells in the BESS were pouch cells.

The LG Chem battery modules used in the BESS were constructed with 28 total pouch cells in a 2P14S configuration, meaning that two battery cells are wired in parallel (P) as a pair and then the 14 pairs are wired in series (S). The physics of Ohm’s and Kirchhoff’s Laws dictate that when two identical batteries are wired in parallel, the current they produce is equal. Thus, for any given current requirement of the module, the current at the cell level would be halved because of this parallel wiring of the battery cells. The voltage of the module is simply the voltage of the cells multiplied by the series number of cells (14, in this case).

These factors are important because they determine the voltage and current conditions of the battery cell, which are directly related to the cell’s charge and discharge output, lifetime, and safety factors. As the owner of the system, APS had no means to change this configuration and therefore the operational parameters of the batteries were left in the “as built” state from LG Chem upon delivery and commissioning by AES.

The LG Chem modules were built from LG’s proprietary NMC battery cells, model JP3. LG manufactured the battery cells and integrated them into modules, and supplied them to AES. The system configuration was as follows [46]:

• 64 Ah, 0.24 kWh pouch cells, operational voltage range 3.0-4.2V

• 28 cells per module in a 2P14S configuration

• 6.7 kWh modules (14 modules per rack see Figure 3)

• 27 fully populated racks in the system, and several additional empty racks for future expansion

• 2.0MW/2.0 MWh nameplate rating for the container

Each rack therefore contained 28\*14 = 392 battery cells. With 27 full racks, there were 10,584 cells in the container.

The system control was based on the LG BMS, which at the system level was called the BPU. APS was unable to alter the hardware or software settings of the system by design, which means the normal operating conditions of the battery cells, modules, and racks were defined by the BMS and BPU protections designed by LG Chem, and these did not change after the system was built and commissioned. Voltage balancing and regulation of current at the battery cell level were therefore an LG Chem BMS responsibility. APS could request the system to perform any function within its design limits, and the BMS was responsible for assuring that the battery cells were managed properly. Control software, communications, and monitoring were integrated by AES.

#### 2.2.2 Fire protection system

The BESS was equipped with a Novec 1230 “total flooding” clean agent fire suppression system. At the time of commissioning, clean agent systems were a common method for managing fire suppression in energy storage containers because clean agents are electrically nonconductive and do not leave a residue. In clean agent systems, the agent is stored under pressure in a large reservoir cylinder as shown in the indicative illustration in the lower left of Figure 4. The figure is illustrative and simplified for the purpose of describing general ESS design principles. The actual Novec 1230 system in the McMicken BESS included a more symmetrical layout based on the location of equipment.

When the fire detection system detects a fire, the fire suppression system discharges the agent through a piping system with nozzles mounted on the ceiling, which emits the agent in a fine mist, dispersing and evaporating in the atmosphere. Clean agents are typically used to extinguish incipient (i.e., small) fires before significant damage occurs to the remaining equipment.

In an August 2017 letter to the NFPA 855 committee during drafts of the NFPA standard, clean agent manufacturer 3M stated that clean agents could not prevent or suppress cascading thermal runaway in Liion battery systems. [44] While this statement was made more than a year after the design of the McMicken BESS, it was not widely circulated. Following the commissioning of the BESS, the comment did serve to inform future designs and standards development.

The McMicken BESS also contained a Very Early Smoke Detection Apparatus (VESDA) laser-based smoke detection system. The VESDA could detect fine particles and diminishing visibility (due to smoke) within a room, by monitoring for any interference of a laser beam. The sensors were placed evenly along the ceiling of the container and were fed to a central power unit with data storage. Upon smoke detection, the VESDA system sent a signal to a Honeywell Notifier RP 2002 fire alarm control and releasing panel, which in turn sent a signal to discharge the Novec 1230 system, manufactured by Kidde.4

### 2.3 Summary of the failure event and subsequent explosion

On the day of the event, the BESS was performing a solar smoothing function, which involved charging during daytime (absorbing solar energy produced from rooftop solar on the circuit) and discharging through the evening load peak. APS collected data during the entire operational period from initial commissioning to the incident, including time-series data on module and battery cell voltages, charge and discharge currents, charge levels, ramp rates, temperature, humidity, and environmental data.

At approximately 16:54 on April 19, 2019, a battery cell in Module 2 of Rack 15 of the BESS experienced a sudden drop in voltage during a charge cycle, equivalent to a single cell voltage drop (4.06 to 3.82 V as shown in Figure 14). Module 2 was in the second module position, numbered upward from the base of the rack.

Moments after the voltage drop occurred, cell 7 in module 2 of rack 15 went into thermal runaway. This event generated off-gassing and smoke that activated the VESDA smoke detection system and led to the discharge of the Novec 1230 agent. The initiating cell’s thermal runaway cascaded into thermal runaway of neighboring cells within module 2 and subsequently into the batteries contained within neighboring modules. The off-gassing of battery cells from the cascading thermal runaway created a flammable atmosphere within the BESS.

Approximately 3 hours after the initial voltage drop and discharge of the fire suppressant, firefighters opened the side container door, and approximately 2-3 minutes after the door was opened, an explosion occurred. The doors on the side and rear of the BESS and other debris were ejected by the explosion as shown in the figures below. The HVAC systems were also significantly damaged.

### 2.4 Investigation procedure

Dr. Hill and DNV GL were not present during any portion of the physical forensics investigation into this event. All information regarding the investigation procedure and third-party involvement in the post-event investigation was obtained from expert reports or provided directly by APS. This information has been included in this report to provide necessary background and context. While DNV GL has no reason to doubt the accuracy of the information provided by APS, it cannot attest to the veracity or accuracy of any information regarding the investigation procedure discussed in this section.

The site investigation and remediation process required that APS preserve the site, while managing numerous parties attempting to perform their own independent investigations. The BESS and its container were significantly damaged by the thermal runaway event and subsequent explosion. Both doors were dislodged from the structure and the BESS container walls were bulging outward.

Given the significant safety hazards discussed below, the participation of multiple interested parties (e.g., AES, Fluence, and LG Chem), the need to develop written procedures for many steps in the site decommissioning and subsequent investigative process, and the need to forensically examine the BESS and its contents, the site investigation and decommissioning spanned close to four months.

To secure and protect the BESS, APS erected a temporary fence around both the site and approximately a ½-acre of surrounding land. The site was monitored by security 24-hours a day.

Firefighters from the Palo Verde Nuclear Generating Station with advanced post-failure training were recruited to acquire 3D images of the site and monitor for toxic and flammable gases. This footage and imagery indicated that Rack 15 had been severely damaged and that the remaining racks likely had voltage on them, as the system was at 90% state of charge when the event occurred. This information led APS and investigating experts to develop a plan to remove and discharge the batteries that likely were still charged.

#### 2.4.1 Data device retrieval

On May 7, 2019, APS retrieved three devices from the BESS, which were suspected to contain useful data for the investigation:

1. VESDA VLF-250 smoke detector (part of the fire suppression system discussed previously)
2. Main server and disk array
3. Schweitzer Engineering Laboratories SEL-751 protection relay (which monitored the fire suppression system, environmental control system, and the ground fault detector, and could trigger the emergency stop circuit and open both AC and DC breakers)

Data from the VESDA VLF-250 was retrieved by a VESDA representative on May 8, 2019. This data showed that the VESDA smoke detection system progressed through all four of its alarm threshold levels within the same minute that smoke was first detected, [52] and deployed the fire suppression agent 30 seconds later, as designed.

The main server disk required data recovery efforts due to its damage and it matched data already captured by APS’ remote monitoring systems. The SEL-751 protection relay did not provide any additional information regarding the operations of the systems it monitored.

#### 2.4.2 Site prep work

APS built a temporary work environment around the BESS, which included power connections, facilities for staff and evidence storage, and a secure, environmentally controlled tent over the entire BESS container (as shown in Figure 10). Prior to the dismantling and discharging of the battery racks and modules, a team had to enter the container and clear access to the critical BESS equipment. This required clearing of debris from the hallway (Figure 9).

#### 2.4.3 Battery discharge

APS and its experts determined from the drone footage and 3D imaging that Rack 15 had severe thermal damage, while the neighboring racks did not have apparent burn or thermal damage, and therefore Rack 15 would be the focus of the investigation. Rack 15 was located at approximately the mid-point of the BESS container.

However, as discussed above, the BESS contained 26 remaining racks, each at approximately 90% state of charge with 90 kWh of stranded energy that would have to be safely discharged before forensic examination of Rack 15 could occur.

APS and its experts, along with participating third parties, were unaware of any established procedures to remove and safely discharge such a large quantity of stranded energy, and spent approximately four weeks outlining a protocol and preparing for the safe removal and discharge of the BESS battery modules.5

The discharge procedure began on May 29, 2019. The discharge effort took over 7 weeks to complete, with the final battery successfully discharged on July 18, 2019.

#### 2.4.4 Rack 15 removal

Throughout the discharge process, Rack 15 remained in place so that the investigation team could conduct an on-site forensic evaluation of the entire rack before it was removed for further examination. The investigation team was able to access Rack 15 on July 22, 2019.

The disassembly of Rack 15 began with removal of non-battery components such as the door and cabling. With the exception of module 14 (the top-most module in the rack), it was discovered that all of the modules were melted or fused together. Module 14 was removed individually, and the remainder of the rack was removed as a unit. Steel plates were used to physically stabilize the rack during transport to Colwell Consulting in Scottsdale, AZ.

Prior to disassembly, APS and a portion of the investigation team (consisting of APS, Fluence, Colwell, SEL, Exponent, FIG, Surprise Fire, Sandia, and Rimkus) discussed and agreed on the protocol for the deconstruction of Rack 15. On August 19, 2019, beginning with Module 13 (the upper-most remaining battery module within Rack 15), the investigation team began separating and extracting each module from the rack assembly, one-by-one using a combination of pry bars, screwdrivers, grinders, and other small tools. At each step of the deconstruction process, all participating third parties were permitted to observe and document the process.

The disassembly of Rack 15 was completed on August 23, 2019. Each individual module and its components—as well as the Rack 15 inverter, BPU, and node controller—were packaged and shipped to SEL’s Michigan-based laboratory for additional analysis. SEL received all Rack 15 modules and components on October 4, 2019. The modules were packaged in a manner that would preserve their condition during transport.

#### 2.4.5 Environmental hazard assessment

APS hired both AECOM and Wood [37] to conduct an evaluation of any possible residual on-site and off-site environmental and human-health impacts. The environmental investigation involved collecting and evaluating on-site environmental media samples, such as surface soils, and performing air dispersion modeling to evaluate the potential for off-site environmental impacts. These investigations evaluated potential mechanisms for environmental exposures to contaminants of concern (COCs) and did not identify evidence of any on-site or off-site impacts or risks to human health or the environment.

##### 2.4.5.1 On-site investigation

On May 6 and 7, 2019, AECOM collected both surficial soil samples from the area around the BESS and wipe samples from within the BESS structure. Exterior soil sampling was collected in a grid pattern from the area surrounding the BESS structure. Along with the facility wipe sampling, nearby-area background samples were also collected as a way to “fingerprint” a range of COCs for further analysis. AECOM also reviewed relevant chemical safety data sheets and other vendor information to identify an appropriate COC scope. Laboratory analysis ultimately involved a range of organic and inorganic compounds suspected to be associated with the April 19, 2019 event.

Based upon subsequent laboratory analysis conducted in May and June 2019, AECOM found:

• Of the COCs evaluated for analysis, the sampling data exhibited either consistently low COC concentrations or non-detect values; samples generally revealed concentrations indistinguishable from background sampling data.

In comparison with applicable state and federal environmental, health, and safety standards, none of the samples exhibited COC concentrations in excess of levels that would require site remediation or other response actions.

• Additional sampling of other environmental media, such as groundwater or soil, was deemed unnecessary.

##### 2.4.5.2 Off-site dispersion analysis

Based on the data collected during the on-site investigation, between May and September 2019, Wood performed an analysis of whether the sequential thermal and pressure events associated with the April 19, 2019 incident created off-site risks to human health or the environment. This analysis involved air dispersion modeling to develop a conservative quantification of likely off-site deposition of COC particulates moving through the air in the plume. Wood’s analysis relied upon conservative emission rates. Satellite imagery and site-specific meteorological data were also utilized, along with an evaluation of the on-site soil and structure wipe sampling data.

As a result of this analysis, Wood found:

• COC particulate matter deposition through the air was minimal and primarily confined to on-site locations near the BESS structure.

• Modeled off-site concentrations of COCs were lower than applicable state and federal standards governing environmental, health, and safety protection from hazardous substance releases.

Based on the analysis from the two environmental studies, additional off-site environmental investigations were not recommended.

## 3 Understanding Thermal Runaway and Off-gassing in Li-Ion Batteries

In order to understand what happened at McMicken, it is first necessary to understand thermal runaway and off-gassing in Li-ion batteries. Following this explanation of what thermal runaway is and how it occurs, a detailed explanation of the forensic findings from the investigation will follow, which will describe how thermal runaway led to the flammable gas mixture that caused the explosion.

### 3.1 Historical understanding of thermal runaway in Li-ion batteries

The hazard of trapped gases from battery thermal runaway has been identified in research and public literature since at least the early 2000s when extensive safety testing on Li-ion batteries was performed by numerous entities, including ABSL, Sony, Sandia National Labs, and others. [5, 9, 10, 11] The study and awareness of Li-ion thermal runaway has evolved over time from academic and R&D, to eventually commercial and standardized testing:

2000-2006: Study of thermal runaway in primary (non-rechargeable) Lithium cells.

2006-2009: Study of thermal runaway in secondary (rechargeable) Li-ion cells, and examination of internal shorts as a result of defects. Thermal runaway “fires” in laptop computers put a focus on the safety of Li-ion Cobalt batteries, and the industry moved toward using batteries with less Cobalt.

2009-2012: Study of the gas composition of Li-ion cells undergoing thermal runaway using accelerating rate calorimetry (ARC) testing.

2009-Present: Public incidents of Li-ion battery failures reported as fires in electric vehicles (EVs) and vessels, coincident with greater adoption rates of hybrid and EVs.

2012-Present: Battery safety and fire testing for automotive, stationary, marine, and aviation at a commercial level.

2017-Present: Public reports of stationary ESS fires in South Korea led to a suspension in operation of 1,490 different ESS installations.

Over the years, several highly publicized events have brought about increased public awareness to the dangers of thermal runaway in Li-ion batteries. In the public and the media, they are commonly referenced as “fires”, but thermal runaway can be distinctly different from a fire as will be explained below. Some commonly reported incidents are as follows:

• (2006-2007) Dell laptop computer battery thermal runaway: After several laptops caught fire and Dell issued a recall, it was determined that the cause of the cell failures was sourced at an internal defect in the battery cells that led to thermal runaway. [12]

(2011) Chevrolet Volt battery thermal runaway after crash testing: A Chevrolet Volt was crash tested by the National Highway Traffic Safety Administration on May 12, 2011. The vehicle was stored in an on-site salvage yard and three weeks later, on June 6, 2011, it was reported that the car caught on fire. The cause was determined to be coolant that leaked over the terminals on the batteries, causing an external direct short, thereby leading the batteries into thermal runaway. There was speculation that an explosive event had damaged window glass in the vehicle, indicating explosive gases may have accumulated. Some of the battery cells themselves were also likely compromised in the initial crash. [8]

(2012) Thermal Runaway, Smoke, and Fire onboard the hybrid tugboat Campbell Foss: A hybrid tugboat, the Campbell Foss, equipped with Corvus batteries, experienced thermal runaway when the BMS failed to balance the batteries and inadvertently caused long-term overcharging of one or more battery cells. At least one cell went into thermal runaway, causing smoke and an emergency response to what was classified as a boat fire. This occurred in the Port of Long Beach. [24]

(2013) Tesla Model S thermal runaway after impact with road debris: A Tesla Model S, somehow impacted a curved piece of metal while driving on the freeway, which wedged under the car and punctured the battery pack, causing it to go into thermal runaway. First responders arrived to see smoke and off-gas and attempted to extinguish the event as they would a normal car fire, but with the additional tactic of piercing the protective metal casing of the battery pack, flooding it with water, and then attempting dry powder fire extinguishing. Tesla later produced guidance for first responders on how to handle a Tesla vehicle that had been involved in an accident, with and without fire. The thermal runaway did not propagate into the passenger compartment of the car. [13]

(2013) Boeing Dreamliner battery thermal runaway: In January 2013, a Boeing 787-8 experienced smoke and heat coming from its Li-ion battery-based auxiliary power unit (APU). It was later determined the cell failure was caused by an internal cell defect, which was exacerbated as thermal runaway cascaded through all the cells in the battery pack, releasing flammable electrolyte and gases. [14, 15]

(2017) Samsung Galaxy Note 7 thermal runaway: These Li-ion battery cell failures were caused by two unrelated, but coincident events. First, when Samsung manufactured the phones, they crimped the Li-ion batteries with the case during assembly and caused damage to them, which led to later shorts and thermal runaway. Second, when Samsung sourced batteries from another supplier, internal shorts, lack of insulation, and other cell defects caused them to fail and go into thermal runaway. The public nature of those coincident failures forced Samsung to perform a massive recall. [7]

(2017-2018) ESS Fires in South Korea: Beginning in August 2017, there were reports that ESS in South Korea were having “fires”. A report was finally released in 2019 which demonstrated the fires were associated with not only NMC batteries but also some LFP batteries, and they occurred in large and small systems regardless of form factor or grid application. The report first concluded that the errors were caused by inappropriate electrical protections (ground faults, electrical shorts), inappropriate operational environments, negligent installation, and inappropriate integration of the multiple protection layers of the BMS to the inverter and control system. [31] Then, in 2020, more information revealed that many of the fires were caused by battery cells with defects. [53]

The lessons the industry has learned from these incidents is that Li-ion batteries are inherently fragile, and any electrical, thermal, or mechanical abuse, along with internal defects, can potentially initiate cell failure and thermal runaway.

In February 2016, a report issued by the Fire Protection Research Foundation (FPRF—a research affiliate of the NFPA), established that cascading thermal runaway in ESS was a hazard. [18] However, the FPRF report focused on the explosive capability of battery cells, in particular, and did not address the potential

flammability of off-gassing from a Li-ion thermal runaway event. In fact, the FPRF report only examined ESS fires involving lead acid batteries, not Li-ion, and therefore lacked an in-depth analysis of the specific heat transfer and flammable gas release mechanisms of a Li-ion thermal runaway event.

Between 2013 and 2017, new test data demonstrated the toxic and flammable content of Li-ion battery offgas, which resulted in numerous new safety recommendations for marine and stationary ESS. [16, 19, 20, 21, 22, 23, 49, 50, 51] Additionally, in July 2016, the Norwegian Maritime Authority acknowledged the potential explosive risk caused by flammable gases from Li-ion battery thermal runaway, urging that hybrid and fully electric vessel designs address such risk. [24] Between 2012 and 2019, DNV GL has issued updates to the maritime battery rules, to include requiring a hazard assessment for flammable gases. Additionally, marine battery systems required testing to International Electrotechnical Commission (IEC) 62619, which has more stringent requirements for cell-to-cell cascading than the UL 1973 standard that was commonly used for stationary energy storage modules. Both IEC 62619 and UL 1973 are viewed as similar battery module testing standards in most industries. [26]

Work has continued to assess these hazards in 2019. [17] The application of this research is not limited to marine battery systems, rather it highlighted, in a general sense, the explosion risk that can occur when closed battery modules trap gases produced from thermal runaway of one or more cells.6

### 3.2 How thermal runaway is different from a fire

Before addressing more details on how the failure occurred and led to an explosion, it is important to describe thermal runaway in Li-ion batteries and the distinction between thermal runaway and a conventional fire.

The fire triangle is used for training purposes to help firefighters, first responders, and safety officials generally understand the essential ingredients of any fire. An example of the fire triangle is shown in the left of Figure 11. [4] The combination of fuel, heat, and oxygen are needed to sustain a conventional fire. If any one of them is removed, the triangle is disassembled, and the fire may be extinguished. Consider the example of a typical campfire. Application of heat through fire or friction will ignite the fuel (the wood). When the wood is stacked and arranged such that air can flow under and through the firewood stack, it provides oxygen to help the wood fuel burn. The campfire is extinguished by removing any one of the corners of the triangle, i.e., suffocating the fire with a blanket to remove oxygen, dousing it with a bucket of water to cool it, or dispersing the embers to remove the wood fuel.

However, thermal runaway in a Li-ion battery differs from the conventional fire triangle, as shown in the right triangle in Figure 11. As a Li-ion battery is consumed in thermal runaway, it is fueled by an internal chemical reaction that releases heat and can continue without oxygen or a visible flame. Accordingly, management of thermal runaway differs from that of a conventional fire, specifically because starving a thermal runaway event of oxygen may have little effect, at least until the exothermic reaction is complete. Instead, heat must be removed (by cooling with water [16]), or the fuel must dispersed in order to stop a thermal runaway event.

While thermal runaway and fire are two different processes in a Li-ion battery failure, one can lead to the other. For example, an external fire can send a battery into thermal runaway, or thermal runaway can create a secondary fire as it begins to the burn plastic and remaining battery materials.

DNV GL has performed hundreds of Li-ion battery cell thermal runaway tests (collected data displayed in Figure 12). The basic thermal runaway process of a single Li-ion battery cell (in air), and the secondary fire behavior, can be summarized into three phases.

First, voltage or temperature instability occurs. During DNV GL’s test, the internal temperature of the cell rose as direct battery cell heating was applied. The cell began emitting gases, sourced mostly from the thermal decomposition of the electrolyte solvents in the battery, before going into thermal runaway.

Second, the voltage drops to zero as the internal cell materials fail, and the anode and cathode experience a direct internal short. This is the start of thermal runaway, as all of the electrical energy stored in the battery flows through the short. Temperatures can briefly spike as high as 300o – 600oC.7 The duration of the temperature peak can be seconds to several minutes depending on the size of the battery. In open air, this thermal runaway event may be accompanied by flame, although, in a tightly packed assembly of cells, flame may not occur. Some firefighters may call this a “deep seated fire” but it is important to remember that substantial heat can be generated without any flame whatsoever.

Third, as the internal materials of the cell are consumed, the thermal runaway event can transition (if oxygen is present) to the consumption of the remaining battery materials—i.e., the electrolyte and polymers and plastics surrounding the battery cells. This would be a secondary fire as a consequence of the heat generated from the exothermic reactions in thermal runaway. The gaseous emissions of this “burn” are consistent with a plastic fire—i.e., H2, CO, HCl, HCN, and HF are emitted along with other hydrocarbons. This plume is flammable and toxic, and if sufficiently concentrated, a flammable atmosphere may be created. 8, 9 It is important to note that the combination of these flammable gases can create a combined flammability

and autoignition limit that is actually lower than the limits of the individual gases. The autoignition temperature of a substance is the lowest temperature at which it spontaneously ignites in normal atmosphere without an external source of ignition, such as a flame or spark.10

In a module, rack, and system, battery cells can fail at different rates under different modes of off-gassing— with or without flaming. Because thermal runaway can occur without flame, gases may not be consumed by an ignition source immediately after they are emitted from a consumed cell. The presence of a fire suppression agent will also inhibit ignition of these off-gases. In fact, the potential of a non-flame thermal runaway event creates a scenario where high heat and combustible gases can coexist without igniting. This is an important aspect of thermal runaway that is relevant to the cause of the explosion at the McMicken facility.

## 4 Series of Events and Contributing Factors Leading to the Incident

The following sections detail the contributing factors and series of events shown in Figure 13 that led to this incident.

### 4.1 Contributing Factor #1: Internal failure in a battery cell initiated thermal runaway

This investigation has established that an internal cell failure resulted in thermal runaway of battery cell 7-2 in Module 2 of Rack 15. The investigation also found evidence of Lithium metal deposition and abnormal dendritic growth in randomly selected battery cell samples without thermal damage taken from the BESS, and non-damaged battery cells taken from a sister site in another part of the APS territory (Festival Ranch). Because the evidence of Lithium metal deposition and abnormal dendritic growth was sufficiently present in the random samples that were analyzed, it was determined to a reasonable degree of scientific certainty to be the anomaly that caused the initial cell failure and ensuing thermal runaway in cell 7-2 of Module 2 in the BESS. Identification of Lithium metal deposition and abnormal dendritic growth, specifically in cell 7-2 of Module 2 of Rack 15, was not directly possible because that battery cell was severely damaged by the thermal runaway.

APS data recorded the minimum and maximum cell voltage of the rack from the moment the system was commissioned. Just prior to the triggering of smoke and fire alarms, the data monitoring systems detected that the voltage in Rack 15 dropped from 799.9 to 796.1 V at 16:54:38 (see Figure 14; x-axis is in Eastern Time). Eight (8) seconds before the rack voltage drop, beginning at 16:54:30, the minimum cell voltage in module 2, battery 7 dropped from 4.06 to 3.82 V. Shortly thereafter, the current flipped from -27.9 A charging to 4.9 A discharging. The measured voltage drops, first in cell 7-2 of Module 2, and subsequently at the rack level, are the first anomalies recorded during the incident. Shortly thereafter, air temperature in the container began to increase (Figure 15) which was indicative of a heating event. [47] The VESDA smoke alarm was activated at 16:55:20 PST (19:55:20 EST), about one minute later.11

The evidence provided by the APS investigation teams provides strong data supporting that rack 15, module 2, battery cell pair 7 was the origin. Specifically, battery 7-2, located at the very rear of the module suffered an internal failure, which was indicated by the ejection of internal battery cell material and further confirmed by visual observations of the heat damage on neighboring cells and the module exterior. The location of battery 7-2 is shown in Figure 16.

With the exception of the telecommunications rack, the official nomenclature of the racks follow a BN-X## format. The rack where the failure occurred was BN-X115. Modules are numbered from the bottom upward. There are 36 rack positions in the system. (BN-X115 position in the BESS is shown in Figure 17.)

For comparison, an example of normal operation is provided from the prior day. On April 18, 2019 (the day before the incident) at 14:00, while charging, the voltage climbed up until the normal cell maximum voltage of 4.18 V was reached. A similar charging cycle was occurring on the following day, the day of the event. The charge current observed the day before was 128 A (nominal 1 C rate).12 During the charge cycle when the failure occurred, the charge current was 101 A. The cells are 64 Ah cells, and since there are two cells in parallel, the 1 C-rate at the cell (64A) would be 128 A at the module. Therefore, the charging current of 101 A represents about 50.5/64 A = 0.798 C at the cells. Generally, for NMC cells, C-rates less than one are considered mild unless directed otherwise by the manufacturer. Thus, the implicated cells (7-1 and 7-2), did not have an abnormally high charge current or C-rate at the time of the incident. This is germane because the operational conditions were considered “normal” at the time of the incident.

During its investigation of Rack 15, SEL observed significant yielding and/or melting of the aluminum on module frames and compromised structural integrity. In some cases, the modules were melted together. The melting point of aluminum is \~660oC, and as shown in Figure 12, it would be possible for a single battery cell to exothermically generate temperatures > 400oC. Since it was highly likely that more than one cell was undergoing thermal runaway at any given time, it was certainly possible that multiple exothermic reactions during cascading thermal runaway were additive to generate temperatures as high as 660oC. The strength of aluminum (and most common metals) decreases as the temperature increases. This is important because the sagging of the modules may have aided in the cascading from one module to the next throughout Rack 15.

As Module 2 was disassembled, care was taken to remove the batteries to preserve their post-failure condition. Many battery modules were scanned with x-ray tomography.13

The X-ray images showed missing materials in the cells (Figure 18 and Figure 19). Cells were removed in sections or groups, two or three at a time. Battery cell 7-2 exhibited a large void. Cells 6-1, 6-2, and 7-1 all bulged into cell 7-2. Lower resolution scans were taken of modules prior to disassembly to verify that any anomalies were not caused by the disassembly process. The last four cells of each scanned module, including the cell pair of 7-1 and 7-2, were removed as a group. They were removed by using metal shears to cut the thin battery tabs from the bus bar without disturbing the battery cells.

Throughout its investigation and analysis, SEL analyzed and inspected a random subset of non-Rack 15 modules, as well as modules from the Festival Ranch BESS. Analysis of battery cells from non-rack 15 battery modules from the BESS revealed internal Lithium deposits and abnormal dendritic growth populating the electrode layers of a significant number of cells. Their presence on the surface of the anode and their propensity to absorb moisture and react exothermically when exposed to open air makes them consistent with metallic Lithium plating. These deposits were also observed in similar frequency and morphology on many of the Festival Ranch battery cells analyzed. For example, Figure 20 demonstrates the internal Lithium deposits and dendritic growth found in battery cell 7-2 of Module 6 of Rack 24 of the McMicken BESS.

The investigative teams disassembled a non-fire damaged battery cell from the McMicken BESS in open air (Rack 24, Module 6, Cell 7-2). Upon further investigation, SEL discovered that the surface of the carbon anode had white dendrites growing on its surface. The investigation team observed immediate discoloration of existing Lithium deposit locations as the dendritic growth region reacted with the air, as would happen if Lithium metal were exposed to air (open air has natural moisture content). The cathode/anode/separator layer in the location of these deposits started to smoke when opened, and eventually ignited, again consistent with Lithium’s reaction in open air. Transfer of carbon onto the separator and delamination of the carbon layer at these sites indicated a decomposition of the battery electrode materials at these locally defected sites.

SEL performed further analysis on the observed deposits and dendritic growth on a Festival Ranch battery. The analysis included visual and chemical characterization of these deposits in order to properly verify the presence of Lithium:

• The morphology of the dendritic growth was examined with a scanning electron microscope on the anode and adjacent separator.

Energy Dispersive X-ray Spectroscopy (EDS) detected the presence of Fluorine and Phosphorus, which indicates the localized decomposition of the electrolyte. EDS cannot detect Li physically (Li is too light of an atom to resolve a signature by EDS), but Aluminum, Phosphorus, and Fluorine elements were observed. The battery electrolyte is a mixture of ethylene carbonate compounds and Lithium hexafluorophosphate (LiPF6), so the presence of Fluorine and Phosphorus indicates the decomposition of the electrolyte on the surfaces and the likely presence of Li.

Bulk Inductively Coupled Plasma – Optical Emission Spectroscopy detected elevated concentrations of Lithium metal within the deposits thereby proving that these deposits were indeed composed of Lithium.

The verification of Lithium deposits in the battery cells is important because the abnormal accumulation of Lithium deposits and dendritic growths can cause battery swelling, electrolyte degradation, exothermic reactions, separator damage and shorting internal to the cell (as shown in Figure 22). This damage to the cell may in turn cause thermal runaway.

SEL determined that the presence of both Lithium deposits and dendritic growths in both McMicken and Festival Ranch battery cells, which had similar operational parameters and use histories, provided evidence that these deposits were inherent in the battery cells themselves, and not the result of any misuse from operation in the BESS.

Because module 2, cell 7 was significantly damaged in the thermal runaway event, it was not possible to directly identify evidence of Lithium plating and dendritic growth in that cell. However, based on the significant evidence of Lithium plating and abnormal dendritic growth found in a significant number of randomly selected representative samples from the BESS and Festival Ranch, and the scientific consensus that Lithium plating and abnormal dendritic growth can cause an internal short leading to thermal runaway, SEL and Dr. Hill believe to a reasonable degree of scientific certainty that the observed cell defects (Lithium plating and abnormal dendritic growth) caused the internal cell failure in module 2, cell 7.

### 4.2 Contributing Factor #2: The fire suppression system was incapable of stopping thermal runaway

CP Fire, along with representatives of other interested parties, visited the BESS in November 2019 and performed an inspection of the VESDA fire detection system and Novec 1230 fire suppression systems. CP Fire also reviewed design documentation of the fire protection systems. [52]

During this review of design documentation, CP Fire noted several shortcomings in the design process of the BESS’s fire suppression system:

Documentation provided by the system designer indicates that the two room integrity tests (air changeover tests) were performed on the BESS in September and December 2016 as part of the Novec 1230 system commissioning. The tests resulted in a predicted agent hold time of 5.4 and 7.6 minutes, respectively. These hold times were less than the absolute minimum hold time of 10 minutes required by the NFPA 2001 standard.

At some point just prior to commissioning, the design basis for the fire suppression system was changed from a 4.5% to a 10% concentration of Novec 1230.14 CP Fire was provided with no documentation to demonstrate the technical justification for this change. Although counterintuitive, increasing the concentration of Novec in the BESS may have shortened its effective hold time.15

#### 4.2.1 The fire suppression system installed in the BESS operated as designed during the incident

CP Fire’s investigation concluded that the fire protection systems installed in the BESS—including the VESDA smoke detection system, the Notifier RP 2002 Releasing Panel, and the Kidde Novec 1230 fire suppression system—operated as designed during the incident.

System monitoring logs indicate that the VESDA fire detection system exceeded all four of its alarm thresholds within a minute of the first battery cell voltage drop. Activation of the VESDA smoke detection system initiated a 30-second countdown before discharge of the Novec 1230 fire suppression system (per design), and monitoring data indicates that the Novec system discharged following a 30-second countdown as designed. During the on-site fire protection system investigation, the Kidde tank that contained the Novec suppressant was inspected, and it appeared to be empty with the discharge pin activated.

#### 4.2.2 Clean agent systems do not stop cascading thermal runaway

The ultimate failure in the design of the fire suppression system was the decision to rely solely on a clean agent fire suppression system to combat thermal runaway. The Novec 1230 clean agent was inadequate and inappropriate to stop or prevent cascading thermal runaway of multiple battery cells or modules.16 In fact, a representative of 3M, the developer of the Novec 1230 fire suppression agent, publicly acknowledged this limitation in August 2017. In a comment dated August 10, 2017, accompanying his vote on a draft version of the NFPA 855 Standard for the Installation of Stationary Energy Storage Systems, the 3M representative stated, “[c]lean agents are demonstrably ineffective on preventing and stopping thermal runaway, as are foam and dry chemical.” [44]

In the McMicken BESS, the Novec 1230 properly discharged shortly after the first battery cell failed, but over time, the agent-air mixture began to dissipate, negating its fire suppression effectiveness. At the same time, as discussed above, thermal runaway continued to propagate through Rack 15 emitting large quantities of flammable gases into the container.17 CP Fire notes that the Novec 1230 suppressant may have played a role in preventing the ignition of flammable gases as they were released from the thermal runaway in Rack 15, thus allowing these gases to accumulate in the BESS and create an explosion hazard. Indeed, based on the data provided in NFPA 2001, a Novec 1230 concentration of 8-9% would have been high enough to

prevent an explosion, and the design concentration of 10% was more than 3 times the minimum extinguishing concentration required for the fire suppression assumptions for Class A materials.18

According to CP Fire, Section 5.4.2.1 of NFPA 2001 recognizes the risk of flammable environments and uses the example of a gas leak or gas jet. The standard cautions that “Under certain conditions, it can be dangerous to extinguish a burning gas jet. As a first measure, the gas supply shall be shut off.” The reason for this is because it may be safer to burn the flammable gas as it is released to prevent the flammable gases from accumulating to form a subsequent explosion hazard. This caution is brought up in this context because a similar, but unspecified, caution applies to cascading thermal runaway, with the added threat that thermal runaway cannot simply be shut off in the same way as a gas supply can. Emergency responders are trained on the hazard of burning gas jets; they need to understand that the hazard of cascading thermal runaway is that it can create flammable gases without a flame to consume them.

CP Fire notes that over time the Novec 1230 agent dissipated, estimating that most of the agent may have leaked from the facility within the first 30 minutes after discharge if estimated by the room integrity test results alone.19

Despite the limitations described above, clean agent fire suppression systems may play an important role in future ESS facilities, because they are effective at extinguishing incipient fires and could potentially prevent the initiation of thermal runaway by external (non-thermal runaway) fire hazards. However, these clean agent systems should be used in conjunction with other fire protection measures and firefighting practices, including ventilation and cooling systems designed to handle thermal runaway and potential explosion hazards. In addition, CP Fire recommends that flammable gas concentration meters be installed in all future energy storage facilities, which would allow operators to continuously monitor any flammable mixture or vapor accumulation, and ensure that flammable gas concentrations remain well below the lower flammable limit.

### 4.3 Contributing Factor #3: Lack of thermal barriers between cells led to cascading thermal runaway

The investigation determined that once thermal runaway was initiated in module 2, battery 7, the close packing of the battery cells within each module allowed that cell to trigger thermal runaway in its neighbor, which spread to the next cell, and so on. This process consumed all the cells in the module and concurrently produced significant heat, which transferred to the neighboring module, and then the next, and so on through Rack 15. Moreover, several modules were melted together and sagging as a result of the intense heat. Eventually, the entire rack was consumed, producing a significant volume of flammable gases. The cell-to-cell packing and module-to-module placement provided no protection to prevent this from occurring. There was no thermal barrier in place between the LG cells to prevent them from directly transferring heat during thermal runaway, contributing to the cascading cell-to-cell thermal runaway that inevitably occurred.

Modules 1-13 were partially melted and fused from the heat of thermal runaway.

However, as demonstrated by Figure 23, the thermal runaway was isolated to Rack 15. In fact, the modules in the neighboring racks all maintained the voltage in their final charge state at the point of system shutdown. This evidence indicates that thermal runaway in Rack 15 was transferred vertically between all battery modules in Rack 15 via heat transfer rather than by flame. For example, the neighboring racks did not show burn damage that would be expected from a conventional fire. As shown in Figure 24, the isolated, module-to-module damage to Rack 15 is characteristic of cascading thermal runaway between modules via heat transfer, rather than a burning fire with open flame and soot, which would have likely caused a damage pattern similar to a vertical “V” shape. In the BESS, it appears as though the materials and air gaps between racks prevented horizontal heat transfer and initiation of thermal runaway in adjacent racks, resulting in only vertical propagation through rack 15.20

Conventional fire, Vshape direction due to direct flame and heat.

A key factor exacerbating this incident was the absence of any means by which the battery modules could prevent the continued propagation and cascading from a single cell. The Novec 1230 discharged shortly after the first cell failed, yet thermal runaway continued to propagate. (Figure 25) The Novec 1230 system could only discharge once, by design, yet the flammable gases continued to emerge as each new cell cascaded in thermal runaway. The Novec 1230 discharge may have played a role in preventing the ignition of flammable gases when they were released, but Novec 1230 and other clean agents are not intended to stop the progression of a cascading thermal runaway. Consequently, the spread of thermal runaway led to the generation of a large quantity of gases, which were not burned as they were emitted. Figure 25 illustrates the chain of events based on the evidence available.

### 4.4 Contributing Factor #4: Flammable off-gases concentrated without a means to ventilate

As designed, the system shut down the air conditioning when the fire suppression system discharged, which allowed the emergent gases to concentrate. It is common practice to shut down the HVAC systems once the clean agent is deployed.

Colwell provided an analysis of the gases that contributed to the explosion in the BESS. The analysis used the following assumptions to model the April 19, 2019 event. [48]

Colwell examined a model of the container by computing the interior volume and subtracting the volume occupied by equipment, arriving at an interior air volume of 199 m3.

Colwell and other experts determined that the Novec 1230 system had 713 lbs. of agent in the bottle (density 13.6 kg/m3) and computed that the discharged Novec 1230 would occupy about 25 m3, or 12.6% of the available container volume.

In the model, Colwell assumed that the electrolyte in the batteries was the main source of gases. This is a fair assumption that likely errs toward conservatism. Colwell used literature data on smaller 18650 (cylindrical), NMC-type Li-ion cells to determine the likely gas composition from the LG Chem cells in the McMicken BESS, supported by a test method similar to ARC testing, an acceptable way to determine gases in a Li-ion battery. Their analysis determined a flammable gas mixture which could be scaled on a per-mass basis for the LG Chem NMC pouch cells.21 The data indicated that the flammable gases present were H2, C2H4, CH4, CO, and CO2. This data was sourced from the literature [49], and is consistent with DNV GL’s battery thermal runaway commercial testing experience and prior research. In Dr. Hill’s experience and DNV GL’s battery databases, the composition of gases from a Li-ion battery thermal runaway event is generally consistent across chemistries, form factors, and manufacturers, so this is a valid assumption made by Colwell. The reasoning for this is that most Li-ion batteries today rely on common electrolyte solvents (typically ethylene carbonate or similar solvent) and have similar polymer materials in the remaining balance of mass of the cell.

The Colwell model assumed that the yield of the gases is proportional to the electrolyte mass, which is proportional to the battery size. This is consistent with DNV GL PHASTTM plume model assumptions for cascading thermal runaway in energy storage systems. The model assumed that there was 4.4 g of electrolyte in the tested battery, which translates to 217 g in the 64 Ah LG Chem pouch cell. The LG cells weighed approximately 1.1 kg each. [37] As shown in Table 2, polymeric materials add an additional 0.75- 2x mass of hydrocarbon content when considered in relation to the mass of electrolytes in a Li-ion battery. [29, 30]

A key parameter in the McMicken event is the lower flammability limit (LFL) of the gas mixture in the container that was produced from Li-ion batteries undergoing thermal runaway. The lower and upper flammability limits of gases (LFL and UFL, respectively) are the limiting fuel concentrations in air that can support flame propagation and lead to an explosion. The progressive addition of an inert gas or other aerosol or clean agent suppression agent to a fuel–air mixture causes the narrowing of the flammability range to the point where the two limits coincide. The limiting oxygen concentration (LOC) is the minimum O2 concentration in a mixture of fuel, air, and an inert gas that will propagate flame. [3]

The model then computed the total volume of gases released from all the batteries in the thermally damaged rack. According to Colwell, the resulting flammable atmosphere had the following properties:

• Air: 67.5%

• Novec: 8.5% (after venting and dissipation, over time, through container seals)

• Hydrogen: 7.4%

• Ethylene: 2.0%

• Methane: 1.6%

• Carbon Monoxide: 3.1%

• Carbon Dioxide: 9.9%

Thus, the resulting atmosphere likely had the following properties:

• 14.1% fuel concentration after thermal runaway

• LFL of 4.5%, UFL 46.4% including reduction in limit by H2

In this static case, the fuel-to-air ratio is within the LFL and UFL bounds. This calculation assumes that the gases were instantaneously released at the beginning of the event. Colwell then advanced the calculation to account for air changeover and estimated the air changeover rate to be 1.7 to 1.8 air changes per hour. At this rate, the total mixture concentration would have depleted to 0.04%-0.08%, below the lower flammability limit. Obviously, an explosion occurred, and the gases were not instantaneously released, so this factor had to be added into the analysis.

Because the gases were emitted over time, Colwell examined the role of convection to drive stratification between the Novec 1230-air mixture and the flammable gases, i.e., hotter gases would rise. Colwell calculated that a 2.3-4.8 ft. upper layer of flammable gases was a physical possibility in these conditions.

Colwell’s analysis of the BESS suggests that the hotter, flammable gases rose to the upper level of the container while the Novec 1230-air mixture would descend due to its higher mixture density and cooler temperature. Over time, the hotter rising flammable gases displaced the Novec 1230-air mixture as it leaked from the doors and other seals in the container. The amount of time from the initial discharge of Novec 1230 to the opening of the door provided ample opportunity for the flammable gases to concentrate, as illustrated in Figure 26. The Novec 1230 is red in the figure, and the flammable gases from thermal runaway are grey.

#### 4.4.1 Cause of explosion

Colwell hypothesized that when the container door was opened, the flammable mixture came into contact with a thermal source, which was likely to be the still hot batteries in Rack 15, as illustrated in Figure 27. It is Dr. Hill’s experience that residual heat within a battery module that has experienced thermal runaway damage can remain for minutes and hours after thermal runaway appears to be complete (see Figure 34 in following sections), so this ignition mode is plausible.

### 4.5 Contributing Factor #5: Emergency response plan did not have an extinguishing, ventilation, and entry procedure

The emergency response plan (ERP) provided by AES to APS did not have instructions on how to respond to a potential explosion or how to enter the system after the fire suppression system had been discharged. Most of the detail in the ERP was associated with electrical shut down procedures and roles and responsibilities between APS and Fluence (as the O&M contractor), as well as when/if to notify the fire department. A smoke alarm and fire suppression trigger procedure was in place, but it did not address when or how to initiate entry into the system. [32] The ERP did not provide information about preparations to enter, including monitoring, measurements, ventilation, and extinguishing.

At the time of the development of this plan, none of the BESS suppliers (including AES and LG Chem) had conveyed that a large flammable gas hazard or cell-to-cell and module-to-module cascading thermal runaway was possible. As will be explained in “Shortcomings that should be addressed in NFPA 855” in later sections (page 48), this kind of communication breakdown must be remedied to avoid future incidents.

It is helpful to examine and demonstrate the need for open communication between all parties involved in constructing and operating an ESS using a technique called the “Johari Window,” which is used in risk analysis and hazard identification exercises to heuristically identify the known-knowns, known-unknowns, and unknown-unknowns of a technology, process, or operation. [38]

As shown in Figure 28, a Johari window can be used to analyze the state of knowledge (both known and unknown) by APS (via its EPC contractor and their subcontractors), at the time the ERP was drafted. In this simplified Johari window, “Internal” refers to APS’ collective knowledge as an organization, and “External” is the realm of knowledge available to APS from their supplier and subcontractors. Although AES drafted the ERP, had APS known of all the hazards and risks associated with the BESS, it could have better analyzed the sufficiency and adequacy of the ERP prior to commissioning.

“Open” hazards and risks were known by all parties at the time the ERP was drafted. All parties (including APS) were likely aware of common electrical hazards and risks, as well as commonly cited public battery fire

hazards from other industries. Indeed, electrical hazards were addressed in the emergency response plan. These hazards are depicted in the upper left quadrant of Figure 28, and represent the known-knowns.

“Hidden” hazards and risks were known by APS at the time the ERP was drafted, but disclosure of information was limited or incomplete. APS was likely aware that LG Chem had test data with regard to battery thermal runaway, such as UL 1973 or UL 1642 and UN 38.3; however, it is highly likely that information presented from these tests was limited to a pass/fail notification without detail. In other words, it is highly doubtful that AES was provided any information about the condition of the cells and modules after the tests. Because these details were not shared with AES, they could not be shared with APS. Similarly, APS likely knew there was a process that AES was following to design the fire suppression system, but the details of that analysis were evidently not shared with APS. These hazards are depicted in the lower left quadrant of Figure 28, and represent the known-unknowns.

“Blind” hazards and risks may or may not have been known to external parties but were unknown to APS at the time the ERP was drafted. It cannot be stated with any degree of certainty what any party knew or did not know regarding the threat of cascading thermal runaway when the BESS was commissioned. However, it can be inferred through interviews and design documentation that APS was unaware of the threat of cascading thermal runaway—including the risk of flammable off-gases, the need for ventilation, and whether clean agents and aerosols could stop thermal runaway. These “blind” hazards are depicted in the upper right hand quadrant of Figure 28, and represent the known-unknowns in the external world, yet blind unknowns to APS. These are areas where APS likely did not know what they didn’t know.

“Unknown” hazards and risks were unknown to all parties at the time the ERP was drafted. It should be noted, that Dr. Hill is unable to conclusively say what each party knew during the design and commissioning of the McMicken BESS. However, based on a review of the design documentation and Dr. Hill’s experience in working with similar parties in the energy storage industry over the last twelve years, it is unlikely that either APS or AES were aware of the fact that a single cell thermal runaway was a threat to the entire rack, and the volume of gases from that thermal runaway event could displace Novec 1230 from container. However, it is possible AES (and perhaps APS) could have recognized the cascading thermal runaway threat if LG Chem had shared more information concerning their battery cell and module testing. The collection of gases and displacement of Novec 1230, as demonstrated by the analyses from Colwell and CP Fire in this document, would have been much more difficult to predict. These hazards are depicted in the lower right quadrant in Figure 28, and represent the unknown-unknowns to APS and AES.

The ERP for the McMicken BESS did not have an extinguishing, ventilation, and entry procedure in the event of cascading thermal runway that would produce significant flammable gases. APS relied on AES to draft the ERP, and because certain risks and hazards specific to cascading thermal runaway were never brought to APS’ attention, APS did not know that the ERP for the McMicken BESS needed to specifically address such an event. However, it seems that AES was equally unaware of the significant gas hazard that could be produced by cell-to-cell and then module-to-module cascading thermal runaway. As such, AES could not draft an appropriate ERP that adequately addressed the risk of cascading thermal runaway, nor could they properly advise APS of these risks.

While not certain, it is likely that AES (and/or APS) would have recognized the risks associated with cascading thermal runaway if LG Chem shared the details of the internal condition of their modules after UL 1973 testing (which is not typically included in test reports), or shared the results of fire testing programs in which they had participated. Without such data from the battery manufacturer, AES (and APS) would have to actively search for the specific hazards of thermal runaway in public reports and databases. However, it is unlikely AES or APS would perform such an exhaustive search without some information demonstrating that cascading thermal runaway – from cell-to-cell, then module-to-module - was a hazard or risk that had a reasonable probability of occurring.

Ultimately, there was a lack of information concerning the potential explosive gas hazard created from unmitigated cascading thermal runaway through an entire battery rack throughout the commissioning process. This is demonstrated by the deficiencies in the ERP, which lacked procedures for extinguishing, ventilation, and entry of the BESS in the event of cascading thermal runway. As such, the ERP provided no guidance to responding firefighters concerning cascading thermal runaway or how to safely ventilate, extinguish, cool, and enter the BESS container.

The Arizona Department of Occupational Safety and Health (ADOSH) conducted an investigation following the incident. On October 22, 2019, ADOSH issued a letter informing APS that “the elements for an OSHA violation were not present” and no citations would be recommended. However, ADOSH’s letter requested that APS bring its emergency response and operations plan in accordance with the newly released NFPA 855 Standard for the Installation of Stationary Energy Storage Systems, particularly section 4.1.3 “Emergency Planning and Training”. ADOSH’s interpretation of this section of NFPA 855 states that such training is the responsibility of the “owner” of the system or their authorized representatives. Section 4.1.3 of NFPA 855 includes a requirement for response plans and procedures to be implemented and drilled, including response procedures to address the release of liquids or vapors. [41] It should be noted that NFPA 855, while in draft during the commissioning and early operation of the McMicken BESS, was not officially released as a standard until summer of 2019 after the McMicken explosion. Therefore, the industry collectively and officially identified ventilation of flammable off-gas22 as a requirement and placed it in the “Open” category of the industry Johari Window of Figure 28 in 2019. As will be explained in later sections, the body of industry standards is still challenged to place cascading thermal runaway in the collective “Open” category.

As reflected above, the emergency response to the BESS incident was hindered by several factors, including an insufficient understanding by all parties as to the hazards posed by unmitigated cascading therma runaway inside an energy storage system.

## 5 Knowns and Unknowns Today – Review of Industry Knowledge and Standards

Certainly, some public events, industry research, and knowledge of thermal runaway and flammable off-gas information was available prior to the commissioning of McMicken. However, this information and data was scattered across a broad range of sectors, including the automotive and maritime industries, as well as scientific communities, national laboratories, and industry-specific committees.

Drafts of UL 9540 were first released in 2015. The public NYSERDA/Con Ed fire testing report on commercial Li-ion ESS batteries was published the month prior to the commissioning date of the McMicken BESS. [16] At that time, UL 9540 had been circulated as a draft for about 1.5 years, and those early drafts had sparse guidance on cascading thermal runaway. [33]

There had been industry programs on general ESS safety for about 1-1.5 years leading up to the commissioning of the McMicken BESS. Concerns began to rise concerning ESS fire safety as “fires” were reported in South Korea on over 23 installations, yet details as to why and how those “fires” occurred were not available at the time. [53] While the NFPA 855 drafts were published for public review within months of the commissioning of the McMicken BESS, NFPA 855 did not officially publish as a standard until late 2019, after the McMicken BESS explosion. Any change to the safety systems of the BESS would have had to occur proactively during the finalization of the BESS design in 2016 as new data was just emerging, or after the system was commissioned and during its operation from March 2017, prior to the explosion in April 2019.

### 5.1 Relevant industry codes and standards

The BESS was designed, built, and commissioned around the time that standards development and industry programs were just beginning to understand the risk of thermal runaway and flammable gases. For context, Table 3 provides a chronology of all relevant industry publications, codes, standards, and events in relation to the key dates for the McMicken BESS.

The present body of codes and standards has been focused on the mitigation of heat, toxicity, and flammability risks using common tools to fight conventional fires. In essence, the community of safety concerning Li-ion batteries has been focused on “fire” and not thermal runaway.

#### 5.1.1 UL 1973

The UL 1973 standard has been used in the United States market to assess the safety of stationary energy storage battery modules since at least 2013. Awareness of the standard has increased since 2016 and has been commonly prescribed and required by system integrators and authorities having jurisdiction (AHJ). The pre-2016 UL 1973 test method, in its “internal fire test”, did not prescribe any particular event initiation method but suggests heating, puncture, or overcharging of internal cells (described only to be at a central

location in the module) to induce thermal runaway. The pass/fail criteria was described as “No fire propagates to the outside of the DUT enclosure or explosion”. 23

#### 5.1.2 UL 9540

UL 9540 is a standard that outlines requirements for ESS to be deployed in the field, identifying what standards may be used for compliance. UL 9540A is a supplementary document to UL 9540. UL 9540A is a test method. The development and first editions of the standards for UL 9540 and UL 9540A were as follows:

UL 9540 (standard for certification)

• Proposals for revisions to draft November 13, 2015 and July 1, 2016

• First edition November 21, 2016

UL 9540A (test method):

• 1st edition: November 2017

• 2nd edition: January 2018

• 3rd edition: June 15, 2018

As the timeline demonstrates, the UL 9540 drafts preceded the commissioning of the McMicken and Festival Ranch BESS. Drafting of UL 9540A was concurrent with the commissioning and first year operations of the BESS. When drafting a standard, UL managed the writing of the standard but formed an industry Standards Technical Panel (STP) for industry engagement and consensus. To participate in the panel, a company or individual requested participation with UL, who then approved or rejected the request. The membership of STP 9540 did not include LG or AES.24 [33]

The UL 9540 standard was issued before the UL 9540A test method was developed. The standard references NFPA 497, which is a recommended practice to classify flammable liquids and vapor. UL 9540 requires a failure mode effects analysis (FMEA) and references IEC 60812, IEC 61025, and MIL-STD-1629A. Section 22 of UL 9540 discusses flammable concentrations of gases and ventilation requirements, but the language of this section is notably lead acid and hydrogen focused.

In section 23.3, it is specified that “scientific data” shall be provided to justify fire suppression selection. This language was, in part, the genesis for the later publication of the UL 9540A test method, as the specific scientific data required at that time was not described anywhere in a codified manner. Otherwise, in the UL 9540 standard, the only reference to cascading thermal runaway resides in Appendix B where details on UL 1973 are provided informatively. Noted in Appendix B2.5: “thermal runaway protection should be provided. Such protection should be verified through testing that it provides sufficient protection of the battery pack.”

It is noted in Appendix B, section B2.14, that a thermal runaway test should determine the worst case failure that can occur. Other than these mentions of thermal runaway cascading, none of UL 9540, UL 9540A, or UL 1973 acknowledge the potential to create an explosive atmosphere that is dependent on the rate of cell failure, or that thermal runaway can continue to propagate from cell-to-cell and module-tomodule without flame, even in an oxygen-deprived environment.

#### 5.1.3 NFPA 855

Drafted versions of sections of NFPA 855 were available in 2016-2017 and the first draft meeting notice was issued on August 10, 2017. NFPA 855 was released on August 25, 2019 as the 2020 edition. Both LG Chem and Fluence had a representative on the NFPA 855 technical committee. [42, 43]

Opportunities to publicly comment on near complete drafts of NFPA 855 occurred in October 2017 and May 2018. Dr. Hill of DNV GL was invited to present fire testing results to an NFPA 855 technical committee meeting in May 2017, and presented DNV GL’s findings on water cooling requirements, ventilation requirements, and the need to address cascading thermal runaway in Li-ion battery cells and modules. Dr. Hill presented test results on how much water and ventilation should be required. Dr. Hill again presented results to the NFPA 855 technical committee members at a NFPA Energy Storage Safety Summit in Denver, Colorado in February 2018. In that meeting, Dr. Hill outlined that the EPC contractor should be fully engaged in training and emergency response.

The final draft of the NFPA 855 standard was not yet published at the time of the commissioning of the BESS, but the direction was clear from the outset: its early drafts and outline forms indicated sections to be written on extinguishing and ventilation directly. The NFPA 855 2020 edition was officially released as a standard in August 2019.

## 6 Recommendations

This event clearly demonstrates the hazards associated with cascading thermal runaway in Li-ion battery energy storage facilities. As the industry moves forward from this incident, and looks to prevent similar incidents from occurring, it should consider implementing the following recommendations:

• Address vulnerabilities to thermal runaway cascading, ventilation, and suppression in existing and operational systems.

Update standards and codes to directly address cascading thermal runaway in future energy storage systems. Merely acknowledging cascading thermal runaway in the annex or appendix of the standard is insufficient to warn the industry of the hazard and falls short of requiring prevention.

• Implement ventilation and extinguishing or cooling systems to manage thermal runaway in future energy storage facilities.

• Implement battery and battery storage system designs that aim to slow or halt cascading or propagation of battery cells and modules during thermal runaway.

Implement education, training, and emergency response procedures that account for the risks and hazards of cascading thermal runaway—including flammable gases—and how to enter systems after a failure.

### 6.1 Current and future standards and codes should directly address cascading thermal runaway

Today’s codes and standards (specifically NFPA 855) do discuss the hazard of cascading thermal runaway, but without prescription. They are reluctant to prescribe that a battery module shall not cascade from cell to cell. One reason for this is that consensus-built codes and standards are intentionally technology-agnostic and should not impose restrictions or solutions on an industry that are perceived to increase cost or be commercially inviable.

But there are solutions available that are both cost effective and commercially viable. The slow standards development cycle is outpaced by the rapid evolution of technology. Today’s energy storage codes and standards must acknowledge this conflict, and attempt to reconcile it in their present drafts and revisions. As discussed in subsequent sub-sections, there are ways to limit or prevent cells from cascading in thermal runaway that are currently commercially viable options. Moreover, the standards and codes should be updated with recent developments in testing, research, and commercially available solutions involving cascading thermal runaway.

#### 6.1.1 Shortcomings that should be addressed in UL 9540 / UL 9540A and UL 1973

UL 1973 preceded the UL 9540A test method and was the only standard that addressed module-level battery fire risk, but it did not directly address the heat and gas load of cell-to-cell cascading thermal runaway. The deficiency in the “pass” criteria for UL 1973 is that the module could still fail catastrophically in thermal runaway even if it did not project flame. Internal heat could consume every battery inside; but, if it did not eject flame, it would pass. It could reach temperatures above 600oC and cause its aluminum frame to sag and yield, potentially contacting neighboring modules and cells. It could radiate enough heat energy to successfully induce thermal runaway in modules directly above or below it. It could create a plume of flammable gases without igniting them. It could do these things and still be considered to “pass” UL 1973 as long as no explosion or external flame was observed. Heat transfer from module to module is what occurred in the McMicken BESS. Hence, UL 1973 pass/fail criteria need to be revised to acknowledge unmitigated cellto-cell cascading and the resulting heat production and flammable gases.

In 2016, UL formed a task group to partially address these deficiencies in UL 1973, aligning the initiation criteria to a single cell, similar to IEC 62619 but not addressing the deficiency of the pass/fail criteria. As of the 2016 edition, UL 1973 fails to provide a means to arrive at a judgement of whether cell-to-cell cascading is an undesirable outcome, nor does it arrive at a judgement of what exterior temperatures are acceptable. [54, 55] IEC 62619 is a similar test that went a step further to prescribe that only a single cell be used for initiation and the DUT enclosure shall not rupture.

UL 9540A is a test method. In the last 2-3 years, the emergence of UL 9540A as an energy storage system thermal runaway test still has not directly addressed cell-to-cell cascading and its role in the creation of a potentially explosive atmosphere. [34]

UL 9540A does not prescribe that the cell cascading rate be measured directly, nor does it define pass/fail criteria. At present, there is no “pass” criteria for UL 9540A.

Measurement of a cell-to-cell cascading rate would be accomplished by instrumenting the module with a thermocouple array with sufficient quantity and density of locations to measure thermal runaway initiations as a function of time. As each cell ignites, it will present in the data as a sudden temperature spike. Post test, the time-series temperature data could be aligned with the locational map of the thermocouples such that a cascading rate and direction can be determined.

But UL 9540A does not require this. Instead, the test is designed to deliberately ignite modules and racks without defining how many cells are involved in the initiation. This is the first shortcoming of the test, because it can artificially “load” the initiating event and therefore affect the outcome. By failing to define how many cells are involved in initiation, the initial heat load can be variable by integer multiples, i.e., 2 cells, 3 cells, 4 cells, etc. Because the size and chemistry of battery cells varies widely across manufacturers, UL 9540A provides no means to benchmark results of the testing because the initiation criteria are uncontrolled. Therefore, the method does not quantify the natural cascading rate from a single cell, which should be a metric used to rank and grade the safety performance of modules.

Even after UL 9540 and UL 9540A were released, there was confusion in the market as to whether UL 9540A testing resulted in a certification or whether it was required. Many manufacturers did not understand whether the result of the testing was a pass/fail evaluation. Ultimately, requirement of such testing is AHJ dependent. UL 9540 and 9540A are now referenced in NFPA 855, so if an AHJ is knowledgeable of this code and enforces it, they will implicitly require UL 9540/9540A. [35] NFPA 855’s treatment of thermal runaway is explained below.

The UL 9540A test method, as it is written today, allows that thermal runaway will proceed to an entire rack and offers testing of suppression systems as an option. The method addresses the symptoms—not the cause—of the problem and does not provide evaluation procedures or criteria to determine what results are acceptable.

The UL 9540A test method is only meant to provide information but does not guide interpretation of the data or deliver a certification. In theory, the unit level test could result in full consumption of a rack and this

result would be reported without a judgement on whether this was poor performance. In fact, a manufacturer could point to the lack of rack-to-rack propagation and present it as a good result.

The test method has not yet defined pass/fail criteria because the expectation is that the industry will use the data to objectively calculate such criteria that are specific to the installation site. However, procedures on how to compute the explosion risk, ventilation requirement, and cascading thermal runaway risk are not defined.

Presentation of this data to an AHJ doesn’t immediately translate to a succinct understanding of the potential risks and what should be done to mitigate them. The AHJ may receive this data without interpretation and be uninformed on whether the result was a good or bad outcome. Since most AHJs are not battery experts, and perhaps not well trained in explosion modeling, they may not know enough to determine that an entire rack failure, or even multiple cells or modules, is a dangerous outcome.

#### 6.1.2 Shortcomings that should be addressed in NFPA 855

Neither UL 9540 nor NFPA 855 acknowledge that cascading thermal runaway should be first addressed at the cell level. If the risk of cascading was reduced, then the requirement for large scale testing is significantly reduced and the heat mitigation, extinguishing, and ventilation requirements may also lessen. Data demonstrates that flammable gases evolve from Li-ion cells undergoing thermal runaway and the chances of reaching the LFL are reduced if the number of cells undergoing thermal runaway is limited. NFPA only addresses cascading thermal runaway in the Annex of the 855 standard but does not prescribe codes or rules to address or prevent it. A single cell undergoing thermal runaway may not produce a flammable environment in a room, but unmitigated cascading will.

Even in today’s form, neither NFPA 855 nor UL 9540 prescribe that cell-to-cell cascading is not permitted, and neither standard acknowledges that testing should quantify the cascading rate.

Another human factors issue with NFPA 855 is that it does not distinguish between the roles of the parties (as shown in Figure 29) involved in the procurement and development of a battery energy storage system, and instead generally states that the owner or their authorized representatives shall provide training and response.

However, the insistence on the owner to provide training is not always appropriate with today’s practice in the energy storage industry, where owners of energy storage systems are more frequently private equity firms, investment funds, and independent power producers (IPPs) who might not be engaged directly in operations—and are less frequently utilities, although that is changing. Today, it is the practice in the industry that the EPC contractor—in this case AES and its subcontractors—take the lead in advising, training, and recommending practices for design and safety, as well as assume maintenance and response operations.

While it is certainly the responsibility of an owner of any piece of equipment to understand the risks associated with its use, it is the inherent responsibility of the manufacturers, designers, system integrators, operators, and service providers of such systems to educate customers on the risks involved and provide the training necessary. It is also the responsibility of the AHJ to know which codes to review and prescribe. This includes interpretation of the UL 9540A test data. Today, if an EPC contractor or system integrator does not have data on cascading thermal runaway, both the owner and the AHJ should request such data. This is the exercise of trying to uncover the “unknown-unknowns” as described in the Johari Window in Figure 28.

The key issue here is that the entire supply chain needs to know what to ask for— otherwise it may not be disclosed. If the battery manufacturer is not a direct participant in providing thermal runaway safety data for their product, the entire supply chain may not receive the necessary data. If the AHJ does not require it, the manufacturer may not voluntarily disclose it. If the owner does not prescribe it, then the supplier may not provide it, and the AHJ may not know to ask for it either. The cycle shown in Figure 30 demonstrates this classic chicken-egg problem—where the whole cycle is broken if any single party does not initiate, respond to, or request the appropriate information.

The fundamental issue of cascading thermal runaway was not addressed by the NFPA 855 drafts. Although, cascading thermal runaway is mentioned 37 times in at least one NFPA 855 draft, no requirements were written to address it. Today, in section C.7.2 (annex) of the 2020 release of NFPA 855, it states [35]: “Passive fire control features should be designed to limit the cascading effects of fire spread. This might include cell to cell (built into the module), module to module (built into the rack/or pack), rack to rack (built into the ESS room or container), or even protection from system to system propagation.”

However, this is in the annex, and is therefore not prescriptive. It does not acknowledge or inform a reader (such as a code official) about the significant heat hazard and flammable gases that will be produced if cascading is allowed to proceed unmitigated.

In Section 4.1.4.2, NFPA 855 states that a hazard mitigation analysis shall be submitted to the AHJ and shall evaluate the consequences of failure of a thermal runaway condition in a single module, array, or unit.

In table 9.2 of NFPA 855, it acknowledges that thermal runaway protection is necessary for Li-ion battery systems, but it assigns this role to the BMS. Thermal runaway, once started, is an electrochemical reaction that can’t be stopped electrically. There is no BMS—which is just a circuit board with control logic connected to sensors and contactors —that can stop thermal runaway in all circumstances.

Indeed, the collective awareness of thermal runaway preceded the commission dates of the McMicken BESS and Festival Ranch BESS. The fundamental concepts of cascading and an understanding that off-gassing could be flammable was publicly known, but not widely known to everyone. Perhaps the collective energy storage community had not yet made the logical calculation concerning how much gas volume and how fast gases could be generated if the thermal runaway propagated through all the battery cells within a module or rack.

Unfortunately, the McMicken BESS explosion demonstrates the consequences of this oversight. The extent of cascading and its rate are not addressed in past and present standards as measurement criteria, but they should be. Information was available to assess this hazard prior to the construction of this project and during its first year of commercial operations, but it was not emphasized enough among the suppliers of these systems or commonly referenced standards.

It is perhaps time for the industry to collectively acknowledge that cascading thermal runaway should be stopped or mitigated at the smallest unit possible. The next iteration of standards should finally acknowledge this in codified form. Until then, it is up to the supply chain to voluntarily develop, seek, design, and deploy solutions, while being at risk of deficiency of information.

The best way to overcome this deficiency in information exchange is to address it directly in the most commonly used codes and standards, i.e., NFPA 855, the upcoming IFC revision, UL 9540 and 9540A, and UL 1973.

### 6.2 Ventilation and cooling

In 2015-2017, DNV GL published results related to the management of flammable gases from Li-ion battery thermal runaway for stationary storage systems and their contribution to an explosion risk. [16] As shown in Figure 31, DNV GL models demonstrate the gas release rate of Li-ion cells as they burn; as one would expect, the gas release is larger when multiple cells are cascading versus when no more than one cell is burning at a time.

Additionally, Figure 32 exhibits the intense heat produced during thermal runaway, and demonstrates the temperature differences between large and small battery modules. For comparison, the McMicken BESS modules were 6.7 kWh, so they would be more in line with the left graph depicting the temperature for the large module.

In the McMicken event, the thermal runaway event cascaded to all 14 modules in Rack 15, ultimately burning 392 cells—releasing heat and flammable gases into the container. As designed, the McMicken BESS had no mechanism to prevent these flammable gases from building inside the container, or otherwise to stop the cascading thermal runaway event. Instead, the only fire suppression system equipped in the BESS was a Novec 1230 clean agent system, which as previously discussed, is unable to halt thermal runaway or mitigate the accumulation of flammable gases.25 In fact, the clean agent can actually exacerbate the hazard because its hold time permits unburned gases to accumulate. This event effectively demonstrates why ventilation and cooling are the necessary next steps for mitigating a thermal runaway event in a battery storage facility.

During the drafting of NFPA 855, Dr. Hill and other DNV GL employees presented to the NFPA 855 committee, and pointed out that the closed environment required for fixed fire suppression systems was in conflict with the need to ventilate flammable gases in Li-ion thermal runaway. DNV GL suggested that NFPA 855 should address the conflict. In most cases, the present language in NFPA 855 errs on the side of caution, requiring large spacing and extinguishing for the installation, unless testing shows that the system can operate with exceptions. These prescriptions address the symptoms rather than the cause of the problem. The obvious implication is that if the testing is not representative of a real-world failure, then the test is uninformative. The acceptance and interpretation of the testing results are ultimately at the discretion of the AHJ, who will be relying on technical analysis from others and is therefore at risk of information deficiency. Even with NFPA 855 now in effect with all of these requirements, cascading thermal runaway still remains unaddressed.

Since 2017-2018, some energy storage system integrators and designers have opted for the use of deflagration panels, which are panels meant to open in the event of a pressure increase (such as an explosion or deflagration) in order to relieve pressure, thereby potentially decreasing the violence of an explosion promulgated by constrained pressure. This practice can reduce the damage caused by an explosion. However, if cascading thermal runaway were limited, the need for such panels may be reduced.

As an additional measure, many energy storage systems are also configured with a “dry pipe” system, which allows first responders to connect water from the exterior of the system and deluge the interior without opening the system. While not a ventilation solution, water can serve as a cooling mechanism and remove heat, slow thermal runaway, and therefore reduce the gas production rate (recall the fire triangle in Figure 11 and the impact of slowed propagation in Figure 31). If cascading thermal runaway were addressed directly, the need for a water deluge would likely be reduced.

Since DNV GL’s commercially relevant testing in 2016 (published in February 2017), DNV GL commonly recommends these two measures—deflagration panels and dry pipe extinguishing—when reviewing the safety system design of energy storage systems. These are commercially available solutions that can often be incorporated by energy storage system integrators until better cascading protections are included in systems.

### 6.3 Preventing cascading thermal runaway

While the additional fire suppression measures (discussed in the previous section) help mitigate thermal runaway that has already started, these fire suppression requirements are greatly reduced if cell-to-cell propagation during thermal runaway is slowed or eliminated.

In Figure 33, based on testing conducted in the Con Ed report, [16] it is clearly demonstrated that cells that are permitted to touch - and subsequently go into thermal runaway - trap a significant amount of heat.

During the test it is shown that even after thermal runaway ended and the exterior temperatures are decreasing, the interior temperature (between the cells) remains nearly unmitigated. This residual heat can cause reignition of battery cells after extinguishing has occurred and can become an ignition source if it comes into contact with the flammable gases released during and after the thermal runaway event. In a large group of battery cells, such as the BESS, this residual heat poses a risk to restart propagation of thermal runaway for minutes to hours. Cells that have not yet undergone thermal runaway eventually will unless the batteries are separated or cooled (recall the fire triangle in Figure 11).

This residual heat or electrical energy that remains in a damaged Li-ion battery can be mechanically disturbed, causing reignition or “delayed cascading”. The term “delayed cascading” is perhaps more accurate than “reignition” because battery cells that touch one another will trap heat, so the thermal runaway triangle is never broken. When cells are in close thermal proximity, cascading will continue. This is what occurred at McMicken.

As an example of how “delayed cascading” occurs, consider the following thermal runaway test performed on a battery module with multiple pouch cells stacked together, face to face, without any barrier between them to prevent cascading. The timeline of what occurred is shown in Table 4, and the results are shown in Figure 34.

This test demonstrates several factors that are relevant to the McMicken incident. Tightly packed batteries trap and hold heat between them, as was shown in Figure 33. If a thermal barrier is absent or ineffective, or if there is no physical separation of the cells to prevent the trapping of heat, or if there is no mechanism to dissipate this heat, it will remain there as a thermal hazard that can reintroduce thermal runaway to any unburned cells and/or potentially ignite any flammable gases that continue to smolder. Cascading will continue in a design that does not prevent battery cells from coming into thermal contact.

Substantial research and testing have been done on the methods to reduce cascading and propagation. Battery manufacturers should take measures in the design of the system with an aim to slow or halt cascading or propagation of battery cells during thermal runaway. It is understood that it is easier to manage the heat and gases from a single cell rather than multiple cells. When multiple cells are undergoing thermal runaway, the associated heat and gas behavior is a building sum of all the smaller events that produced it (recall Figure 31). This is illustrated conceptually in Figure 35. The heat load and gas load follow similar behaviors, looking like bell curves over time.

However, there are several challenges in battery design related to the competing design goals of safe operation, good performance, high energy density, and low cost. These four design considerations factor into the performance and safety parameters of a battery system:

Compression of cells: Li-ion batteries may experience swelling and contraction during cycling. This requires compensation by some form of compression. Prismatic cells often have a case that is strong enough that they sometimes do not require external compression. Cylindrical cells need some form of constraint, though their case provides some structural resistance to swelling. Soft pouch cells, due to their inherent fragility, typically require compression. This can be done with aluminum or plastic plates. However, in designs where there is no protective thermal or physical barrier between cells, pouch cells are compressed as a unit, allowing pouch-to-pouch contact. This was how the LG modules in the McMicken BESS were designed.

Cooling of cells: The best cooling solution is a water or glycol cooled heat exchange mechanism, where the heat exchanger is usually metal and in direct contact with all or part of the battery cells. Many systems use air-cooling. Depending on the plate design, the cooling system can also serve as a thermal barrier to cascading thermal runaway. The LG modules in the BESS were air-cooled.

Removal of heat during thermal runaway: While an aluminum plate can cool a battery cell, thermal runaway can still propagate via concentrated heat touchpoints if the plate has no means to dissipate heat to an external reservoir (such as a coolant loop or open air). The aluminum, however, adds extra thermal mass and slows the transmission of heat. The redirection of heat away from neighboring cells is essential. Thus, a cooling system—passive or active—may also provide some form of passive heat dissipation to slow or prevent cells from cascading, even in the event that coolant may not be flowing. A simple aluminum plate between cells may not be enough to reduce cascading risk alone, if there is no means to redirect heat from the plates. For this reason, active cooled systems with plates in contact with a coolant loop are a better solution. There was no means to dissipate heat between cells in the LG modules in the BESS. There were no aluminum plates between cells in McMicken.

Low cost and high energy density: A lower overall balance of materials (BOM) will reduce the cost and increase the energy density of a battery system. However, the addition of materials to physically separate and create thermal barriers between cells may increase mass and cost, although offering better protections against cell-to-cell and module-to-module heat transfer. The LG modules in the BESS are considered to be of high energy density at a competitive cost.

As demonstrated in Figure 36, there are several commercially available approaches that can be used to achieve all of the above goals.

A. The Chevrolet Volt battery pack uses aluminum plates sandwiched between LG Chem battery pouch cells to provide superior cooling and substantial protection against cascading.26 The aluminum plates are torqued to a compression specification that keeps the pouch cells compressed. Because the aluminum plates are actively water cooled, they are highly effective at distributing and controlling heat. Also, the plates are thick enough that they can dissipate heat which slows or prevents heat from being concentrated in neighboring cells during thermal runaway. This is a very robust design, and as such, can increase cost or the BOM of the battery pack. The plates themselves provide structural rigidity, which reduces the need for exterior structural enhancements. Similar alternating aluminum plate designs have been adopted by several battery pack integrators in the marine industry because of the high safety and performance benefits.

B. High temperature resistant plastics, such as some types of Acrylonitrile Butadiene Styrene (ABS) plastic, can have a strong resistance to direct flame and are also poor enough conductors of heat that they may slow or prevent propagation of thermal runaway from a single initiating cell.27 More advanced technologies such as intumescent materials (like what is offered by Pyrophobic), or phase change materials (such as what is offered by AllCell Technologies) can absorb the heat of a single cell and prevent its transfer to the nearest neighbors. These materials can also be used to provide compressive strength. Phase change materials are better thermal conductors and thermal heat sinks than intumescent materials or some plastics, though both phase change materials and intumescent materials are shown to effectively slow and halt thermal runaway between nearest neighbors of

cylindrical cells. These materials can be used optimally to reduce mass; however, not without cost considerations. The directional venting of gases should also be managed, as jetting gases can also transfer heat and produce cascading. [22]

C. The Tesla Model S and Model 3 battery module approach begins with smaller cells that contain less energy such that if one were to catastrophically fail, the total heat energy it would release is limited by its small size—i.e., there is less heat concentrated because there is less initiating mass, and because the cells are highly energy dense, the energy density and BOM are also reduced. 28 In addition to the small cell size, an aluminum cooling ribbon is woven into the battery pack structure to provide cooling and some structural reinforcement to the already-semi-rigid cylindrical cells. The aluminum ribbon is the main mechanism to cool the cells; however, it is part of a combined mass of materials to reduce the rate of heat transfer, and thus cascading, during a failure. While not as deliberate in its design as the Volt pack, the aluminum in the Tesla pack adds thermal mass to slow cascading. And because the cells themselves are small, a single cell initiation is brief and less intense than it would be if the cells were larger.

D. The last technique is to increase the air gap between cells. This is often done with prismatic cells such as these Samsung cells, which provide their own structural strength and prevent swelling of the interior battery materials. The air gap needs to be analyzed to determine its sufficiency in preventing the concentration or transfer of heat or cell-to-cell contact during a thermal runaway event. The air gap can be optimally calculated to reduce heat transfer while also accounting for probable swelling, while preserving energy density for the module. An air gap alone may not prevent cascading and the directional venting of gases are also a consideration to avoid heat transfer.

The examples above suggest that while there is no universal method for completely mitigating cascading thermal runaway, just as there is no silver bullet for completely mitigating gasoline fires, there are means within which the likelihood of cell-to-cell and module-to-module heat transfer, and therefore therma runaway, can be diminished. The four methods summarized above all achieve the same end: they either reduce, redirect, absorb, deflect, or slow the production and transmission of heat.29

Another method for mitigating cascading thermal runaway in battery design is to use smaller cells and compartmentalization of batteries. Put simply, limited mass means limited fuel, and limited fuel means less energy released over time, which in turn decreases the overall power of the event.

As shown in Figure 37, in the aggregation of dozens of individual thermal runaway tests on battery cells, a plot of duration vs. energy density (Ah/kg) is shown. A cluster of data points in the 2-20-minute timeframe is shown in the upper left within the red circle. These tests were performed on cylindrical 18650 battery cells. The rest of the data points are pouch or prismatic battery cells. The cylindrical cells have somewhat higher energy density, but more notable is that the duration of the burn is much less, perhaps 3x less, than a pouch or prismatic cell. Cylindrical cells are typically 3-5 Ah whereas pouch cells are larger, typically > 30 Ah.

This demonstrates that smaller cells burn for a shorter amount of time and emit less gas overall. Therefore, a design philosophy that can isolate, contain, and independently vent sub-groups of batteries will allow for safer management of battery thermal runaway by reducing the overall quantity of explosive gases. Both smaller cells and smaller sub-groups would achieve the same goal. It should then be no surprise that there is a correlation in Figure 38 between module mass and peak temperatures reached.

There is an acknowledgement from multiple suppliers that next generation system designs will incorporate either liquid cooling into rack-type, containerized systems, or more modular designs that eliminate the container altogether; and instead focus on rack-like, self-contained, compartmentalized, pod-like modular systems that can be rapidly assembled on site. Self-contained and modular systems with IP-rated enclosures an independent BMS, and sometimes an independent inverter, isolate batteries into sub-groups and reduce the available mass for thermal runaway.

Another benefit of such compartmentalized and pod-like systems is they can be accessed from the exterior without need for human entry. While there is still a need to vent flammable gases, suppress incipient fires, and provide deflagration protection, the hazard to human health and safety is reduced. The engineering and design methods for such systems exists today.

In sum, systems designed with smaller, self-contained, compartmentalized groups of batteries (cells and/or modules) that independently and controllably vent to the open atmosphere will avoid the accumulation of large volumes of trapped gases and reduce explosion risk. Of course, no matter how small the batteries (cells or modules), any system that encloses the batteries within any type of enclosure where gases can concentrate will create a potential explosion hazard if a substantial amount of batteries are off-gassing without ventilation.

### 6.4 Changes in emergency response procedures

There are many stakeholders who may be involved in the response to a potential battery thermal runaway event. This report addresses several key stakeholders:

1. First responders (including fire fighters and ambulance services) are trained to respond, assess, and extinguish fires and treat the injured. They want to understand the operating and risk characteristics of facilities to which they may need to respond. Firefighters must safely mitigate the hazard and stay at the scene as long as necessary to deem the hazard under control but must eventually relinquish the scene to someone who has control over it, because first responders may need to depart to another emergency.
2. Operations and maintenance staff monitor the system and make decisions based on system status and public and personnel safety risk. O&M staff are already on call and are typically expected to respond to a site within a limited time (30 minutes to 2 hours) depending on location. The response staff from the O&M party is expected to be knowledgeable of the site, able to access the site, and provide guidance to the first responders. They should be in possession of the applicable documents that constitute an emergency response approach. It is often the O&M staff that take control of the site after first responders depart.
3. Staff representing the ownership of the project(s), if applicable and local, may also be involved in response to the site. In the case of APS, which is a utility not unfamiliar with managing, maintaining, and responding to calls involving project sites, their response to the site is uniquely different than, for example, a private equity firm or project owner that may not have the technical staff or capability to respond. Usually the O&M provider offers and can execute a safety management plan or ERP on behalf of the owner.

As was shown in the APS Johari Window in Figure 28, the industry must work to identify as many “unknownunknowns” as possible and move them to the categories of “known-knowns” and “known-unknowns”. Dr. Hill and DNV GL have the privilege of performing this analysis after the fact. DNV GL’s historical perspective on battery safety may be beneficial to help explain why the APS Johari Window may look this way. Dr. Hill identified flammability and explosion risk from off-gassing electrolytes as a “known-unknown” as early as 2010 while doing battery safety research, but that doesn’t mean that the industry had collectively come to this conclusion. For DNV GL, explosion risk of off-gases from individual battery cells became a “known-known” in 2012 when testing was conducted to confirm it. [20] DNV GL commissioned its first FMEA on a commercial marine energy storage system in 2012 and addressed this hazard. DNV GL’s first commercial FMEA on stationary energy storage with a Bowtie model was performed in 2014, and the first commercial fire plume study on stationary storage was commissioned in 2015, both of which addressed the risk of cascading thermal runaway and management of explosive gases. DNV GL conducted its first commercial quantitative risk analysis on stationary storage in 2016-2017, including safety integrity leve (SIL) and layers of protection analysis (LOPA) much like it is done in the oil and gas industry. It wasn’t until 2015-2016 that Dr. Hill and DNV GL began actively marketing these services and incorporating them into a designated service line, presenting results to committees, AHJs, standards organizations, and public events. And, for Dr. Hill and his colleagues at DNV GL, it wasn’t until the large scale fire testing that occurred in the Con Ed/NYSERDA program that the potential for large quantities of flammable gases and the significant deflagration potential became “known-knowns”, at which point the results were published publicly in that report in February 2017. Therefore, while DNV GL’s “known-unknowns” matured earlier than much of the

industry, that only occurred because there was an active effort to uncover them through R&D and strategic customer relationships.

Today, the best way to advance knowledge for stakeholders is to conduct HAZID and/or FMEA exercises with Bowtie models during the design and commissioning of projects, and include worst-case thermal runaway hazards in plume studies as projects are being commissioned.30,31,32 These exercises reduce the quantity of items in the right-hand side of the Johari Window and move them to the left-hand side. The risks should be sorted and identified from greatest to least, so the greatest risks can be mitigated.

Emergency response plans should then be constructed from the outcomes of these exercises. The ERP documentation itself should include details such as current system drawings, specific emergency responder instructions and guidance, applicable Safety Data Sheets (SDS), a comprehensive list of any combustible products and their off-gassing capabilities and other hazardous materials, site evacuation plans, and other relevant environmental and supporting material.

Once these emergency response plans are derived from the risk-based analyses described above, the system integrator and/or O&M contractor is likely to be the party most familiar with the details of the system and should have a standard operating procedure (SOP) in place—which may be the same document as the ERP—in order to advise or inform the facility owner, facility operator and firefighting personnel. It is important to ensure that the ERP clearly articulates the various roles of responders and that participating parties in a response are educated and understand those roles before an event occurs. Each of the stakeholder groups referred to earlier should be outlined in the ERP along with a command hierarchy and specific responsibilities for each entity. It is best practice to provide training to the fire department before and during commissioning of an energy storage project, alerting them to what is being installed, what it may do during a fire, what hazards may be present, how to assess the system condition, and what response to take. All potential first responders from multiple jurisdictions should be considered for inclusion in training as it is common practice for multiple departments across townships, municipalities, counties, etc. to share in response efforts; especially across specialty groups such as HAZMAT units. Periodic updates and refresher training should also be supplied because personnel change over time and information will be updated. Training should also be delivered to appropriate internal owner groups and any third-party contractors working for the owners or O&M provider. This training should refer back to the risk analyses described above.

The owner may be involved in training of the first responders, but because the on-call O&M staff are usually arriving at the scene at the same time or shortly after first responders, they are typically the first point of contact. O&M personnel likely have remote data access to determine the status of what is occurring inside the BESS container. O&M personnel are expected to be knowledgeable about the system operations and act as the subject matter expert to the first responders. However, it may be the case that the owner of the system may acquire more responsibility for O&M or emergency response over time, relying on training from the O&M provider.

An owner relies heavily on its suppliers for informing a viable emergency response plan. Development of safety plans requires data sharing, transparency, and acknowledgement of safety risks, even if they are perceived to have a low probability of occurring. This is the purpose of the HAZID, FMEA, and Bowtie exercises.

The same is true for the inclusion of compliance with other NFPA codes in the system design. While NFPA 68 or NFPA 69 addressed explosion hazards in 2016, and such codes could have addressed the risk at this site, they would not have been considered if the system integrator/EPC contractor was given the impression from the battery manufacturer that any explosion risk was minimal. Similarly, the owner would likely not ask for a review of these codes if there was an impression that explosion risk was minimal. Therefore, discussions of explosion risk would not have been presented to the AHJ, and the AHJ would not cite NFPA codes related to explosion risk as a requirement for permitting.

Today, there is an expectation that safety testing and certifications are conducted and obtained, respectively, by the battery manufacturer and provided to the system integrator, EPC contractor, and owner. This was true even in 2016 when UL 1973 was commonly prescribed. But today, it is expected that the EPC contractor, developer, and owner will participate in the extra step of using this information to inform the ERP. The risk modelling exercises described herein aid in creating transparency in this process.

The EPC contractor would then take appropriate design measures and present their analysis and design changes to the AHJ, who would then prescribe any additional NFPA compliance requirements. Training materials and outreach can be prepared by the EPC contractor or system integrator for the owner, the owner’s O&M service provider, municipal first responders, AHJs, and other third parties who would benefit from such training.

Yet, even if all parties take the training steps prescribed above, the fundamental issue of cascading and the rate of generation of gases must still be addressed in the near term, through technical and commercia solutions and standards development.

Training for first responders must directly address the explosion risk, state clearly how, where and what kind of gases are generated, and explain how to detect gases and ventilate, regardless of what the standards presently require. Guidance should also include information to assist first responders in determining how and where to create and enforce a safe perimeter around a facility during an incident and provide specific instruction on how and when to safely open or enter a BESS. Safety perimeters can be informed by the plume study.

By allowing all parties to observe the HAZID and/or FMEA and Bowtie-based risk analyses, it is much more likely that hazards will be moved to the “known-knowns” and “known-unknowns” of the Johari Window for all of the parties involved such that better emergency response plans can be created and communicated.

Just as the lack of information during the design and build phases of the project contributed to the event, so too did the lack of real-time situational awareness during the emergency response. Additional technologies should be utilized to give responders information on the environment within a BESS (e.g., flammable gas concentration, heat level, etc.) and these technologies should be designed for remote access and increased resiliency using external power sources, backup communications channels, and similar hardening techniques.

Lastly, in this event, the on-site accessibility of the ERP and other emergency-related materials, such as the SDSs, was hampered because they were stored inside the BESS container. This incident underscored the need to have multiple copies of these materials available both outside the container and in readily accessible electronic formats (held by both the owner and the O&M provider) so they can be referenced when necessary.

## 7 Conclusions

There are best practices available in the industry that have been outlined and documented in this report. As shown in Figure 39, a few barriers, if in place, would have likely prevented this incident. First, cell quality should be a major focus in the industry going forward. In addition, the following factors should be addressed:

Cell-to-cell cascading: A barrier to limit the rate, or prevent cascading altogether, would have reduced the quantity of gases in the atmosphere of the container such that they would have never met the LFL, regardless of whether the Novec 1230 leaked from the BESS container. See “Preventing cascading ” on page 53.

Module-to-module cascading: If cell-to-cell cascading is addressed, it is likely that module-tomodule cascading would also be addressed. See “Preventing cascading ” on page 53. However, the close proximity of modules in the McMicken BESS, separated by aluminum (with its corresponding low yield strength at these temperatures), allowing it to sag, permitted the thermal runaway to continue to propagate long after the Novec 1230 was discharged. The rate of propagation allowed the gases to exceed the LFL.

Ventilation and cooling: Eventually, the container would need to be accessed in order to regain safe control. While a lower air changeover rate would have allowed the Novec 1230 to remain for a longer period of time, the quantity of Novec 1230 or any other fixed suppression agent will always be limited by the space available, and the flammable gases must eventually be dissipated before access is gained. A means to safely ventilate the system in a controlled manner will reduce the flammability of the atmosphere. Prior to ventilating, the McMicken system could have been cooled with water using a dry pipe system. The use of deflagration panels may have decreased the pressure of an explosion if it were to occur. See “Ventilation and cooling” on page 51.

Proper extinguishing: A clean agent or aerosol alone is not enough to prevent cascading thermal runaway or manage the concentration of flammable gases that will follow. A combined strategy of fire suppression followed by ventilation and cooling strategies should be a requirement going forward. See “Ventilation and cooling” on page 51.

Response plans and entry: Response procedures that incorporated system monitoring, the detection of gases, ventilation practices, extinguishing methods, and information to gather before entry, may have prevented first responders from opening the door in unsafe conditions. If such training is conducted early in the project development and commissioning process, it permits the local jurisdiction and their first responder parties to learn about the system and be aware of its hazards. These procedures should be documented, available outside the BESS container or building, and demonstrated through training which should be refreshed and updated periodically.

At any stage of Figure 39, if the mass available to burn was limited, compartmentalized, and independently vented, the rest of the chain may have been broken. It should be noted that the final barrier is actually system entry, meaning that all the prior measures must be complete and verified before access is granted.

Present codes and standards are insufficient in addressing cascading thermal runaway and prescribe the extinguishment of thermal runaway through conventional firefighting means. The standards should instead acknowledge that cell-to-cell cascading is the primary risk to address in the scale of the thermal runaway hazard, and there are solutions which prevent or slow cascading thermal runaway. Owners, system integrators, vendors and suppliers, and O&M service providers should also focus on educating stakeholders on the processes shown in Figure 39, and develop and implement training necessary to execute the third, fourth, and fifth barriers.

In addition to what was proposed in “Recommendations” on page 46, it is a strong recommendation of this report that:

a. The first (limit cell cascading) and second (limit module cascading) barriers be addressed by the industry both technically and commercially; and

b. The next iteration of standards addresses ventilation, extinguishing, cooling, and training while acknowledging the unique, non-fire properties of cascading thermal runaway, the gases it generates, and ways to mitigate it.
