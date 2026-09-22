#### 3.3.1 State of Charge and Depth of Discharge

The user typically wishes to know the battery's SOC or DOD, for the same reason that a car driver wishes to know how much fuel is left in the car's tank—to estimate how long before it is empty.

For however inaccurate car fuel gauges may be, fluid level measurement is easier to do than SOC estimation. The fuel level in a tank can be measured directly, while a cell's SOC cannot. Estimating SOC and DOD in a Li- Ion pack is, at best, an inexact science, and, at worse, a wild guessing game.

There is no direct way of measuring the SOC of a Li- Ion battery. There are indirect ways of estimating it, but each suffers from limitations. Among them, two commonly used methods are:

- Voltage translation;
- Current integration (coulomb counting).

Both techniques are useful, but each by itself is unable to reliably estimate SOC in a Li- Ion battery. By combining them, a reasonable estimate of SOC is possible.

##### 3.3.1.1 Voltage Translation

With some cell chemistries, the battery voltage decreases more or less linearly as the battery is discharged, so one may consider using a simple voltmeter as an SOC indicator (Figure 3.26). Knowing the relationship of open- circuit battery voltage and SOC (which is the idea behind voltage translation) allows the voltmeter to be calibrated to report an approximate SOC.

A major limitation with this technique is that the battery's terminal voltage is affected by parameters other than SOC. Having prior knowledge of the way those parameters affect the terminal voltage, it may be possible to provide a certain amount of compensation, allowing voltage translation to be a useful way to estimate that battery's SOC.

The usefulness of voltage translation in Li- Ion batteries is limited, because, for most of its SOC range, the voltage of a Li- Ion cell remains rather constant (Figure 3.27). With zero cell current, using a precise voltmeter (with accuracy on the order of \(1\mathrm{mV}\) ), and allowing a long time for the cell voltage to settle (the time constant is on the order of tens of minutes), voltage translation is possible in a laboratory (though mostly impractical in a product). Yet, the voltage of a Li- Ion cell does change significantly at both ends of its OCV versus SOC curve. Therefore, voltage

translation can be used to estimate the SOC of a Li- Ion cell when it is nearly full or nearly empty.

##### 3.3.1.2 Coulomb Counting

Integrating the current into or out of a battery gives the relative value of its charge, just as counting currency in and out of a bank account gives the relative amount in the account. The operative word here is relative. Like any definite integral, coulomb

counting needs a starting point. If the initial charge in the battery is known, from then on coulomb counting can be used to calculate charge. For example, a 2- A current into a battery, for 3 hours, will add \(2^{*}3 = 6\) Ah charge to the battery (Figure 3.28). The battery's DOD will have decreased by \(6\) Ah. Without knowing the initial DOD, however, we cannot know the final DOD. (The charging process is \(100\%\) efficient. See Section 1.2.5.2. )

Coulomb Counting can be a very accurate technique, with two limitations:

- Leakage current within a cell does not go through the current sensor and is therefore not taken into account;- Offset in the measurement of the battery current will result in the SOC drifting up (or down) over time (in any integration, a nonzero constant in the variable being integrated causes the integral to change over time).

Coulomb counting works well with Li- Ion cells because they have low leakage. Drift remains a major limitation (Figure 3.29) due to the offset in the current sensor, especially Hall effect sensors (see Section 3.1.3.2).

Drift can become significant in applications that, for long periods, use very little battery current or shuttle current back and forth. In particular:

- Standby batteries: even if the battery is full, a small offset in the current sensor in the discharging direction will result in the reported SOC drifting all the way to \(0\%\) SOC over time.- HEV pack: uses energy from the battery when it needs it, and replenishes it when it can, trying to maintain a \(50\%\) SOC. While the reported SOC may very well stay around \(50\%\) , over time the actual SOC will drift due to the small offset in the current sensor. Eventually the actual battery charge will approach either the full or the empty state (Figure 3.30).

An HEV can calibrate its current sensor to mostly eliminate SOC drift due to offset in the current sensor. Once in a while, the vehicle control unit (VCU) may

turn off the motor AC inverter for a while and notify the BMS that the battery current is 0 so that the BMS may save the current sensor reading as its offset, and later use that offset to correct readings.

##### 3.3.1.3 Combining the Techniques

Coulomb counting can be used to estimate the DOD of a Li- Ion pack as long as there's a way of calibrating it at some point, and often enough to overcome drift. Going back to the bank account analogy: balancing your check book synchronizes the amount you believe is in your account with the amount that your bank says is in that account. Similarly, coulomb counting needs a way to calibrate its result, so that the charge it reports is the actual DOD. Voltage translation provides a way of doing so, just as balancing a checkbook does for a bank account. Combining these

two techniques results in a reasonable way of estimating of DOD in a Li- Ion cell (Figure 3.31):

- The battery current is integrated (coulomb counting) to get the relative charge in and out of the battery.- The battery voltage is monitored, to calibrate the DOD when the actual charge approaches either end.

If the DOD estimated through coulomb counting is uncalibrated (it is not equal to the actual DOD), eventually the battery will be charged or discharged so far that voltage translation can be used to estimate SOC and, knowing the capacity can be translated to DOD, the estimated DOD can be calibrated. For example, if the actual DOD of a 100- Ah Li- Ion cell is 20 Ah but the BMS estimates its DOD to be 50 Ah, the cell may be charged until its voltage reaches a threshold (say, 3.4V), which corresponds to an actual SOC (say, \(90\%\) ). At that point, the BMS sets the estimated SOC to \(90\%\) , calculates the corresponding DOD at 10 Ah, and calibrates the DOD (Figure 3.32).

Going back to the issue of drift, let's see how combining these two techniques affect DOD estimation in the two applications we considered earlier.

- Standby systems: the pack is kept full, so voltage translation is used, avoiding the long term drift of coulomb counting.- Hybrid car (HEV) traction packs: when the actual SOC drifts in such way that a cell voltage reaches a threshold at either end, the BMS calibrates the SOC based on that voltage (Figure 3.33). A more sophisticated approach can calibrate the SOC in the process of balancing the traction pack.

For the above method to work, the actual pack capacity must be known so that the conversion between SOC and DOD will be correct. Otherwise, the estimated

SOC will appear to change too slowly or too quickly, and SOC calibration will be incorrect (Figure 3.34). In an application in which this could present a problem, the battery capacity must be measured (see Section 3.3.2).
