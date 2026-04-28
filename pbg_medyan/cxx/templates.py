"""Bundled MEDYAN chemistry input templates.

These match the upstream ``examples/actin_only/`` and
``examples/50filaments_motor_linker/`` chemistry files (with attribution).
A user can pass their own chemistry text via ``MedyanCxxProcess(config={
'chemistry_text': '...'})`` to fully customize.
"""

# Source: https://github.com/medyan-dev/medyan-public/blob/main/examples/actin_only/chemistryinput.txt
#
# Note on naming: despite the upstream directory being called "actin_only",
# the file actually declares SPECIESMOTOR / SPECIESLINKER and the
# corresponding MOTORREACTION / LINKERREACTION / MOTORWALKINGREACTION lines.
# MEDYAN's chemistry validator requires the reactions to register the species
# as "motor species" — without them, declaring NUMMOTORHEADSMIN/etc. in
# the systeminput fails with "minimum motor heads (1) does not match the
# number of motor species (0)". We mirror upstream verbatim so the preset
# is compatible with the bundled motor/linker mechanics defaults.
ACTIN_ONLY = """\
############################ SPECIES ###########################
# SPECIESDIFFUSING <NAME> <COPYNUMBER> <DIFFCOEFF> <RELEASETIME> <REMOVALTIME> REG
SPECIESDIFFUSING: AD 1000 20e6 0.0 0.0 REG

SPECIESDIFFUSING: MD 0 0.2e6 10.0 0.0 REG
SPECIESDIFFUSING: LD 0 2.0e6 0.0 0.0 REG

SPECIESFILAMENT: FA 0

SPECIESPLUSEND: PA 0

SPECIESMINUSEND: MA 0

SPECIESBOUND: LEA 0
SPECIESBOUND: MEA 0

SPECIESMOTOR: MOA 0
SPECIESLINKER: LA 0

LINKERBINDINGSITE: LEA 0
MOTORBINDINGSITE: MEA 0

########################### REACTIONS ##########################

# Normal polymerization
POLYMERIZATIONREACTION: 0 AD:DIFFUSING + PA:PLUSEND -> FA:FILAMENT + PA:PLUSEND 0.151
POLYMERIZATIONREACTION: 0 AD:DIFFUSING + MA:MINUSEND -> FA:FILAMENT + MA:MINUSEND 0.017

# Normal depolymerization
DEPOLYMERIZATIONREACTION: 0 FA:FILAMENT + PA:PLUSEND -> AD:DIFFUSING + PA:PLUSEND 1.4
DEPOLYMERIZATIONREACTION: 0 FA:FILAMENT + MA:MINUSEND -> AD:DIFFUSING + MA:MINUSEND 0.8

# Motor and linker binding and unbinding
MOTORREACTION: 0 MEA:BOUND:1 + MEA:BOUND:2 + MD:DIFFUSING <-> MOA:MOTOR:1 + MOA:MOTOR:2 0.2 1.7 175.0 225.0
LINKERREACTION: 0 LEA:BOUND:1 + LEA:BOUND:2 + LD:DIFFUSING <-> LA:LINKER:1 + LA:LINKER:2 0.009 0.3 30.0 40.0

# Motor walking
MOTORWALKINGREACTION: 0 MOA:MOTOR:N + MEA:BOUND:N+1 -> MOA:MOTOR:N+1 + MEA:BOUND:N 0.2
"""

# Source: https://github.com/medyan-dev/medyan-public/blob/main/examples/50filaments_motor_linker/chemistryinput.txt
ACTIN_MOTOR_LINKER = """\
############################ SPECIES ###########################
SPECIESDIFFUSING: AD 1000 20e6 0.0 0.0 REG
SPECIESDIFFUSING: MD 0 0.2e6 10.0 0.0 REG
SPECIESDIFFUSING: LD 0 2.0e6 0.0 0.0 REG

SPECIESFILAMENT: FA 0
SPECIESPLUSEND: PA 0
SPECIESMINUSEND: MA 0

SPECIESBOUND: LEA 0
SPECIESBOUND: MEA 0

SPECIESMOTOR: MOA 0
SPECIESLINKER: LA 0

LINKERBINDINGSITE: LEA 0
MOTORBINDINGSITE: MEA 0

########################### REACTIONS ##########################

POLYMERIZATIONREACTION: 0 AD:DIFFUSING + PA:PLUSEND -> FA:FILAMENT + PA:PLUSEND 0.151
POLYMERIZATIONREACTION: 0 AD:DIFFUSING + MA:MINUSEND -> FA:FILAMENT + MA:MINUSEND 0.017
DEPOLYMERIZATIONREACTION: 0 FA:FILAMENT + PA:PLUSEND -> AD:DIFFUSING + PA:PLUSEND 1.4
DEPOLYMERIZATIONREACTION: 0 FA:FILAMENT + MA:MINUSEND -> AD:DIFFUSING + MA:MINUSEND 0.8

MOTORREACTION: 0 MEA:BOUND:1 + MEA:BOUND:2 + MD:DIFFUSING <-> MOA:MOTOR:1 + MOA:MOTOR:2 0.2 1.7 175.0 225.0
LINKERREACTION: 0 LEA:BOUND:1 + LEA:BOUND:2 + LD:DIFFUSING <-> LA:LINKER:1 + LA:LINKER:2 0.009 0.3 30.0 40.0

MOTORWALKINGREACTION: 0 MOA:MOTOR:N + MEA:BOUND:N+1 -> MOA:MOTOR:N+1 + MEA:BOUND:N 0.2
"""


PRESETS = {
    'actin_only': ACTIN_ONLY,
    'actin_motor_linker': ACTIN_MOTOR_LINKER,
}
