from dataclasses import dataclass


@dataclass
class AccessState:
    """
    Discrete 'state of access' of a flexibility unit.

    Examples:
        - A PV inverter that can only run at [0%, 30%, 70%, 100%] of available power.
        - A heat pump with modes [OFF, LOW, HIGH].
        - A demand response asset with [NO_SHED, PARTIAL_SHED, FULL_SHED].

    The 'utilisation' value is a normalised control factor in [-1.0, 1.0],
    interpreted by the subclass. Typical conventions:

        utilisation > 0 : drawing from flexibility (more consumption / less generation)
        utilisation < 0 : injecting to flexibility (less consumption / more generation)
        utilisation = 0 : no activation (baseline)

    Subclasses decide how exactly utilisation maps to power / energy.
    """
    name: str
    utilisation: float
    description: str = ""
