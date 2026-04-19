"""Unit Converter tool — convert between physical units.

Especially useful for nuclear/physics domains where unit conversions
between eV, MeV, GeV, Joules, atomic mass units, etc. are frequent.
"""

import re
from typing import Any

from src.tools.tool_registry import BaseTool
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Conversion factors to SI base units
# Energy: base = Joules
ENERGY_TO_J = {
    "j": 1.0, "joule": 1.0, "joules": 1.0,
    "kj": 1e3, "kilojoule": 1e3,
    "mj": 1e6, "megajoule": 1e6,
    "cal": 4.184, "calorie": 4.184,
    "kcal": 4184.0, "kilocalorie": 4184.0,
    "kwh": 3.6e6,
    "ev": 1.602176634e-19, "electronvolt": 1.602176634e-19,
    "kev": 1.602176634e-16,
    "mev": 1.602176634e-13,
    "gev": 1.602176634e-10,
    "tev": 1.602176634e-7,
    "btu": 1055.06,
    "erg": 1e-7,
}

# Mass: base = kg
MASS_TO_KG = {
    "kg": 1.0, "kilogram": 1.0,
    "g": 1e-3, "gram": 1e-3,
    "mg": 1e-6, "milligram": 1e-6,
    "ug": 1e-9, "microgram": 1e-9,
    "lb": 0.453592, "pound": 0.453592,
    "oz": 0.0283495, "ounce": 0.0283495,
    "ton": 1000.0, "tonne": 1000.0, "metric_ton": 1000.0,
    "amu": 1.66053906660e-27, "dalton": 1.66053906660e-27, "u": 1.66053906660e-27,
}

# Length: base = meters
LENGTH_TO_M = {
    "m": 1.0, "meter": 1.0, "metre": 1.0,
    "km": 1e3, "kilometer": 1e3,
    "cm": 1e-2, "centimeter": 1e-2,
    "mm": 1e-3, "millimeter": 1e-3,
    "um": 1e-6, "micrometer": 1e-6, "micron": 1e-6,
    "nm": 1e-9, "nanometer": 1e-9,
    "pm": 1e-12, "picometer": 1e-12,
    "fm": 1e-15, "femtometer": 1e-15, "fermi": 1e-15,
    "angstrom": 1e-10, "å": 1e-10,
    "mi": 1609.344, "mile": 1609.344,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "in": 0.0254, "inch": 0.0254,
    "yd": 0.9144, "yard": 0.9144,
    "au": 1.496e11,
    "ly": 9.461e15, "light_year": 9.461e15,
    "pc": 3.086e16, "parsec": 3.086e16,
}

# Time: base = seconds
TIME_TO_S = {
    "s": 1.0, "sec": 1.0, "second": 1.0,
    "ms": 1e-3, "millisecond": 1e-3,
    "us": 1e-6, "microsecond": 1e-6,
    "ns": 1e-9, "nanosecond": 1e-9,
    "min": 60.0, "minute": 60.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0,
    "day": 86400.0,
    "week": 604800.0,
    "year": 3.156e7, "yr": 3.156e7,
}

# Temperature conversion is special (offset-based)
# Radioactivity: base = Becquerel (Bq)
RADIOACTIVITY_TO_BQ = {
    "bq": 1.0, "becquerel": 1.0,
    "kbq": 1e3,
    "mbq": 1e6,
    "gbq": 1e9,
    "ci": 3.7e10, "curie": 3.7e10,
    "mci": 3.7e7, "millicurie": 3.7e7,
    "uci": 3.7e4, "microcurie": 3.7e4,
}

# Radiation dose: base = Sievert (Sv)
DOSE_TO_SV = {
    "sv": 1.0, "sievert": 1.0,
    "msv": 1e-3, "millisievert": 1e-3,
    "usv": 1e-6, "microsievert": 1e-6,
    "rem": 0.01,
    "mrem": 1e-5, "millirem": 1e-5,
    "gy": 1.0, "gray": 1.0,  # Approximate for gamma
    "mgy": 1e-3,
    "rad": 0.01,
}

# Pressure: base = Pascal
PRESSURE_TO_PA = {
    "pa": 1.0, "pascal": 1.0,
    "kpa": 1e3,
    "mpa": 1e6,
    "bar": 1e5,
    "mbar": 100.0,
    "atm": 101325.0, "atmosphere": 101325.0,
    "psi": 6894.76,
    "torr": 133.322,
    "mmhg": 133.322,
}

UNIT_CATEGORIES = {
    "energy": ENERGY_TO_J,
    "mass": MASS_TO_KG,
    "length": LENGTH_TO_M,
    "time": TIME_TO_S,
    "radioactivity": RADIOACTIVITY_TO_BQ,
    "dose": DOSE_TO_SV,
    "pressure": PRESSURE_TO_PA,
}


def _find_unit(unit_str: str) -> tuple[str, dict, float] | None:
    """Find which category a unit belongs to and its conversion factor."""
    unit_lower = unit_str.lower().strip()
    for category, table in UNIT_CATEGORIES.items():
        if unit_lower in table:
            return category, table, table[unit_lower]
    return None


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float | None:
    """Handle temperature conversions (offset-based)."""
    from_u = from_unit.lower().strip()
    to_u = to_unit.lower().strip()

    temp_names = {
        "c": "c", "celsius": "c", "°c": "c",
        "f": "f", "fahrenheit": "f", "°f": "f",
        "k": "k", "kelvin": "k",
        "r": "r", "rankine": "r",
    }

    f = temp_names.get(from_u)
    t = temp_names.get(to_u)
    if not f or not t:
        return None

    # Convert to Kelvin first
    if f == "c":
        k = value + 273.15
    elif f == "f":
        k = (value - 32) * 5 / 9 + 273.15
    elif f == "k":
        k = value
    elif f == "r":
        k = value * 5 / 9
    else:
        return None

    # Convert from Kelvin to target
    if t == "c":
        return k - 273.15
    elif t == "f":
        return (k - 273.15) * 9 / 5 + 32
    elif t == "k":
        return k
    elif t == "r":
        return k * 9 / 5
    return None


class UnitConverterTool(BaseTool):
    name = "unit_converter"
    description = (
        "Convert between physical units. Supports energy (eV, MeV, GeV, J, cal, kWh), "
        "mass (kg, g, amu, lb), length (m, km, nm, fm, angstrom, AU, light-year), "
        "time (s, min, hr, year), temperature (C, F, K), radioactivity (Bq, Ci), "
        "radiation dose (Sv, rem, Gy), and pressure (Pa, atm, bar, psi). "
        "Input format: '<value> <from_unit> to <to_unit>' (e.g., '13.6 eV to J')."
    )

    async def execute(self, query: str = "", **kwargs: Any) -> dict:
        """Parse and execute a unit conversion."""
        if not query.strip():
            return {"result": "No conversion query provided.", "raw": ""}

        # Parse: "<value> <from_unit> to <to_unit>"
        match = re.match(
            r"([\d.eE+\-]+)\s+(\S+)\s+(?:to|in|->|=>)\s+(\S+)",
            query.strip(), re.IGNORECASE,
        )
        if not match:
            return {
                "result": f"Could not parse: '{query}'. Use format: '<value> <from_unit> to <to_unit>'",
                "raw": "",
            }

        try:
            value = float(match.group(1))
        except ValueError:
            return {"result": f"Invalid number: {match.group(1)}", "raw": ""}

        from_unit = match.group(2)
        to_unit = match.group(3)

        # Try temperature first
        temp_result = _convert_temperature(value, from_unit, to_unit)
        if temp_result is not None:
            result_str = f"{value} {from_unit} = {temp_result:.6g} {to_unit}"
            return {"result": result_str, "raw": result_str}

        # Try standard conversions
        from_info = _find_unit(from_unit)
        to_info = _find_unit(to_unit)

        if from_info is None:
            return {"result": f"Unknown unit: {from_unit}", "raw": ""}
        if to_info is None:
            return {"result": f"Unknown unit: {to_unit}", "raw": ""}

        from_cat, _, from_factor = from_info
        to_cat, _, to_factor = to_info

        if from_cat != to_cat:
            return {
                "result": f"Cannot convert between {from_cat} ({from_unit}) and {to_cat} ({to_unit})",
                "raw": "",
            }

        # Convert: value * from_factor / to_factor
        result_value = value * from_factor / to_factor
        result_str = f"{value} {from_unit} = {result_value:.6g} {to_unit}"
        return {"result": result_str, "raw": result_str}
