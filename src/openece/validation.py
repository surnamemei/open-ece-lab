def check_range(name: str, value: float, minimum: float | None = None, maximum: float | None = None):
    passed = True
    if minimum is not None: passed &= value >= minimum
    if maximum is not None: passed &= value <= maximum
    return {"name": name, "value": float(value), "min": minimum, "max": maximum, "passed": bool(passed)}
