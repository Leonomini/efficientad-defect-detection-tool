
def to_class_set(x) -> set[str]:
    # accepts "good" or ["good_clean", "good_noisy"] or set(...)
    if isinstance(x, str):
        return {x}
    return set(x)