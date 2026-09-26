from openece.recipes.loader import load_recipe

def test_recipe_loads():
    r = load_recipe("examples/recipes/rc_lowpass.yaml")
    assert r["stimulus"]["type"] == "sine_sweep"
