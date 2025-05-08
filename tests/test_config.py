def test_import():
    from fluxy.config import set_print_settings


def test_initialize_settings():
    from fluxy.config import set_print_settings

    set_print_settings()


def test_model_colors():

    # Legacy function
    # This function should be changed, as the color model should be somehow cleaner
    from fluxy.config import set_model_colors

    model_colors = set_model_colors(
        models=["intem_name_edgar", "elris_name_edgar", "rhime_name_edgar"]
    )


def test_get_next_color():
    from fluxy.config import get_default_colors

    colors = get_default_colors()
    assert isinstance(colors, list)
