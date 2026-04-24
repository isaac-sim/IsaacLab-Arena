"""RQA (Robot Question Answering) templates for task generation.

Each category tests a different manipulation competency.
Templates use placeholders that the LLM fills with scene-specific objects.

Expanded from RoboLab's 7 categories with more templates + articulated object support.
"""

# RQA templates organized by category
TASK_RQA = {
    # --- Object identification (short + long-horizon) ---
    "recognition": [
        "Pick <object> and <spatial verb> <fixture>",
        "Grab <object>",
        "Move <object> to the other side of the table",
        "Find <object> and place it at the center of the table",
        "Remove <object> from <fixture>",
        # Long-horizon
        "Pick <object A>, place on <fixture>, then pick <object B> and place to the right of it",
        "Move <object A> to <fixture>, then move <object B> to a different <fixture>",
    ],
    # --- Quantity handling (short + long-horizon) ---
    "counting": [
        "Put N <object> in <fixture>",
        "Take N <object> out from <fixture>",
        "Move exactly N <objects> onto <fixture>",
        "Gather N <objects> and group them together on <fixture>",
        # Long-horizon
        "Put N <objects> in <fixture A>, then put M <objects> in <fixture B>",
        "Pick all <objects> one by one and place each on <fixture>",
    ],
    # --- Spatial reasoning (short + long-horizon) ---
    "spatial": [
        "Put <object> <spatial verb> <object B>",
        "Move <object> to be <spatial verb> <object B>",
        "Swap the positions of <object A> and <object B>",
        "Place <object> between <object A> and <object B>",
        # Long-horizon
        "Place <object A> left of <object B>, then place <object C> right of <object B>",
        "Move <object A> in front of <object B>, then move <object C> behind <object B>",
    ],
    # --- Visual attribute recognition (short + long-horizon) ---
    "color": [
        "Put <color A> objects <spatial verb> <fixture>",
        "Group all <color> items together on <fixture>",
        "Separate <color A> and <color B> objects onto different sides",
        "Pick the <color> <object> and place it <spatial verb> <fixture>",
        # Long-horizon
        "Sort objects by color: <color A> items on <fixture A>, <color B> items on <fixture B>",
        "Pick all <color> objects and arrange them in a line on <fixture>",
    ],
    # --- Category/property understanding (short + long-horizon) ---
    "semantics": [
        "Pick <semantic characteristics> <spatial verb> <fixture>",
        "Put <semantic categories> <spatial verb> <fixture>",
        "Get the <larger/smaller> <object> <spatial verb> <fixture>",
        "Move all <category> items onto <fixture>",
        "Find the tallest object and move it <spatial verb> <fixture>",
        # Long-horizon
        "Separate food items from non-food items onto different sides of the table",
        "Sort objects by size: smallest on <fixture A>, largest on <fixture B>",
        "Group all fruits together on <fixture>, then group all tools together on another <fixture>",
    ],
    # --- Ordering and arrangement (short + long-horizon) ---
    "sorting": [
        "Stack <objects> <order>",
        "Put <objects> in a line",
        "Arrange <objects> from <attribute> to <attribute>",
        "Line up <objects> from smallest to largest",
        # Long-horizon
        "Stack <objects> with the largest on the bottom, then place <object> to the left of the stack",
        "Arrange a meal: place <plate> at center, <utensil> to the left, <food> on the plate",
        "Build a tower of 3 objects, then place a small object on top",
    ],
    # --- Multi-object logic (short + long-horizon) ---
    "conjunction": [
        "Put <object A> and <object B> <spatial verb> <fixture>",
        "Place <object A> or <object B> <spatial verb> <fixture>",
        "Make sure <object A> then <object B> <spatial verb> <fixture>",
        "Pick <object A> and <object B> <spatial verb> <fixture>",
        # Long-horizon
        "First move <object A> to <fixture>, then move <object B> to the right of it, then <object C> behind them",
        "Pick <object A>, <object B>, and <object C>, place all on <fixture>",
        "Pick <object A> and place left of <fixture>, then <object B> right of <fixture>, then <object C> on top of <fixture>",
    ],
    # --- Articulated object interaction (mixed with pick-and-place) ---
    "affordance": [
        "Open the <appliance> door",
        "Close the <appliance> door",
        "Put <object> inside the <appliance> and close it",
        "Take <object> out of the <appliance>",
        "Open the <appliance>, place <object> inside, then close it",
        "Pick <object> from the table and put it inside the <appliance>",
        "Take <object> out of the <appliance> and place it on <fixture>",
        "Move <object> from <fixture> into the <appliance>",
    ],
    # --- Multi-step with articulated + pick-and-place ---
    "sequential": [
        "Open the <appliance>, put <object> inside, close it, then place <object B> on <fixture>",
        "Pick <object> from <fixture>, open <appliance>, place inside",
        "Take <object> out of <appliance> and place on <fixture>",
        "Open <appliance>, remove all items, place them on <fixture>, then close <appliance>",
        "Pick <object A> and place it on <fixture>, then put <object B> inside the <appliance>",
        "Take <object> out of the <appliance>, place it <spatial verb> <object B>",
        "Pick <object> from the table, open <appliance>, put it inside, close, then pick <object B> and place on <fixture>",
    ],
}

# Spatial verbs the LLM can use. "next to" was removed because it's
# directionless — it maps to NextTo(side=+Y) in predicate_to_relation.py,
# which breaks when the LLM uses multiple `next_to` with the same target
# (they all collapse onto the +Y side → solver runaway or overlap). Use the
# directional alternatives (left of / right of / in front of / behind).
SPATIAL_VERBS = [
    "left of", "right of", "in front of", "behind",
    "on top of", "inside of", "outside of",
    "between", "center of",
]

# Categories that require articulated objects in the scene
ARTICULATED_CATEGORIES = {"affordance", "sequential"}


def get_task_rqa(categories=None):
    """Format RQA templates for LLM prompt.

    Args:
        categories: List of category names. If None, uses all.

    Returns:
        Formatted string with category and template tags.
    """
    tasks_dict = TASK_RQA
    if categories is not None:
        tasks_dict = {k: v for k, v in TASK_RQA.items() if k in categories}

    lines = []
    for category, rqas in tasks_dict.items():
        lines.append(f"<task_category> {category}</task_category>")
        for rqa in rqas:
            lines.append(f"    <rqa>{rqa}</rqa>")
    return "\n".join(lines)


def get_categories_for_scene(has_articulated: bool = False) -> list[str]:
    """Get applicable RQA categories for a scene.

    Args:
        has_articulated: Whether the scene contains articulated objects.

    Returns:
        List of category names applicable to this scene.
    """
    categories = [c for c in TASK_RQA if c not in ARTICULATED_CATEGORIES]
    if has_articulated:
        categories.extend(ARTICULATED_CATEGORIES)
    return categories
