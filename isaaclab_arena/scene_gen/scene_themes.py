"""Scene theme templates for diverse automated scene generation.

Themes are designed around Arena's full 719-object catalog:
- 150 containers (bins, boxes, cases, pails, jugs, bottles)
- 59 tools (hammers, spatulas, tongs, whisks, spoons)
- 51 food items (fruits, veggies, canned goods, snacks)
- 356 lightwheel objects (kitchen items, groceries, tableware)
- 22 articulated objects (microwaves, toasters, coffee machines, stand mixer)
"""

import random
from typing import List, Dict

from isaaclab_arena.scene_gen.arena_asset_manager import RACK_OBJECTS

# Scene themes organized by manipulation task types
MANIPULATION_THEMES = {
    "pick_and_place": [
        "scattered tools on a workbench",
        "groceries on a kitchen counter",
        "office supplies on a desk",
        "toys spread across a play table",
        "kitchen utensils on a counter",
        "hardware parts on a workshop table",
        "art supplies on a craft table",
        "electronic components on a work surface",
        "gardening tools on a potting bench",
        "cleaning supplies on a utility shelf",
    ],
    "sorting_and_organizing": [
        "mixed fruits and vegetables to be sorted",
        "assorted tools that need organizing",
        "various containers and boxes to arrange",
        "mixed office items to categorize",
        "different colored objects to sort",
        "kitchen items to organize by type",
        "hardware fasteners to separate",
        "books and magazines to arrange",
        "craft materials to sort",
        "recycling items to separate",
    ],
    "stacking_and_nesting": [
        "stackable containers and bowls",
        "cups and dishes to stack",
        "boxes of various sizes to nest",
        "food storage containers to organize",
        "plates and bowls to stack neatly",
        "bins and baskets to nest together",
        "cooking pots to stack",
        "plastic containers to organize",
        "lunch boxes to stack",
        "storage boxes to nest",
    ],
    "container_manipulation": [
        "bowl filled with assorted fruits",
        "basket containing vegetables",
        "box with various tools inside",
        "bin filled with toys",
        "container with kitchen items",
        "tray with small objects",
        "bucket with cleaning supplies",
        "jar with craft supplies",
        "crate with hardware parts",
        "bag with groceries",
    ],
    "meal_preparation": [
        "ingredients for a simple salad",
        "breakfast items on a table",
        "sandwich making ingredients",
        "fruit cutting setup",
        "drink preparation station",
        "snack arrangement setup",
        "baking ingredients layout",
        "cooking ingredients on counter",
        "food prep with cutting board",
        "dinner ingredients setup",
    ],
    "tool_usage": [
        "hand tools arranged for a task",
        "kitchen tools for cooking",
        "cleaning tools and supplies",
        "gardening tools and materials",
        "art tools and brushes",
        "repair tools on a workbench",
        "crafting tools and materials",
        "measuring tools and items",
        "cutting tools and materials",
        "assembly tools and parts",
    ],
    "dense_clutter": [
        "crowded workspace with many items",
        "cluttered kitchen counter",
        "busy craft table with supplies",
        "packed toolbox contents",
        "full desk with office items",
        "crowded shelving contents",
        "busy workshop surface",
        "full pantry shelf items",
        "cluttered art station",
        "packed storage area items",
    ],
    # --- New themes for Arena's extended object library ---
    "kitchen_appliances": [
        "a microwave with food items ready to heat",
        "a toaster with bread and spreads nearby",
        "a coffee machine with mugs and supplies around it",
        "a stand mixer with baking ingredients",
        "kitchen counter with a microwave, plates, and food",
        "breakfast station with toaster, cereal, and milk",
        "coffee corner with machine, mugs, and sugar",
        "cooking station with mixer, bowls, and ingredients",
        "appliance-heavy kitchen counter with food prep items",
        "morning routine setup with toaster, kettle, and plates",
    ],
    "grocery_unpacking": [
        "grocery bags being unpacked onto a counter",
        "fresh produce spread out from shopping",
        "canned goods and bottled items from a grocery run",
        "assorted beverages and snacks from the store",
        "dairy products and fresh items to put away",
        "mixed grocery items with containers to sort into",
        "weekly groceries with bins for organizing",
        "fresh fruits and vegetables with storage containers",
        "pantry restocking with cans, boxes, and bottles",
        "shopping haul with bags, bottles, and canned food",
    ],
    "industrial_workspace": [
        "plastic containers and pails on a work surface",
        "industrial bottles and jerry cans arranged on a table",
        "storage cases and bins for parts organization",
        "utility jugs and containers for a workshop",
        "packing station with boxes and containers",
        "shipping area with cases and packing materials",
        "warehouse sorting station with bins and boxes",
        "quality control station with containers and tools",
        "lab bench with bottles, containers, and instruments",
        "supply room table with assorted storage containers",
    ],
    "tableware_setup": [
        "dinner table setting with plates, bowls, and utensils",
        "place setting with mugs, plates, and cutlery",
        "tea party setup with cups, saucers, and snacks",
        "breakfast table with bowls, spoons, and cereal",
        "lunch setup with plates, forks, and food items",
        "picnic spread with plates, cups, and fruit",
        "buffet station with serving bowls and utensils",
        "dining setup with plates stacked and silverware",
        "dessert station with bowls, spoons, and sweets",
        "snack station with plates, cups, and treats",
    ],
    "cooking_prep": [
        "cutting board with vegetables and a knife",
        "baking prep with bowls, spatulas, and ingredients",
        "salad making station with bowl, tongs, and produce",
        "stir fry prep with pan, spatula, and vegetables",
        "soup making setup with pot, ladle, and ingredients",
        "seasoning station with bottles, shakers, and spoons",
        "marinade prep with bottles, bowls, and meat",
        "pizza prep with cutting board, roller, and toppings",
        "smoothie station with fruits, cups, and containers",
        "meal prep with containers, knife, and ingredients",
    ],
}

# Themes that should include articulated objects (microwave, toaster, coffee machine, mixer)
ARTICULATED_THEMES = {"kitchen_appliances"}

# Rack-based scene templates (15% of scenes)
RACK_SCENE_TEMPLATES = [
    "a {rack} on the table with {theme}",
    "{theme} organized on a {rack}",
    "a workspace with a {rack} holding {theme}",
    "{theme} displayed on a {rack} with additional items nearby",
    "a {rack} on the counter with {theme} around it",
]

# Simple theme lists for easy access
SCENE_THEMES = {
    "easy": MANIPULATION_THEMES["pick_and_place"],
    "medium": MANIPULATION_THEMES["sorting_and_organizing"],
    "hard": MANIPULATION_THEMES["dense_clutter"],
}


def generate_scene_prompts(
    num_easy: int = 15, num_medium: int = 70, num_hard: int = 15,
    appliance_ratio: float = 0.15,
) -> List[Dict]:
    """Generate diverse scene prompts for batch generation.

    Not limited to 100 — pass any numbers. Default 100 matches RoboLab-80 scale.

    Args:
        num_easy: Number of easy scenes (2-5 objects)
        num_medium: Number of medium scenes (7-10 objects)
        num_hard: Number of hard scenes (12-20 objects)
        appliance_ratio: Fraction of scenes that should include articulated objects.

    Returns:
        List of scene configurations with prompts and parameters
    """
    scenes = []
    total_scenes = num_easy + num_medium + num_hard
    num_rack_scenes = int(total_scenes * 0.10)  # 10% with racks
    num_appliance_scenes = int(total_scenes * appliance_ratio)

    # Collect all themes
    all_themes = []
    for category, themes in MANIPULATION_THEMES.items():
        all_themes.extend([(category, theme) for theme in themes])

    random.shuffle(all_themes)

    # Generate easy scenes
    for i in range(num_easy):
        category, theme = all_themes[i % len(all_themes)]
        use_rack = i < (num_rack_scenes * num_easy // total_scenes)

        if use_rack:
            rack = random.choice(list(RACK_OBJECTS))
            template = random.choice(RACK_SCENE_TEMPLATES)
            prompt = template.format(rack=rack, theme=theme)
        else:
            prompt = theme

        scenes.append(
            {
                "difficulty": "easy",
                "max_objects": 5,
                "prompt": prompt,
                "category": category,
                "has_rack": use_rack,
                "name": f"easy_{i+1:03d}",
            }
        )

    # Generate medium scenes
    for i in range(num_medium):
        category, theme = all_themes[(num_easy + i) % len(all_themes)]
        use_rack = i < (num_rack_scenes * num_medium // total_scenes)

        if use_rack:
            rack = random.choice(list(RACK_OBJECTS))
            template = random.choice(RACK_SCENE_TEMPLATES)
            prompt = template.format(rack=rack, theme=theme)
        else:
            prompt = theme

        # Medium scenes: 5-10 objects
        max_obj = random.randint(7, 10)

        scenes.append(
            {
                "difficulty": "medium",
                "max_objects": max_obj,
                "prompt": prompt,
                "category": category,
                "has_rack": use_rack,
                "name": f"medium_{i+1:03d}",
            }
        )

    # Generate hard scenes
    for i in range(num_hard):
        # Hard scenes should focus on dense_clutter and complex tasks
        if i < len(MANIPULATION_THEMES["dense_clutter"]):
            category = "dense_clutter"
            theme = MANIPULATION_THEMES["dense_clutter"][i]
        else:
            category, theme = all_themes[(num_easy + num_medium + i) % len(all_themes)]

        use_rack = i < (num_rack_scenes * num_hard // total_scenes)

        if use_rack:
            rack = random.choice(list(RACK_OBJECTS))
            template = random.choice(RACK_SCENE_TEMPLATES)
            prompt = template.format(rack=rack, theme=theme)
        else:
            prompt = theme

        # Hard scenes: 11-20 objects
        max_obj = random.randint(12, 20)

        scenes.append(
            {
                "difficulty": "hard",
                "max_objects": max_obj,
                "prompt": prompt,
                "category": category,
                "has_rack": use_rack,
                "name": f"hard_{i+1:03d}",
            }
        )

    # Inject appliance scenes across all difficulties
    if num_appliance_scenes > 0:
        appliance_themes = MANIPULATION_THEMES.get("kitchen_appliances", [])
        if appliance_themes:
            # Spread evenly across difficulties
            indices = list(range(len(scenes)))
            random.shuffle(indices)
            injected = 0
            for idx in indices:
                if injected >= num_appliance_scenes:
                    break
                if scenes[idx]["category"] != "kitchen_appliances":
                    scenes[idx]["prompt"] = appliance_themes[injected % len(appliance_themes)]
                    scenes[idx]["category"] = "kitchen_appliances"
                    scenes[idx]["has_articulated"] = True
                    injected += 1

    return scenes


def print_scene_summary(scenes: List[Dict]):
    """Print a summary of generated scenes."""
    print(f"\nTotal scenes: {len(scenes)}")
    print(f"  Easy (<=5 obj): {sum(1 for s in scenes if s['difficulty'] == 'easy')}")
    print(f"  Medium (5-10): {sum(1 for s in scenes if s['difficulty'] == 'medium')}")
    print(f"  Hard (>10): {sum(1 for s in scenes if s['difficulty'] == 'hard')}")
    print(f"\nScenes with racks: {sum(1 for s in scenes if s['has_rack'])}")
    print(f"Scenes with articulated: {sum(1 for s in scenes if s.get('has_articulated'))}")

    print(f"\nCategories:")
    categories = {}
    for scene in scenes:
        cat = scene["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    unique_prompts = set(s["prompt"] for s in scenes)
    print(f"\nUnique prompts: {len(unique_prompts)} / {len(scenes)} scenes")


if __name__ == "__main__":
    scenes = generate_scene_prompts(num_easy=50)
    print_scene_summary(scenes)

    # Show examples
    print("\nExample scenes:")
    for difficulty in ["easy", "medium", "hard"]:
        examples = [s for s in scenes if s["difficulty"] == difficulty][:3]
        print(f"\n{difficulty.upper()}:")
        for ex in examples:
            rack_marker = " [+RACK]" if ex["has_rack"] else ""
            artic_marker = " [+ARTIC]" if ex.get("has_articulated") else ""
            print(
                f"  {ex['name']}: \"{ex['prompt']}\" (max {ex['max_objects']} obj){rack_marker}{artic_marker}"
            )
