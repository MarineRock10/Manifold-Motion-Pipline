from manifold_motion.dataio.seed_capability_catalog import FAMILIES, actor_split, classify


def test_catalog_has_30_unique_families() -> None:
    assert len(FAMILIES) == 30
    assert [family.family_id for family in FAMILIES] == list(range(30))
    assert len({family.name for family in FAMILIES}) == 30


def test_representative_seed_names_are_classified() -> None:
    cases = {
        "lateral_speed_step_ff_270_R_001__A359": "walk_lateral",
        "neutral_avoid_obstacle_bend_down_walk_ff_180_R_001__A542": "bend_duck_walk",
        "spider_crawl_R_001__A360": "spider_crawl",
        "inside_door_handle_left_side_open_walk_turn_close_R_001__A512": "door_interaction",
        "ladder_climbing_down_loop_R_003__A300": "ladder",
    }
    for name, expected in cases.items():
        assert expected in {family.name for family in classify(name)}


def test_actor_split_is_stable() -> None:
    assert actor_split("A359") == actor_split("A359")
    assert actor_split("A359") in {"train", "validation", "test"}