"""Landsat compositing policy (no network): sensor choice, scene ranking, scaling."""
import datetime as dt
from types import SimpleNamespace

import numpy as np

from app import config, imagery


def _item(platform, date, cloud):
    return SimpleNamespace(
        properties={"platform": platform, "eo:cloud_cover": cloud},
        datetime=dt.datetime.fromisoformat(date),
    )


def test_platforms_by_era():
    assert imagery.platforms_for(1986) == imagery.LEGACY_PLATFORMS
    assert imagery.platforms_for(2012) == imagery.LEGACY_PLATFORMS
    assert imagery.platforms_for(2013) == imagery.OLI_PLATFORMS


def test_striped_landsat7_ranks_behind_landsat5():
    items = [
        _item("landsat-7", "2008-06-01", 1),   # SLC-off: last despite being clearest
        _item("landsat-5", "2008-07-01", 30),
        _item("landsat-7", "2001-06-01", 5),   # pre-failure L7 is fine
    ]
    picked = imagery.select_scenes(items)
    assert [it.properties["eo:cloud_cover"] for it in picked] == [5, 30, 1]


def test_slc_off_only_year_takes_extra_scenes(monkeypatch):
    monkeypatch.setattr(config, "LANDSAT_MAX_SCENES", 2)
    items = [_item("landsat-7", f"2012-0{m}-01", m) for m in range(1, 8)]
    assert len(imagery.select_scenes(items)) == 2 * config.LANDSAT_SLC_OFF_SCENE_FACTOR
    mixed = items + [_item("landsat-5", "2012-01-01", 50)]
    assert len(imagery.select_scenes(mixed)) == 2


def test_reflectance_scaling_and_optional_harmonisation(monkeypatch):
    dn = np.array([20000], dtype=np.uint16)
    # Off by default: C2 surface reflectance is used as-is for every sensor.
    assert config.LANDSAT_HARMONIZE is False
    assert np.isclose(imagery.to_reflectance(dn, "nir08", legacy=True)[0], 20000 * 2.75e-5 - 0.2)

    monkeypatch.setattr(config, "LANDSAT_HARMONIZE", True)
    oli = imagery.to_reflectance(dn, "nir08", legacy=False)
    assert np.isclose(oli[0], 20000 * 2.75e-5 - 0.2)
    tm = imagery.to_reflectance(dn, "nir08", legacy=True)
    slope, intercept = config.LANDSAT_ETM_TO_OLI["nir08"]
    assert np.isclose(tm[0], oli[0] * slope + intercept)
