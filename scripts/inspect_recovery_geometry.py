#!/usr/bin/env python3
"""Derive conservative recovery bounds from the unperturbed nominal MJCF.

From the repository root::

    PYTHONPATH=src python scripts/inspect_recovery_geometry.py
    PYTHONPATH=src python scripts/inspect_recovery_geometry.py --output /tmp/geometry.json

The script prints a summary and optionally saves the complete audit, including
suggested configuration values. It never updates recovery configuration files.
No perturbed plant, fitted parameters, or validation measurements are used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import mujoco
import numpy as np
import robot_model


MARGIN_FACTOR = 1.10
ROBOT_NAMES = ("rrr", "franka")
DEFINITIONS = {
    "frame": "each link's own coordinate frame (the MDH/MuJoCo body frame)",
    "units": {"length": "m", "mass": "kg", "inertia": "kg*m^2"},
    "center": "unperturbed nominal model.body_ipos (nominal COM)",
    "geometry_scope": "all visual and collision geoms attached directly to each link; excludes other links and sites",
    "mesh_transform": "compiled mesh_vert @ rotation(geom_quat).T + geom_pos",
    "geometry_radius": "largest distance from center to any mesh vertex or capsule surface",
    "nominal_rms_radius": "sqrt(trace(I_com_nominal) / (2 * m_nominal))",
    "envelope_radius": "1.10 * max(geometry_radius, nominal_rms_radius)",
    "com_box": "union of geom AABBs, with every half-width multiplied by 1.10 about the box midpoint",
    "constraint": "trace(I_com)/2 + m*||COM-center||^2 <= m*envelope_radius^2",
    "interpretation": "conservative geometry/nominal-inertia envelope; a necessary second-moment bound, not exact CAD support or a unique physical realization",
    "margin_selection": "fixed 10% length margin, chosen without plant truth, fitting results, or validation errors",
}


def _rotation(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def _source_location(path):
    """Describe the source without embedding a developer's machine path."""
    resolved = path.resolve()
    repository = Path(__file__).resolve().parents[1]
    if resolved.is_relative_to(repository):
        return resolved.relative_to(repository).as_posix(), "repository"
    package = Path(robot_model.__file__).resolve().parent
    if resolved.is_relative_to(package):
        relative = Path("robot_model") / resolved.relative_to(package)
        return relative.as_posix(), "installed package"
    raise ValueError(f"Nominal model is outside this repository and robot_model package: {path.name}")


def _geom_bounds(model, geom, center):
    rotation = _rotation(model.geom_quat[geom])
    position = model.geom_pos[geom]
    kind = model.geom_type[geom]
    details = {
        "geom": int(geom),
        "geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
        "group": int(model.geom_group[geom]),
        "position": position.tolist(),
        "quaternion": model.geom_quat[geom].tolist(),
    }
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        if count < 1:
            raise ValueError(f"Mesh geom {geom} has no vertices")
        vertices = np.asarray(model.mesh_vert[start:start + count], dtype=float)
        # MuJoCo's compilation re-centers/re-orients meshes; the compiled geom
        # transform already includes that correction. Do not apply mesh_pos or
        # mesh_quat a second time. Mesh scaling is already in mesh_vert.
        points = vertices @ rotation.T + position
        low, high = points.min(axis=0), points.max(axis=0)
        radius = np.linalg.norm(points - center, axis=1).max()
        details.update(
            type="mesh",
            mesh=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh),
            vertices=int(count),
        )
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        sphere_radius, half_length = model.geom_size[geom, :2]
        endpoints = position + np.array([-1, 1])[:, None] * half_length * rotation[:, 2]
        low = endpoints.min(axis=0) - sphere_radius
        high = endpoints.max(axis=0) + sphere_radius
        radius = np.linalg.norm(endpoints - center, axis=1).max() + sphere_radius
        details.update(
            type="capsule", endpoints=endpoints.tolist(),
            capsule_radius=float(sphere_radius),
        )
    else:
        # Fail explicitly if a future model adds another primitive. Ignoring
        # an unsupported geom would make the claimed envelope nonconservative.
        raise ValueError(f"Unsupported geometry type {int(kind)} for geom {geom}")
    details.update(radius=float(radius), aabb_lower=low.tolist(), aabb_upper=high.tolist())
    return low, high, float(radius), details


def inspect_robot(name):
    # Paths and body lists come from the current experiment setup, not copied
    # constants that could diverge when the robot's nominal model changes.
    setup = importlib.import_module(f"robot_model.experiments.{name}").setup()
    path = Path(setup.xml_path)
    source, source_root = _source_location(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    result = {
        "robot": name, "source": source, "source_root": source_root,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "setup_name": setup.name, "mujoco_version": mujoco.__version__,
        "margin_factor": MARGIN_FACTOR, "definitions": DEFINITIONS,
        "links": [],
    }
    for name in setup.body_names:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            raise ValueError(f"Nominal MJCF has no body named {name}")
        center = model.body_ipos[body].copy()
        mass = float(model.body_mass[body])
        if mass <= 0:
            raise ValueError(f"Body {name} must have positive nominal mass")
        geoms = np.flatnonzero(model.geom_bodyid == body)
        if not len(geoms):
            raise ValueError(f"Body {name} has no geometry for an envelope")
        bounds = [_geom_bounds(model, geom, center) for geom in geoms]
        low = np.min([item[0] for item in bounds], axis=0)
        high = np.max([item[1] for item in bounds], axis=0)
        midpoint, half_width = (high + low) / 2, (high - low) / 2
        geom_radius = max(item[2] for item in bounds)
        trace = float(model.body_inertia[body].sum())
        nominal_radius = float(np.sqrt(trace / (2 * mass)))
        radius = MARGIN_FACTOR * max(geom_radius, nominal_radius)
        result["links"].append({
            "name": name, "mass": mass, "center": center.tolist(),
            "geometry_aabb_lower": low.tolist(), "geometry_aabb_upper": high.tolist(),
            "com_box_lower": (midpoint - MARGIN_FACTOR * half_width).tolist(),
            "com_box_upper": (midpoint + MARGIN_FACTOR * half_width).tolist(),
            "geometry_radius": geom_radius,
            "nominal_trace_inertia": trace, "nominal_rms_radius": nominal_radius,
            "nominal_fits_geometry_sphere": nominal_radius <= geom_radius,
            "nominal_inertia_to_geometry_limit_ratio": trace / (2 * mass * geom_radius**2),
            "envelope_radius": radius,
            "nominal_envelope_margin_kg_m2": mass * radius**2 - trace / 2,
            "provenance": [item[3] for item in bounds],
        })
    result["incompatible_nominal_links"] = [
        link["name"] for link in result["links"]
        if not link["nominal_fits_geometry_sphere"]
    ]
    result["notes"] = [
        "Geometry and nominal inertia are separate model inputs; visual/collision meshes are not a calibrated mass distribution.",
        "Incompatible links require the nominal-inertia term in the envelope. This is not an exact CAD constraint.",
    ]
    if result["incompatible_nominal_links"]:
        links = ", ".join(result["incompatible_nominal_links"])
        result["notes"].append(
            f"{result['robot']} {links}: even the nominal inertia violates the "
            "geometry-only sphere bound; the reported envelope explicitly includes nominal inertia."
        )
    result["suggested_config"] = {
        "geometry_center": [link["center"] for link in result["links"]],
        "geometry_radius": [link["envelope_radius"] for link in result["links"]],
        "com_lower": [link["com_box_lower"] for link in result["links"]],
        "com_upper": [link["com_box_upper"] for link in result["links"]],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("all", *ROBOT_NAMES), default="all")
    parser.add_argument("--output", type=Path, help="Save the complete audit as JSON (does not update configs)")
    args = parser.parse_args()
    names = ROBOT_NAMES if args.robot == "all" else (args.robot,)
    result = {name: inspect_robot(name) for name in names}
    print("Nominal geometry/inertia audit: fixed 1.10 length margin; units are m, kg, kg*m^2.")
    for name, robot in result.items():
        print(f"\n{name}: {robot['source']} ({robot['source_root']})")
        for link in robot["links"]:
            compatible = "compatible" if link["nominal_fits_geometry_sphere"] else "INCOMPATIBLE"
            print(
                f"  {link['name']}: geometry R={link['geometry_radius']:.9f}; "
                f"nominal RMS R={link['nominal_rms_radius']:.9f}; "
                f"envelope R={link['envelope_radius']:.9f}; geometry-only {compatible}"
            )
        if robot["incompatible_nominal_links"]:
            links = ", ".join(robot["incompatible_nominal_links"])
            print(f"  {links}: nominal inertia exceeds the geometry sphere bound.")
            print("  The proposed bound is a geometry/nominal-inertia envelope, not exact CAD support.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nFull audit saved to {args.output}")


if __name__ == "__main__":
    main()
