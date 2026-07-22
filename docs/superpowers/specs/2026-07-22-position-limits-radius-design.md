# Radial PositionLimits Design

## Goal

Extend the generic `PositionLimits` unary relation with world-XY annular bounds. Scenes can constrain an object's placement origin to lie inside a disk or between two radii without adding a CAP- or task-specific relation.

## Public API

`PositionLimits` gains these optional parameters:

- `center_x: float | None`
- `center_y: float | None`
- `radius_min: float | None`
- `radius_max: float | None`

YAML uses the existing relation kind:

```yaml
- kind: position_limits
  subject: tool
  params:
    center_x: 0.45
    center_y: 0.00
    radius_min: 0.10
    radius_max: 0.35
```

The radial bounds apply to the object's placement origin in world XY. They compose with existing X/Y/Z bounds by intersection. No existing YAML changes meaning.

## Validation and Loss

At least one existing axis bound or radial bound is required. A radial bound requires both center coordinates. Radii are non-negative; if both are present, `radius_min < radius_max`.

The loss is zero for radii inside the permitted annulus. Below `radius_min` it grows linearly with the radial deficit; above `radius_max` it grows linearly with the radial excess. The existing `relation_loss_weight` scales this radial loss exactly as it does axis-limit losses.

The radial constraint is included in `PositionLimitsLossStrategy` and registered through the existing `RelationSolverParams` strategy table. This preserves the current graph-schema and relation registry path.

## Scope and Tests

This change touches only generic Arena placement code. It does not change collision policy, physics filters, task code, or Cap/GaP integration.

Tests cover constructor validation, zero/positive lower- and upper-radius losses, batched tensors, loss-weight scaling, composition with an axis bound, solver convergence, and YAML graph construction. Existing rectangular PositionLimits tests remain unchanged.
