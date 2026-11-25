# G1 spatial-memory task catalog

The release selection contains three pilot controls and 27 authored challenges. Use the full registry ID shown for each task. The original pilot results and the earlier shortcut studies are in [LOCO_PLAN.md](LOCO_PLAN.md). The previous planning table is historical; this catalog defines the selected environments.

Every scenario uses the G1 head camera. Enable cameras and select the `mapped` observation variant to give a policy the registered depth cloud. Fusion starts on the first post-reset control step so an image from the previous layout cannot enter the new map. The partition control also needs RGB to identify its equal-shaped colored plaques. A separately registered depth-readable silhouette experiment is documented in the [suite README](README.md).

The 22 workcells use the same inspection procedure. Inspect the named surfaces, return to the common staging area and look away for 3.5 seconds. The public instruction signal then enables the first approach. Seventeen tasks score a backward approach to a remembered alignment, including one that also scores the initial palm height. Three score the initial walking direction. The sled scores the first push direction, and the jig scores workpiece orientation at the first lift. Reobserving a target before commitment or committing at the wrong alignment fails the episode. Current vision is available for local manipulation and live obstacle avoidance after commitment. The sampled workcell location, orientation and handedness change at reset. A motion clock cannot identify the sampled static geometry.

The task-specific conditions below use actual body or object poses, palm contacts, support contacts, consecutive holds, or measured fixture motion as applicable. The 22 workcells expose the normal G1 action space for external policies. Scripted oracle and blind plans are provided only for the controls and the five carried-load challenges. These authoring checks do not establish learned success or the necessity of a particular representation.

## Pilot controls

| Task and registry ID | Scored operation | Geometry to remember | Real-life example |
| --- | --- | --- | --- |
| **1. Reverse through a gap**<br>`spatial_memory_reverse_gap` | Survey the opening, lift the carton, then back through the remembered gap and deliver. | The lateral position of the barrier opening behind the robot at pickup. | A mover carries a carton backward through a storeroom doorway. |
| **2. Restock the empty shelf bay**<br>`spatial_memory_shelf_bay` | Inspect the stocked bays, return to pickup and put the carton in the previously empty bay. | Which of four shelf interiors was empty before the robot faced the loading table. | A grocery stocker returns from a pallet with a case for the one vacant shelf bay. |
| **3. Find the requested room**<br>`spatial_memory_partition_delivery` | Inspect four room markers. After pickup, deliver to the room named in the request. | The entrance associated with the requested plaque color. The original control needs RGB for color identity. | A hospital porter learns the marked room entrances before collecting a delivery. |

## Carry and route planning

| Task and registry ID | Scored operation | Geometry to remember | Real-life example |
| --- | --- | --- | --- |
| **4. Back out around a rack end**<br>`spatial_memory_dogleg_exit` | Collect the carton and retreat far enough to clear the rack end before crossing the divider. | The distance to the gap between two rack ends and the direction of the exit around the divider. | A stockroom worker backs a bulky case out of a dead-end aisle and around a shelving unit. |
| **5. Turn inside a pocket**<br>`spatial_memory_turning_pocket` | Use the deep recess to turn the carried load before continuing to dispatch. | Which recess has room for the carried carton to turn. | A mover uses a hallway alcove to turn a large box through a corner. |
| **6. Withdraw past staggered shelf lips**<br>`spatial_memory_staggered_lips` | Clear two projecting shelf lips on the correct sides while withdrawing the carton. | The open side of each shelf lip at two different withdrawal depths. | An archive worker eases a document box past two offset dividers. |
| **7. Combine two views of an exit**<br>`spatial_memory_two_view_exit` | Inspect two separated gates and carry through the lane that is open at both. | The intersection of the lanes left open by two gates inspected from separate viewpoints. | A warehouse operator checks two rows of temporary barriers before moving a load. |
| **8. Replan through a column grid**<br>`spatial_memory_column_grid` | After the request, pass the required inspection points through the surveyed grid and release at the dock within the route budget. | Which openings connect the pickup, two late-requested inspection points and the receiving dock. | A fulfillment worker routes a tote past two inspection stations to an assigned dispatch dock. |

## Inspection and locomotion

| Task and registry ID | Scored operation | Geometry to remember | Real-life example |
| --- | --- | --- | --- |
| **9. Inspect a trench and reverse out**<br>`spatial_memory_service_trench` | Walk headfirst to the end of the narrow service trench. Back out to the mouth facing the end. | The mouth, depth and heading of the trench that the robot must leave in reverse. | A technician backs out of a server aisle after inspecting a rear connector. |
| **10. Stop short of a dock edge**<br>`spatial_memory_dock_edge` | Back toward an inspected dock sill, stop at the standoff line facing away, then walk clear. | The location of the sill and the safe standoff distance in front of it. | A dock spotter backs up to a loading-bay edge while watching the approaching vehicle. |
| **11. Sidestep between stacks**<br>`spatial_memory_sideways_stacks` | Face the shelving and sidestep between the stacks, maintaining that heading at three stations. | The centerline between the high and low stacks and which way to face while sidestepping. | An archivist shuffles sideways between compact shelves while reading the spines. |
| **12. Duck under a low header**<br>`spatial_memory_low_header` | Lower the pelvis beneath the inspected lintel and emerge on its far side. | The lateral center and height of the lintel relative to the staging position. | A maintenance worker ducks under ductwork in a mechanical room. |
| **13. Walk around a tank**<br>`spatial_memory_tank_perimeter` | Inspect all four faces of an enclosed tank in order and return to the first face. | The initial walking direction toward the tank and the clearance behind it. | A plant operator checks a vessel from each side before signing off a walkaround. |
| **14. Recover a survey sightline**<br>`spatial_memory_sightline` | Inspect a pair of survey posts, then occupy their extended sightline and face along it. | The line connecting two independently offset survey posts, extended back toward the instrument station. | A surveyor reoccupies a line of stakes after walking back to the instrument station. |

## Hands and object handling

| Task and registry ID | Scored operation | Geometry to remember | Real-life example |
| --- | --- | --- | --- |
| **15. Reach into a recessed control**<br>`spatial_memory_recessed_reset` | Preset the left palm to the surveyed pad height, touch the recessed control, then withdraw the hand. | Which of four mounting heights sets the palm height before the pad returns to view. | A technician reaches into a recessed reset station under a machine guard. |
| **16. Engage a two-hand interlock**<br>`spatial_memory_two_hand_interlock` | Hold both palms at the two separated interlock pads simultaneously. | The center of the two pads and their separation across the console. | A press operator uses both hands on separated controls to enable a cycle. |
| **17. Trace a stepped seam**<br>`spatial_memory_seam_trace` | Follow a stepped seam with the right palm, passing the start, bend and raised end in order. | The low end, raised end and handedness of the step in the seam. | A maintenance inspector follows a weld bead around a step in a metal enclosure. |
| **18. Reach behind a flange**<br>`spatial_memory_blind_return_hook` | Guide a palm around the free edge of a screen, behind its flange and back out. | The free edge of the screen and the pad concealed behind it. | A mechanic reaches around a splash shield to find a concealed inspection point. |
| **19. Park a floor sled**<br>`spatial_memory_floor_sled` | Set the floor sled moving toward the remembered bumper, park it without lifting it, then step clear. | The bearing to the parking bumper when the sled first moves at staging. | A warehouse worker nudges a low parts dolly into its marked parking stop. |
| **20. Slide a tray through a hatch**<br>`spatial_memory_counter_slide` | Slide a tray laterally under a low pass-through cover and leave it at the outlet. | Which side of the pass-through cover is the inlet and how the tray clears the outlet. | A canteen worker slides a tray along a counter through a low serving hatch. |
| **21. Orient a workpiece before lifting**<br>`spatial_memory_jig_rotation` | Align a long workpiece to the remembered jig before lifting it from staging, then settle it beside the locating fence. | The remote fence axis that determines the workpiece orientation at its first lift. | A machinist aligns a long part to a remote fixture before moving it off the staging bench. |
| **22. Extract a tote before lifting**<br>`spatial_memory_tote_extraction` | Draw a tote out from under a shelf lip. Bring it beyond the apron before raising it. | The opening beneath the shelf lip and the distance the tote must travel before it can rise. | A stockroom worker pulls a parts bin straight out before lifting it past the shelf above. |
| **23. Weigh and transfer a parcel**<br>`spatial_memory_scale_transfer` | Set the case down on the scale, release it for a reading, then remove it to the outbound pad. | The weighing platform and the outbound pad on opposite parts of the workbench. | A parcel clerk weighs a box before moving it to the dispatch counter. |
| **24. Lift a case over a sill**<br>`spatial_memory_raised_sill` | Lift the case above a raised sill, carry it across the divider and lower it onto the far support. | Which side holds the case, where the divider stands and the clearance above it. | A worker transfers a case over the raised lip separating two cleanroom counters. |

## Moving obstacles and receivers

| Task and registry ID | Scored operation | Geometry to remember | Real-life example |
| --- | --- | --- | --- |
| **25. Yield in a cart refuge**<br>`spatial_memory_cart_refuge` | Yield inside an inspected passing recess as the cart goes by; then rejoin the aisle. | The first walking direction to the fixed refuge. The cart moves independently in the adjacent lane. | A porter steps into a corridor recess to let a linen cart pass. |
| **26. Skirt a swinging boom**<br>`spatial_memory_swinging_boom` | Round the outside of a sweeping boom and reach the service side of its pedestal. | The bearing toward the safe side of the pivot and the space clear of the moving boom. | A ground-crew member walks around the sweep of a pivoting baggage arm. |
| **27. Cross under a lift gate**<br>`spatial_memory_lift_gate` | Stage at the remembered portal, cross beneath the raised shutter and clear its exit. | The fixed portal center between the posts. Live shutter height determines when to cross. | A depot worker crosses a loading portal when its overhead shutter is high enough. |
| **28. Escort a shuttle**<br>`spatial_memory_shuttle_escort` | Meet the shuttle beside its guide rail, escort it past the midpoint and leave at the far berth. | The guide-rail alignment and exit berth. The robot must find the shuttle at three points along its travel. | A factory attendant walks beside a slow inspection shuttle between two work bays. |
| **29. Load a moving receiver**<br>`spatial_memory_moving_receiver` | Lift the parcel from the fixed bench and release it on the moving receiver near the dispatch end. | The fixed pickup bench and its approach; the receiving platform has an independently moving position. | A packer loads a parcel onto a slow transfer carriage. |
| **30. Cross a bollard chicane**<br>`spatial_memory_bollard_chicane` | Negotiate two offset retracting bollards. Cross each footprint only when that bollard is down. | The side of the divider used on approach; the two bollard sockets have different live clearance windows. | A yard inspector crosses a staggered pair of retractable vehicle barriers. |

## Export all 30 previews

The preview utility resets each selected environment and writes its G1 head-camera image and sampled fixture geometry to the output directory. Pass one or more registry IDs with `--tasks` to export a subset.

```bash
isaaclab.sh -p isaaclab_arena_examples/spatial_memory/preview_loco_tasks.py \
    --output /tmp/g1-loco-scenes
```

## Reproduce a scene

From the repository’s supported Arena runtime, build a task with the environment runner. Substitute any registry ID above.

```bash
isaaclab.sh -p isaaclab_arena/scripts/environment_runner.py --enable_cameras --num_envs 1 \
    spatial_memory_service_trench --observation_variant mapped
```
