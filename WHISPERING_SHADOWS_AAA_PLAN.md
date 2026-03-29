# WHISPERING SHADOWS – AAA Mobile Horror Production Plan

This file converts the requested design direction into an implementation-ready production blueprint for a full Unity/URP mobile title.

## 1) Product definition
- **Title:** Whispering Shadows
- **Platform:** iOS + Android (high-end profile + scalable quality presets)
- **Engine target:** Unity 2022.3 LTS + URP
- **Target package size:** 350–400 MB
- **Core loop:** Explore → solve puzzle → avoid ghost AI → collect ritual artifacts → choose ending.

## 2) Scope tiers
To make delivery realistic, development is split into milestones:

### Vertical Slice (8–12 weeks)
- 1 floor of mansion (foyer, kitchen, chapel, 1 bedroom)
- 1 major puzzle chain + 1 alternate route
- Ghost AI states (patrol/search/hunt/lost)
- Core controls (joystick, look, crouch, sprint, inventory)
- 1 ending path

### Alpha (4–6 months)
- 20+ rooms, basement + attic + garden
- Day/night escalation, sanity system, full inventory interactions
- 3 ending routes + persistence

### Beta/Release (6–12+ months)
- Full voice acting, polished AI behavior tree, optimization pass, localization, QA hardening

## 3) System architecture

### Gameplay systems
- `PlayerController` (movement, crouch, sprint, camera, stamina)
- `InteractionSystem` (raycast interaction + contextual actions)
- `InventorySystem` (6 slots, combinable items, radial UI)
- `GhostDirector` (difficulty, day progression, events)
- `SpecterAI` (FSM + senses + anchors + hunt lock-down)
- `SanitySystem` (state, effects, audio/post-processing hooks)
- `PuzzleStateMachine` (node graph for puzzle dependencies)
- `SaveManager` (binary state + checkpoint snapshots)

### Data-driven content
Use ScriptableObjects for:
- item definitions
- room metadata
- puzzle nodes
- audio events
- ghost behavior tuning profiles by day

## 4) AI requirements (implementation detail)
- Perception:
  - hearing (event bus with sound intensity/range)
  - sight cone with LOS checks
  - memory of last known player position
- State machine:
  - patrol -> search -> hunt -> lost
- Night-only abilities:
  - anchor teleport
  - shadow dash cooldown
- Anti-exploit logic:
  - repeated hiding spot checks
  - predictive pathing based on player habits

## 5) Environment production package
- 3-story manor + basement + attic + exterior
- room kits: wall/floor/ceiling modules
- hero props: staircase, chapel altar, grandfather clock
- interactable props: closets, beds, drawers, doors, switch panels
- hidden passages and puzzle-specific moving geometry

## 6) Audio direction
- FMOD event buses:
  - ambience
  - ghost proximity layers
  - chase music stem switching
  - sanity distortion bus
- voice sets:
  - Specter (200+ lines)
  - Elias logs (50+ lines)

## 7) Performance plan
- mobile tiers: low / medium / high
- ASTC texture compression
- baked GI + selective dynamic lights
- occlusion culling + LOD + impostors
- dynamic resolution scaler for thermal throttling

## 8) Milestone acceptance criteria

### Vertical Slice done when:
- 30 FPS stable on target minimum device profile
- One full run from spawn to ritual completion path
- Ghost can complete full patrol/search/hunt loop without nav failures
- Save/load works across app restart

### Release done when:
- all endings playable
- no blocker crashes across device test matrix
- complete localization/subtitle coverage
- audio mix pass and accessibility verification complete

## 9) Repo status note
Current repository contains a **prototype web implementation** and Android APK wrapping workflow.
This document is the production-grade roadmap to transition into a true Unity AAA pipeline.
